from torch import nn
import torch
import torch.nn.functional as F


class MUMModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(MUMModule, self).__init__()
        # 通道级判别器：输入 GAP 后的特征 [B, C]
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            # nn.Sigmoid()  # 输出 P_c: 每个通道属于 RGB 的概率
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape

        # 1. 全局平均池化得到通道描述符
        feat_gap = F.avg_pool2d(x, (h, w)).view(b, c)

        # 2. 预测模态概率 P
        P_logits = self.discriminator(feat_gap)  # [B, C] 未经Sigmoid

        # 生成mask时才用Sigmoid
        P = torch.sigmoid(P_logits)  # [B, C] ∈ [0,1]

        # 3. 生成软掩码 (Soft Mask)
        # Shared Mask: 采用三角波函数，在 0.5 处取得极大值 1
        mask_sh = 1 - torch.abs(2 * P - 1)

        # Specific Mask: 在 0 或 1 处取得极大值 1
        mask_sp = torch.abs(2 * P - 1)

        # 4. 调整维度以便与特征图相乘 [B, C, 1, 1]
        mask_sh = mask_sh.view(b, c, 1, 1)
        mask_sp = mask_sp.view(b, c, 1, 1)

        return mask_sh, mask_sp, P_logits

# Identity-Specific Compensation Module
class ISCM(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()

        # 1. 通道注意力 (SE-like)
        self.id_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 4, in_dim, 1),
            nn.Sigmoid()
        )
        # 2. 融合门控 (Gating Parameter)
        self.beta = nn.Parameter(torch.tensor(0.0))
        # 3. Instance Normalization
        # affine=False 很重要，我们要强制去除风格统计量
        self.in_layer = nn.InstanceNorm2d(in_dim, affine=False)

    def forward(self, f_sh, f_sp, labels=None):
        """
        f_sh: 经过 mask_sh 的特征 (Batch, C, H, W)
        f_sp: 经过 mask_sp 的特征 (Batch, C, H, W)
        """
        # --- 步骤 A: 提取有用的 Specific 信息 (Mining) ---
        # 1. 计算 Attention (即寻找含有 ID 信息的通道)
        attn_weights = self.id_attention(f_sp)  # [B, C, 1, 1]
        # 2. 加权筛选
        f_sp_id = f_sp * attn_weights
        # --- 步骤 B: 风格剥离 (Style Stripping via IN) ---
        # 使用 IN 去除模态特定的均值和方差 (即去除了 RGB/IR 的光照/热辐射风格)
        # 剩下的 f_sp_stripped 理论上只包含结构/形状信息
        f_sp_id = self.in_layer(f_sp_id)
        # --- 步骤 C: 补偿融合 (Compensation) ---
        # f_enhanced = Shared + alpha * Normalize(Specific)
        f_sh_enhanced = f_sh + self.beta * f_sp_id
        return f_sh_enhanced, f_sp_id

class GlobalContextBlock(nn.Module):
    """全局上下文注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # 全局上下文建模
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # 通道变换
        self.channel_add = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 全局上下文聚合
        context_mask = self.conv_mask(x)  # (B, 1, H, W)
        context_mask = context_mask.view(B, 1, H * W)
        context_mask = self.softmax(context_mask)  # (B, 1, H*W)

        context = torch.matmul(
            x.view(B, C, H * W),
            context_mask.transpose(1, 2)
        )  # (B, C, 1)
        context = context.unsqueeze(-1)  # (B, C, 1, 1)

        # 通道增强
        transform = self.channel_add(context)
        return x + transform


class LargeKernelAttention(nn.Module):
    """大核注意力"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()

        # 深度可分离大核卷积
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels
        )

        # 1×1逐点卷积
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 大感受野特征
        attn = self.dw_conv(x)
        attn = self.pw_conv(attn)

        # 空间门控
        gate = self.spatial_gate(attn)

        return x * gate + x


class AdaptiveGlobalModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_attn = GlobalContextBlock(channels)
        self.lka = LargeKernelAttention(channels)

        # 门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

        # 缩放因子
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 全局特征
        global_feat = self.global_attn(x)
        global_feat = self.lka(global_feat)

        # 自适应门控
        gate = self.gate(x)

        # 小权重残差
        out = x + self.scale * gate * global_feat
        return out