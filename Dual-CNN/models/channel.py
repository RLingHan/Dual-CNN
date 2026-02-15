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

class FeatureDecomposition(nn.Module):
    """将特征分解为模态共享和模态特定"""

    def __init__(self, dim, reduction=16):
        super().__init__()

        # 学习分解mask
        self.shared_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 生成共享/特定mask
        shared_mask = self.shared_gate(x)  # (B, C, 1, 1)
        specific_mask = 1 - shared_mask

        feat_shared = x * shared_mask
        feat_specific = x * specific_mask

        return feat_shared, feat_specific

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

    def __init__(self, channels, kernel_size=7):
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