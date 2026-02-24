from torch import nn
import torch
import torch.nn.functional as F


class MUMModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(MUMModule, self).__init__()

        # 通道注意力（原有）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
        )

        # 空间注意力（新增）：判断哪些位置是模态相关的
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 通道维度的模态概率
        P_logits = self.channel_attn(x)  # [B, C]
        P = torch.sigmoid(P_logits)

        mask_sh_c = (1 - torch.abs(2 * P - 1)).view(b, c, 1, 1)
        mask_sp_c = torch.abs(2 * P - 1).view(b, c, 1, 1)

        # 空间维度的模态权重
        sp_map = torch.sigmoid(self.spatial_attn(x))  # [B, 1, H, W]

        # 联合 mask：通道 × 空间
        mask_sh = mask_sh_c * (1 - sp_map)  # 模态无关：通道无关 且 空间无关
        mask_sp = mask_sp_c * sp_map  # 模态特有：通道相关 且 空间相关

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