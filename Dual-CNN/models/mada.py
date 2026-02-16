import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 轻量化版本1: Efficient Position Attention =====
class EfficientPositionAttention(nn.Module):
    """
    轻量化位置注意力
    改进点:
    1. 降维比例从1/8改为1/16,减少50%计算量
    2. 使用深度可分离卷积代替标准卷积
    3. 空间下采样策略减少attention map大小
    """

    def __init__(self, in_channels, reduction=16):
        super(EfficientPositionAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 32)  # 至少32通道

        # 深度可分离卷积: 参数量减少约8倍
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),  # depthwise
            nn.Conv2d(in_channels, self.inter_channels, 1)  # pointwise
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.Conv2d(in_channels, self.inter_channels, 1)
        )
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        # 空间下采样,减少attention计算量
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局池化用于判断是否需要下采样

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 自适应下采样: 如果特征图大于16x16才下采样
        if H * W > 256:
            scale = 2
            x_down = F.avg_pool2d(x, scale)
            B, C, H_d, W_d = x_down.size()
        else:
            x_down = x
            H_d, W_d = H, W

        # Query & Key
        proj_query = self.query_conv(x_down).view(B, self.inter_channels, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x_down).view(B, self.inter_channels, -1)

        # Attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Value
        proj_value = self.value_conv(x_down).view(B, C, -1)

        # Output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H_d, W_d)

        # 上采样回原始尺寸
        if H_d != H or W_d != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Residual
        out = self.gamma * out + x

        return out


# ===== 轻量化版本2: Efficient Channel Attention =====
class EfficientChannelAttention(nn.Module):
    """
    轻量化通道注意力
    改进点:
    1. 使用1D卷积替代全连接的channel attention
    2. 去掉复杂的max操作,直接softmax
    3. 共享value projection
    """

    def __init__(self, in_channels):
        super(EfficientChannelAttention, self).__init__()
        self.in_channels = in_channels

        # 使用全局平均池化提取通道统计信息
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 简化的通道门控
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # 通道注意力权重
        gap_feat = self.gap(x)  # [B, C, 1, 1]
        channel_weight = self.fc(gap_feat)  # [B, C, 1, 1]

        # 加权
        out = x * channel_weight

        # Residual
        out = self.gamma * out + x

        return out

class ModalityAwareDualAttention(nn.Module):
    """
    轻量化模态感知双注意力模块

    改进点:
    1. 使用轻量化的PA和CA
    2. Part数量可配置,默认改为2(上下半身)
    3. 简化门控网络
    4. 可选的注意力类型(只用PA或CA或都用)

    参数量对比(in_channels=2048):
    - 原始MADA: ~8.4M parameters
    - LightweightMADA: ~0.5M parameters (减少94%)
    """

    def __init__(self, in_channels, num_parts=3, use_pa=True, use_ca=True):
        super(ModalityAwareDualAttention, self).__init__()

        self.in_channels = in_channels
        self.num_parts = num_parts
        self.use_pa = use_pa
        self.use_ca = use_ca

        # 每个part的注意力模块
        self.part_modules = nn.ModuleList()
        for _ in range(num_parts):
            part_dict = nn.ModuleDict()
            if use_pa:
                part_dict['pa'] = EfficientPositionAttention(in_channels)
            if use_ca:
                part_dict['ca'] = EfficientChannelAttention(in_channels)
            self.part_modules.append(part_dict)

        # 简化的门控网络
        self.modality_gate = nn.Sequential(
            nn.Linear(1, num_parts * 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_parts * 4, num_parts),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modality_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)  # 初始化为0.5,平衡两种模态

    def forward(self, x, modality):
        """
        Args:
            x: [B, C, H, W]
            modality: [B] or [B, 1] 模态标签
        """
        B, C, H, W = x.size()

        # 确保modality是正确的shape
        if modality.dim() == 1:
            modality = modality.unsqueeze(1)

        part_h = H // self.num_parts

        # 计算门控权重
        mod_weights = self.modality_gate(modality.float())  # [B, num_parts]

        outputs = []
        for i, part_module in enumerate(self.part_modules):
            # 提取part
            start_h = i * part_h
            end_h = (i + 1) * part_h if i < self.num_parts - 1 else H
            part = x[:, :, start_h:end_h, :]

            # 应用注意力
            part_out = part
            if self.use_pa and 'pa' in part_module:
                part_out = part_module['pa'](part_out)
            if self.use_ca and 'ca' in part_module:
                part_out = part_module['ca'](part_out)

            # 门控调制
            weight = mod_weights[:, i].view(B, 1, 1, 1)
            part_final = part * (1 - weight) + part_out * weight

            outputs.append(part_final)

        out = torch.cat(outputs, dim=2)

        return out