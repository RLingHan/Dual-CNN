import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionAttention(nn.Module):
    """
    Position Attention Module (PAM) from DANet
    捕获空间位置之间的长距离依赖
    """

    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.in_channels = in_channels

        # 降维,减少计算量
        self.inter_channels = in_channels // 8

        # Query, Key, Value的投影
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        # 可学习的权重参数,初始化为0
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Query: [B, C', H*W] -> [B, H*W, C']
        proj_query = self.query_conv(x).view(B, self.inter_channels, -1).permute(0, 2, 1)

        # Key: [B, C', H*W]
        proj_key = self.key_conv(x).view(B, self.inter_channels, -1)

        # Attention map: [B, H*W, H*W]
        # 每个位置对所有位置的注意力分数
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Value: [B, C, H*W]
        proj_value = self.value_conv(x).view(B, C, -1)

        # Output: [B, C, H*W] -> [B, C, H, W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


# ===== 新增: Channel Attention for DANet (不同于CBAM的CA) =====
class ChannelAttentionDANet(nn.Module):
    """
    Channel Attention Module (CAM) from DANet
    捕获通道之间的相互依赖关系
    """

    def __init__(self, in_channels):
        super(ChannelAttentionDANet, self).__init__()
        self.in_channels = in_channels

        # 可学习的权重参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Reshape: [B, C, H*W]
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)

        # Channel attention map: [B, C, C]
        energy = torch.bmm(proj_query, proj_key)

        # 使用max而不是直接softmax,效果更好
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # Value
        proj_value = x.view(B, C, -1)

        # Output: [B, C, H*W] -> [B, C, H, W]
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)

        # Residual connection
        out = self.gamma * out + x

        return out


# ===== 主模块: Modality-Aware Dual Attention =====
class ModalityAwareDualAttention(nn.Module):
    """
    Modality-Aware Dual Attention (MADA) Module

    创新点:
    1. Part-based分解: 水平分成num_parts个部分,适合行人ReID
    2. Dual Attention: 对每个part同时应用Position和Channel注意力
    3. Modality-aware Gating: 根据模态(RGB/IR)自适应调整每个part的权重

    优势:
    - 适合VI-ReID: 不同身体部位的模态差异不同
    - 头部: RGB清晰,IR模糊 -> 给RGB高权重
    - 躯干: 两者都可以 -> 平衡权重
    - 腿部: IR可能更好 -> 给IR高权重
    """

    def __init__(self, in_channels, num_parts=3):
        super(ModalityAwareDualAttention, self).__init__()

        self.in_channels = in_channels
        self.num_parts = num_parts

        # 每个part一个独立的dual attention模块
        self.part_modules = nn.ModuleList([
            nn.ModuleDict({
                'pa': PositionAttention(in_channels),  # 空间注意力
                'ca': ChannelAttentionDANet(in_channels),  # 通道注意力
            }) for _ in range(num_parts)
        ])

        # 模态感知门控网络
        # 输入: 模态标签(0或1)
        # 输出: 每个part的权重 [num_parts]
        self.modality_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_parts),
            nn.Sigmoid()  # 输出0-1之间的权重
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modality_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, modality):
        """
        Args:
            x: [B, C, H, W] 输入特征图
            modality: [B] 模态标签 (0=RGB, 1=IR)

        Returns:
            out: [B, C, H, W] 增强后的特征图
        """
        B, C, H, W = x.size()

        # 每个part的高度
        part_h = H // self.num_parts

        # 计算模态门控权重 [B, num_parts]
        mod_weights = self.modality_gate(modality.float().unsqueeze(1))

        outputs = []
        for i, part_module in enumerate(self.part_modules):
            # 1. 提取当前part
            start_h = i * part_h
            end_h = (i + 1) * part_h if i < self.num_parts - 1 else H  # 最后一个part包含剩余
            part = x[:, :, start_h:end_h, :]  # [B, C, part_h, W]

            # 2. 应用Dual Attention
            pa_out = part_module['pa'](part)  # Position attention
            ca_out = part_module['ca'](part)  # Channel attention
            part_out = pa_out + ca_out  # 融合

            # 3. 模态自适应调制
            # weight: [B, 1, 1, 1] 每个样本的权重不同
            weight = mod_weights[:, i].view(B, 1, 1, 1)

            # 线性插值: part * (1-w) + part_out * w
            # - weight接近0: 保持原始特征 (该part不需要增强)
            # - weight接近1: 使用增强特征 (该part需要增强)
            part_final = part * (1 - weight) + part_out * weight

            outputs.append(part_final)

        # 4. 拼接所有part
        out = torch.cat(outputs, dim=2)  # [B, C, H, W]

        return out

