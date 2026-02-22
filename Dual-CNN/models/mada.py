import torch
import torch.nn as nn
import torch.nn.functional as F


class PartSoftmaxAttention(nn.Module):
    """
    轻量化 Part Softmax Attention

    流程:
        [B, C, H, W]
          ↓ 1×1 Conv  (C → num_parts)
        [B, num_parts, H, W]
          ↓ 对每个 part 通道做 softmax 空间注意力
            (展平空间维度 → softmax → 加权回自身)
        [B, num_parts, H, W]
          ↓ 1×1 Conv  (num_parts → C)
        [B, C, H, W]  + 残差

    接口与原 ModalityAwareDualAttention 完全兼容:
        forward(x, modality=None)
    """

    def __init__(self, in_channels: int, num_parts: int = 3):
        super(PartSoftmaxAttention, self).__init__()

        self.in_channels = in_channels
        self.num_parts   = num_parts

        # 压缩: C → num_parts
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, num_parts, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_parts),
            nn.ReLU(inplace=True)
        )
        # 恢复: num_parts → C
        self.expand = nn.Sequential(
            nn.Conv2d(num_parts, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x        : [B, C, H, W]
        Returns:
            out      : [B, C, H, W]
        """
        B, C, H, W = x.size()
        identity = x
        # 1. 压缩到 num_parts 通道
        compressed = self.compress(x)                            # [B, num_parts, H, W]
        # 2. 每个 part 通道独立做 softmax 空间注意力
        flat  = compressed.view(B, self.num_parts, -1)           # [B, num_parts, H*W]
        attn  = F.softmax(flat, dim=-1)                          # 空间维度归一化
        attended = (flat * attn).view(B, self.num_parts, H, W)   # [B, num_parts, H, W]
        # 3. 恢复到原始通道数
        out = self.expand(attended)                              # [B, C, H, W]
        # 4. 残差
        out = out + identity

        return out