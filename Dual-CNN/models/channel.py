from torch import nn

#models/channel.py
class ChannelMaskGenerator(nn.Module):
    """生成三个通道掩码：M_v, M_i, M_s"""

    def __init__(self, dim=512, r=16):
        super(ChannelMaskGenerator, self).__init__()

        # 可见光掩码生成器
        self.mask_v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # 红外掩码生成器
        self.mask_i = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # 共享掩码生成器（直接用完整x2）
        self.mask_s = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_v, x_i, x_all):
        """
        x_v: 可见光特征 [B_v, C, H, W]
        x_i: 红外特征 [B_i, C, H, W]
        x_all: 完整x2 [B, C, H, W]
        """
        M_v = self.mask_v(x_v)  # [B_v, C, 1, 1]
        M_i = self.mask_i(x_i)  # [B_i, C, 1, 1]
        M_s = self.mask_s(x_all)  # [B, C, 1, 1] 直接用x2生成

        return M_v, M_i, M_s
