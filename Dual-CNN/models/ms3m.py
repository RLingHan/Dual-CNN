import torch
import torch.nn as nn
import torch.nn.functional as F


class MS3M(nn.Module):
    def __init__(self, in_channels, reduction=16, scales=[3, 5, 7]):
        super().__init__()
        self.scales = scales
        mid_channels = max(in_channels // reduction, 32)

        self.IN = nn.InstanceNorm2d(in_channels, track_running_stats=False)

        self.delta_net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 新增：由delta动态生成尺度权重
        # 输入是全局delta均值[B, C] -> 输出是[B, num_scales]
        self.scale_selector = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(scales)),
            nn.Softmax(dim=1)
        )

        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels,
                      kernel_size=k, padding=k // 2,
                      groups=in_channels, bias=False)
            for k in scales
        ])

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_sh, f_sp):
        B, C, H, W = f_sh.size()

        # Step1: 生成delta
        f_sp_in = self.IN(f_sp)
        style_res = f_sp - f_sp_in
        delta = self.delta_net(style_res)  # [B, C, H, W]

        # Step2: 用delta的全局均值动态决定尺度权重
        delta_gap = delta.mean(dim=[2, 3])  # [B, C] 全局模态差异强度
        scale_w = self.scale_selector(delta_gap)  # [B, num_scales] 每个样本不同权重

        # Step3: 垂直方向序列化
        x_col = f_sh.mean(dim=3)  # [B, C, H]

        # Step4: 多尺度Conv1d，权重由delta动态决定
        scale_outputs = torch.stack(
            [conv(x_col) for conv in self.scale_convs], dim=1
        )  # [B, num_scales, C, H]

        # scale_w: [B, num_scales] -> [B, num_scales, 1, 1]
        scale_w = scale_w.unsqueeze(-1).unsqueeze(-1)
        x_multi = (scale_outputs * scale_w).sum(dim=1)  # [B, C, H]

        # Step5: 扩展回空间 + delta调制
        x_multi = x_multi.unsqueeze(-1).expand(B, C, H, W)
        delta_expand = delta
        x_out = x_multi * delta_expand + f_sh * (1 - delta_expand)

        # Step6: 输出投影 + 残差
        x_out = self.out_proj(x_out)
        f_sh_enhanced = self.gamma * x_out + f_sh

        return f_sh_enhanced, f_sp

