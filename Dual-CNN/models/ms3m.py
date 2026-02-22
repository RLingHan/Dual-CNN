import torch
import torch.nn as nn
import torch.nn.functional as F


class MS3M(nn.Module):
    """
    Modality-Specific Selective State Module (MS3M)

    核心思想:
        用 f_sp 的模态风格残差生成位置自适应选择性信号 Δ,
        驱动 f_sh 在空间序列上的多尺度状态传播强度,
        实现模态差异感知的动态全局建模。

    输入:
        f_sh: [B, C, H, W]  模态无关特征 (来自 MUM 的 x_sh3 * m_sh)
        f_sp: [B, C, H, W]  模态特有特征 (来自 MUM 的 x_sh3 * m_sp)
    输出:
        f_sh_enhanced: [B, C, H, W]  增强后的模态无关特征
        f_sp:          [B, C, H, W]  原样返回, 保持后续流程不变
    """

    def __init__(self, in_channels: int, reduction: int = 16, scales: list = None):
        super(MS3M, self).__init__()

        self.in_channels = in_channels
        # 多尺度 kernel, 模拟不同范围的状态传播
        self.scales = scales if scales is not None else [3, 7, 15]
        mid_channels = max(in_channels // reduction, 32)

        # ─── Step 1: 选择性信号 Δ 生成网络 ───────────────────────────
        # 从 f_sp 的风格残差提取位置自适应门控
        self.IN = nn.InstanceNorm2d(in_channels, track_running_stats=False)

        # 风格残差 → 通道压缩 → 空间保留的 Δ
        self.delta_net = nn.Sequential(
            # 通道压缩, 保留空间维度
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 空间感知 (3×3 捕捉局部风格结构)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 恢复到 in_channels, 输出空间自适应 Δ
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # Δ ∈ (0, 1), 控制状态混合强度
        )

        # ─── Step 2 & 3: 多尺度 depthwise Conv1d 状态传播 ────────────
        # 每个尺度独立的 depthwise Conv1d (沿 L=H*W 方向)
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels, in_channels,
                kernel_size=k,
                padding=k // 2,
                groups=in_channels,  # depthwise: 每通道独立
                bias=False
            )
            for k in self.scales
        ])

        # 多尺度融合的可学习权重 (softmax 归一化)
        self.scale_weights = nn.Parameter(
            torch.ones(len(self.scales)) / len(self.scales)
        )

        # ─── Step 4: 输出投影 ─────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # 残差权重, 初始 0 保证训练早期稳定
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f_sh, f_sp):
        """
        Args:
            f_sh: [B, C, H, W]
            f_sp: [B, C, H, W]
        Returns:
            f_sh_enhanced: [B, C, H, W]
            f_sp:          [B, C, H, W]  不变
        """
        B, C, H, W = f_sh.size()
        L = H * W

        # ══════════════════════════════════════════════════════
        # Step 1: 从 f_sp 提取选择性信号 Δ
        # ══════════════════════════════════════════════════════
        # 实例归一化残差 = 模态风格信息
        f_sp_in   = self.IN(f_sp)               # 去风格版本
        style_res = f_sp - f_sp_in              # 纯风格残差 [B, C, H, W]

        # 生成空间自适应 Δ: 模态差异大的位置值大
        delta = self.delta_net(style_res)        # [B, C, H, W], ∈ (0,1)

        # ══════════════════════════════════════════════════════
        # Step 2: f_sh 序列化 + 多尺度状态传播
        # ══════════════════════════════════════════════════════
        # 展平空间维度, 沿序列方向做 Conv1d
        x_seq = f_sh.view(B, C, L)              # [B, C, L]

        # 多尺度 depthwise Conv1d, 全并行
        scale_w = F.softmax(self.scale_weights, dim=0)  # 归一化融合权重
        x_multi = sum(
            scale_w[i] * conv(x_seq)
            for i, conv in enumerate(self.scale_convs)
        )                                        # [B, C, L]

        # ══════════════════════════════════════════════════════
        # Step 3: Δ 驱动的选择性调制
        # ══════════════════════════════════════════════════════
        delta_seq = delta.view(B, C, L)          # [B, C, L]

        # 选择性混合: Δ 大 → 状态传播强 → 依赖全局上下文
        #            Δ 小 → 保持原始特征 → 局部特征可靠
        x_out = x_multi * delta_seq + x_seq * (1 - delta_seq)  # [B, C, L]

        # ══════════════════════════════════════════════════════
        # Step 4: 输出投影 + 残差
        # ══════════════════════════════════════════════════════
        x_out = x_out.view(B, C, H, W)          # reshape 回空间
        x_out = self.out_proj(x_out)             # [B, C, H, W]

        f_sh_enhanced = self.gamma * x_out + f_sh  # 残差连接

        return f_sh_enhanced, f_sp , self.gamma

