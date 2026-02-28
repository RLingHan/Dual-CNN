import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalAdaptiveIN(nn.Module):
    """
    条件自适应归一化：
    针对每种模态学习独立的 scale/shift，
    消除模态特有风格，保留跨模态稳定的结构信息。
    """
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(channels, affine=False)
        self.modal_scale = nn.Embedding(2, channels)
        self.modal_shift = nn.Embedding(2, channels)
        nn.init.ones_(self.modal_scale.weight)
        nn.init.zeros_(self.modal_shift.weight)

    def forward(self, x, sub):
        sub = sub.long()  # Bool → Long，Embedding需要整数索引
        x_norm = self.in_norm(x)
        scale = self.modal_scale(sub).view(-1, x.size(1), 1, 1)
        shift = self.modal_shift(sub).view(-1, x.size(1), 1, 1)
        return x_norm * scale + shift


class MultiGranularitySkip(nn.Module):
    """
    轻量多粒度门控跳跃连接。

    防过拟合设计：
        1. gate_fc 最后一层 bias 初始化为 -2.0
           → sigmoid(-2) ≈ 0.12，训练初期gate很小，主路稳定
        2. skip_dropout(p=0.2) 作用于两路跳跃特征
           → 防止网络把x2/f_sh当成捷径，强迫主路保持判别能力
        3. 两路独立门控，网络自适应决定各粒度的贡献比例

    参数量约 1.6M，占 ResNet50 的 ~6.3%
    """

    def __init__(self,
                 x2_channels=512,
                 fsh_channels=1024,
                 out_channels=2048,
                 inner_dim=256,
                 dropout_p=0.2):
        super().__init__()

        # ── x2 分支：模态提纯 + 压缩 + 对齐 ──────────────────
        self.modal_adaptive_in = ModalAdaptiveIN(x2_channels)
        self.proj_x2 = nn.Sequential(
            nn.Conv2d(x2_channels, inner_dim, 1, bias=False),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=True),
        )
        self.align_x2 = nn.Sequential(
            nn.Conv2d(inner_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── f_sh 分支：压缩 + 对齐 ────────────────────────────
        self.proj_fsh = nn.Sequential(
            nn.Conv2d(fsh_channels, inner_dim, 1, bias=False),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=True),
        )
        self.align_fsh = nn.Sequential(
            nn.Conv2d(inner_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── Dropout：防止过度依赖跳跃路径 ─────────────────────
        self.skip_dropout = nn.Dropout(p=dropout_p)

        # ── 门控：预测两路各自的注入强度 ──────────────────────
        gate_in = out_channels + inner_dim * 2  # 2048+256+256 = 2560
        self.gate_fc = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        # 关键：最后一层 bias 初始化为 -2.0
        # sigmoid(-2) ≈ 0.12，训练初期 gate 很小，不破坏预训练特征
        nn.init.constant_(self.gate_fc[-2].bias, -2.0)

    def forward(self, x2, f_sh, x_sh4, sub):
        """
        Args:
            x2   : (B, 512,  H,   W  )  Layer2 输出
            f_sh : (B, 1024, H/2, W/2)  MDIA 的 f_out
            x_sh4: (B, 2048, H/4, W/4)  Layer4 输出，主路
            sub  : (B,) 模态标签 0=RGB 1=IR
        Returns:
            x_sh4_enhanced: (B, 2048, H/4, W/4)
            gates          : (B, 2)  [gate_x2, gate_fsh]，可打印观察
        """
        target_size = x_sh4.shape[2:]
        dtype = x_sh4.dtype

        # ── x2 分支 ───────────────────────────────────────────
        x2_clean   = self.modal_adaptive_in(x2, sub)
        x2_small   = self.proj_x2(x2_clean)
        x2_down    = F.adaptive_avg_pool2d(x2_small, target_size)
        x2_aligned = self.align_x2(x2_down)
        x2_aligned = self.skip_dropout(x2_aligned.float()).to(dtype)  # dropout

        # ── f_sh 分支 ─────────────────────────────────────────
        fsh_small   = self.proj_fsh(f_sh)
        fsh_down    = F.adaptive_avg_pool2d(fsh_small, target_size)
        fsh_aligned = self.align_fsh(fsh_down)
        fsh_aligned = self.skip_dropout(fsh_aligned.float()).to(dtype)  # dropout

        # ── 门控预测 ──────────────────────────────────────────
        x4_gap  = F.adaptive_avg_pool2d(x_sh4,    1).flatten(1).float()
        x2_gap  = F.adaptive_avg_pool2d(x2_down,  1).flatten(1).float()
        fsh_gap = F.adaptive_avg_pool2d(fsh_down, 1).flatten(1).float()

        gates    = self.gate_fc(torch.cat([x4_gap, x2_gap, fsh_gap], dim=1))
        gate_x2  = gates[:, 0].view(-1, 1, 1, 1).to(dtype)
        gate_fsh = gates[:, 1].view(-1, 1, 1, 1).to(dtype)

        # ── 加权残差融合 ──────────────────────────────────────
        x_sh4_enhanced = (x_sh4
                          + gate_x2  * x2_aligned
                          + gate_fsh * fsh_aligned)

        return x_sh4_enhanced, gates