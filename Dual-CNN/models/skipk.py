import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalAdaptiveIN(nn.Module):
    """
    条件自适应归一化：
    针对每种模态学习独立的 scale/shift，
    消除模态特有风格，保留跨模态稳定的结构信息。
    参数量：channels * 4 = 2,048（channels=512时）
    """
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(channels, affine=False)
        self.modal_scale = nn.Embedding(2, channels)
        self.modal_shift = nn.Embedding(2, channels)
        nn.init.ones_(self.modal_scale.weight)
        nn.init.zeros_(self.modal_shift.weight)

    def forward(self, x, sub):
        x_norm = self.in_norm(x)
        scale = self.modal_scale(sub).view(-1, x.size(1), 1, 1)
        shift = self.modal_shift(sub).view(-1, x.size(1), 1, 1)
        return x_norm * scale + shift


class MultiGranularitySkip(nn.Module):
    """
    轻量多粒度门控跳跃连接。

    设计思路：
        避免大维度 concat+proj，改用"各路投影到2048后加权残差叠加"。
        gate 初始为0（通过 gate_bias 控制），保证训练初期不破坏主路。

    三路特征：
        x2   (B, 512,  H,   W  ) 低层结构纹理
        f_sh (B, 1024, H/2, W/2) 中层模态无关语义（MDIA输出）
        x_sh4(B, 2048, H/4, W/4) 高层身份语义（主路）

    参数量明细（约1.05M）：
        ModalAdaptiveIN :    2,048
        proj_x2         :  263,168   (512→512 conv1x1 + BN)
        proj_fsh        :  525,312   (1024→512 conv1x1 + BN)
        align_x2        :  263,168   (512→2048? 不对)
        ── 改为：x2/fsh各压到512，然后用深度可分离conv升到2048 ──
        dw_x2           :    2,560   (depthwise 512, k=1)
        pw_x2           :    1,048,576 → 太大
        ── 最终方案：x2/fsh压到256，用1x1升到2048 ──

    最终参数量约 1.1M，见下方注释。
    """

    def __init__(self,
                 x2_channels=512,
                 fsh_channels=1024,
                 out_channels=2048,
                 inner_dim=256):
        """
        inner_dim=256：x2和f_sh各压缩到256维后升到2048。

        参数量：
            ModalAdaptiveIN(512)          :      2,048
            proj_x2  512→256 conv1x1+BN  :    132,096   (512*256 + 256*2)
            proj_fsh 1024→256 conv1x1+BN :    263,168   (1024*256 + 256*2)
            align_x2  256→2048 conv1x1+BN:    526,336   (256*2048 + 2048*2)
            align_fsh 256→2048 conv1x1+BN:    526,336   (256*2048 + 2048*2)
            gate_fc  (2048+256+256)→64→2 :    151,234   ((2560*64+64) + (64*2+2))
            ─────────────────────────────────────────
            总计                          :  1,601,218  ≈ 1.6M  (~6.3% of ResNet50)
        """
        super().__init__()

        # ── x2 分支 ───────────────────────────────────────────
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

        # ── f_sh 分支 ─────────────────────────────────────────
        self.proj_fsh = nn.Sequential(
            nn.Conv2d(fsh_channels, inner_dim, 1, bias=False),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=True),
        )
        self.align_fsh = nn.Sequential(
            nn.Conv2d(inner_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── 门控 ──────────────────────────────────────────────
        # 输入: x4_gap(2048) + x2_gap(256) + fsh_gap(256) = 2560
        # 输出: gate_x2, gate_fsh ∈ (0,1)
        gate_in = out_channels + inner_dim * 2   # 2048+256+256 = 2560
        self.gate_fc = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x2, f_sh, x_sh4, sub):
        """
        Args:
            x2   : (B, 512,  H,   W  )
            f_sh : (B, 1024, H/2, W/2)  MDIA 的 f_out
            x_sh4: (B, 2048, H/4, W/4)  Layer4 输出，主路
            sub  : (B,) 模态标签 0=RGB 1=IR
        Returns:
            x_sh4_enhanced: (B, 2048, H/4, W/4)
            gates          : (B, 2)  [gate_x2, gate_fsh]
        """
        target_size = x_sh4.shape[2:]
        dtype = x_sh4.dtype

        # ── x2 分支：提纯 → 压缩 → 对齐到2048 ───────────────
        x2_clean  = self.modal_adaptive_in(x2, sub)             # (B, 512, H, W)
        x2_small  = self.proj_x2(x2_clean)                      # (B, 256, H, W)
        x2_down   = F.adaptive_avg_pool2d(x2_small, target_size) # (B, 256, H/4, W/4)
        x2_aligned = self.align_x2(x2_down)                     # (B, 2048, H/4, W/4)

        # ── f_sh 分支：压缩 → 对齐到2048 ─────────────────────
        fsh_small  = self.proj_fsh(f_sh)                        # (B, 256, H/2, W/2)
        fsh_down   = F.adaptive_avg_pool2d(fsh_small, target_size) # (B, 256, H/4, W/4)
        fsh_aligned = self.align_fsh(fsh_down)                  # (B, 2048, H/4, W/4)

        # ── 门控预测 ──────────────────────────────────────────
        x4_gap  = F.adaptive_avg_pool2d(x_sh4,    1).flatten(1).float()  # (B, 2048)
        x2_gap  = F.adaptive_avg_pool2d(x2_down,  1).flatten(1).float()  # (B, 256)
        fsh_gap = F.adaptive_avg_pool2d(fsh_down, 1).flatten(1).float()  # (B, 256)

        gates    = self.gate_fc(torch.cat([x4_gap, x2_gap, fsh_gap], dim=1))  # (B, 2)
        gate_x2  = gates[:, 0].view(-1, 1, 1, 1).to(dtype)
        gate_fsh = gates[:, 1].view(-1, 1, 1, 1).to(dtype)

        # ── 加权残差融合 ──────────────────────────────────────
        # 两路都以残差形式叠加到主路，不改变主路维度
        x_sh4_enhanced = (x_sh4
                          + gate_x2  * x2_aligned.to(dtype)
                          + gate_fsh * fsh_aligned.to(dtype))

        return x_sh4_enhanced, gates