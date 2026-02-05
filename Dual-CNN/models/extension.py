# models/extension.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from einops import rearrange, repeat


# ==================== Mamba-2 版本的交叉注入模块 ====================

class SimplifiedMamba2Block_V2(nn.Module):
    """
    基于 Mamba-2 的简化版状态空间模型 + B矩阵交换机制

    核心改进（相比 Mamba-1）：
    1. 多头机制（类似 Transformer）
    2. 更大的状态维度（d_state=64）
    3. Mamba-2 的状态更新公式（dA, dBx）
    4. 保留 B 矩阵交换的核心创新
    5. chunk 分块处理（提升效率）

    相比完整 Mamba-2 的简化：
    - 不使用完整的 SSD 算法（保持代码清晰）
    - 保留原始的 B_override 交换机制
    """

    def __init__(self, d_model, d_state=64, headdim=64, expand_factor=2, chunk_size=16):
        super(SimplifiedMamba2Block_V2, self).__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)
        self.headdim = headdim
        self.chunk_size = chunk_size

        # 确保可以整除（多头机制）
        assert self.d_inner % headdim == 0, f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
        self.nheads = self.d_inner // headdim

        # ============ 输入投影 ============
        # Mamba-2 风格：z, x, B, C, dt
        d_in_proj = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # ============ SSM 参数 ============
        # A: 每个头一个参数（Mamba-2 风格）
        A_init = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))  # [nheads]
        self.A_log._no_weight_decay = True

        # dt_bias: 每个头的时间步长偏置
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.dt_bias._no_weight_decay = True

        # D: skip connection 参数（每个头）
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # ============ 输出投影 ============
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ============ 稳定性组件 ============
        self.norm_input = nn.LayerNorm(d_model)
        self.norm_output = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化"""
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

        # dt_bias 初始化为小的正值
        with torch.no_grad():
            self.dt_bias.copy_(torch.randn(self.nheads) * 0.01 + 0.1)

    def forward(self, x, B_override=None, C_override=None, freeze_params=False):
        """
        x: [B, L, D] - Batch, Length, Dimension
        B_override: [B, L, N] - 可选的外部B矩阵（用于交叉注入）
        C_override: [B, L, N] - 可选的外部C矩阵
        freeze_params: 是否冻结B/C（交叉时使用）

        返回: [B, L, D]
        """
        B_batch, L, D = x.shape

        # ============ 安全检查 ============
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️  WARNING: Invalid input to Mamba2! NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            return torch.zeros_like(x)

        # ============ 输入归一化 ============
        x_norm = self.norm_input(x)

        # ============ 输入投影（Mamba-2 风格）============
        # 投影得到: z (gate), x (input), B, C, dt
        zxbcdt = self.in_proj(x_norm)  # [B, L, d_in_proj]

        # 分割各个组件
        z, x_proj, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1
        )

        # 进一步分割 x_proj 得到 x, B, C
        x_ssm, B_generated, C_generated = torch.split(
            x_proj,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1
        )

        # ============ 使用外部B/C（交叉注入的关键）============
        B_to_use = B_override if B_override is not None else B_generated
        C_to_use = C_override if C_override is not None else C_generated

        if freeze_params and B_override is None:
            B_to_use = B_to_use.detach()
            C_to_use = C_to_use.detach()

        # ============ dt 处理（每个头独立的时间步长）============
        dt = F.softplus(dt + self.dt_bias.unsqueeze(0).unsqueeze(0))  # [B, L, nheads]
        dt = torch.clamp(dt, min=0.001, max=0.1)  # 限制范围

        # ============ A 矩阵（每个头一个标量）============
        A = -torch.exp(self.A_log.float())  # [nheads], 保证负数

        # ============ 重塑为多头形式 ============
        # x_ssm: [B, L, d_inner] -> [B, L, nheads, headdim]
        x_multi = rearrange(x_ssm, 'b l (h p) -> b l h p', h=self.nheads, p=self.headdim)

        # B, C 扩展head维度: [B, L, N] -> [B, L, 1, N] (广播到所有头)
        B_multi = B_to_use.unsqueeze(2)  # [B, L, 1, N]
        C_multi = C_to_use.unsqueeze(2)  # [B, L, 1, N]

        # dt 扩展: [B, L, nheads] -> [B, L, nheads, 1]
        dt_multi = dt.unsqueeze(-1)  # [B, L, nheads, 1]

        # ============ Chunk 分块处理 ============
        y = self._selective_scan_chunked(
            x_multi, A, B_multi, C_multi, dt_multi
        )  # [B, L, nheads, headdim]

        # 安全检查
        if torch.isnan(y).any():
            print("⚠️  WARNING: NaN in SSM output!")
            return torch.zeros_like(x)

        # ============ 合并多头 ============
        y = rearrange(y, 'b l h p -> b l (h p)')  # [B, L, d_inner]

        # ============ Skip Connection (D) ============
        # D 参数：每个头独立
        D_expanded = self.D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, nheads, 1]
        x_skip = x_multi * D_expanded  # [B, L, nheads, headdim]
        x_skip = rearrange(x_skip, 'b l h p -> b l (h p)')

        y = y + x_skip

        # ============ 门控机制 ============
        y = y * F.silu(z)

        # ============ 归一化 ============
        y = self.norm(y)

        # ============ 输出投影 ============
        y = self.out_proj(y)
        y = torch.clamp(y, min=-10, max=10)  # 防止爆炸

        # ============ 最终归一化 ============
        y = self.norm_output(y)

        return y

    def _selective_scan_chunked(self, x, A, B, C, dt):
        """
        分块的选择性扫描（Mamba-2 启发）

        x: [B, L, nheads, headdim]
        A: [nheads]
        B: [B, L, 1, N]
        C: [B, L, 1, N]
        dt: [B, L, nheads, 1]

        返回: [B, L, nheads, headdim]
        """
        B_batch, L, nheads, headdim = x.shape
        N = B.shape[-1]

        # ============ Padding 到 chunk_size 的倍数 ============
        pad_len = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        # ============ 分块 ============
        num_chunks = L_padded // self.chunk_size

        # Reshape 成 chunks: [B, num_chunks, chunk_size, ...]
        x_chunked = rearrange(x, 'b (c l) h p -> b c l h p', c=num_chunks)
        B_chunked = rearrange(B, 'b (c l) h n -> b c l h n', c=num_chunks)
        C_chunked = rearrange(C, 'b (c l) h n -> b c l h n', c=num_chunks)
        dt_chunked = rearrange(dt, 'b (c l) h p -> b c l h p', c=num_chunks)

        # ============ 逐 chunk 处理（可以并行，但这里用循环保持简单）============
        outputs = []
        h = torch.zeros(B_batch, nheads, headdim, N, device=x.device, dtype=x.dtype)

        for chunk_idx in range(num_chunks):
            x_chunk = x_chunked[:, chunk_idx]  # [B, chunk_size, nheads, headdim]
            B_chunk = B_chunked[:, chunk_idx]  # [B, chunk_size, 1, N]
            C_chunk = C_chunked[:, chunk_idx]  # [B, chunk_size, 1, N]
            dt_chunk = dt_chunked[:, chunk_idx]  # [B, chunk_size, nheads, 1]

            # Chunk 内的扫描
            y_chunk, h = self._scan_chunk(x_chunk, h, A, B_chunk, C_chunk, dt_chunk)
            outputs.append(y_chunk)

        # ============ 合并 chunks ============
        y = torch.cat(outputs, dim=1)  # [B, L_padded, nheads, headdim]

        # ============ 去除 padding ============
        if pad_len > 0:
            y = y[:, :L]

        return y

    def _scan_chunk(self, x, h_init, A, B, C, dt):
        """
        单个 chunk 内的扫描（Mamba-2 风格的状态更新）

        x: [B, chunk_size, nheads, headdim]
        h_init: [B, nheads, headdim, N] - 初始隐藏状态
        A: [nheads]
        B: [B, chunk_size, 1, N]
        C: [B, chunk_size, 1, N]
        dt: [B, chunk_size, nheads, 1]

        返回:
        - y: [B, chunk_size, nheads, headdim]
        - h_final: [B, nheads, headdim, N]
        """
        B_batch, chunk_size, nheads, headdim = x.shape
        N = B.shape[-1]

        # ============ Mamba-2 风格的离散化 ============
        # dA = exp(dt * A)
        A_expanded = A.view(1, 1, nheads, 1)  # [1, 1, nheads, 1]
        dA = torch.exp(dt * A_expanded)  # [B, chunk_size, nheads, 1]
        dA = torch.clamp(dA, max=0.99)  # 防止不稳定

        # ============ 逐时间步扫描 ============
        h = h_init.clone()
        outputs = []

        for t in range(chunk_size):
            x_t = x[:, t]  # [B, nheads, headdim]
            B_t = B[:, t]  # [B, 1, N]
            C_t = C[:, t]  # [B, 1, N]
            dA_t = dA[:, t]  # [B, nheads, 1]

            # ============ Mamba-2 的状态更新公式 ============
            # dBx = dt * B * x
            # dBx: [B, nheads, headdim, N]
            dBx = torch.einsum('bhp,bhn->bhpn', x_t, B_t.squeeze(1))

            # h_new = dA * h + dBx
            h = dA_t.unsqueeze(-1).unsqueeze(-1) * h + dBx

            # 裁剪防止爆炸
            h = torch.clamp(h, min=-10, max=10)

            # ============ 输出 ============
            # y = C * h
            # C: [B, 1, N], h: [B, nheads, headdim, N] -> [B, nheads, headdim]
            y_t = torch.einsum('bhn,bhpn->bhp', C_t.squeeze(1), h)

            outputs.append(y_t)

        # 堆叠输出
        y = torch.stack(outputs, dim=1)  # [B, chunk_size, nheads, headdim]

        return y, h


class MambaCrossBlock_V2(nn.Module):
    """
    Mamba-2 版本的交叉注入模块

    核心创新：
    1. V 和 I 各有独立的 Mamba2 模块
    2. 交换生成的 B 矩阵（和 C 矩阵）
    3. 可学习的融合权重
    """

    def __init__(self, in_channels, d_model=256, d_state=64, headdim=64, chunk_size=16):
        super(MambaCrossBlock_V2, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        # ============ 通道调整 ============
        if in_channels != d_model:
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(in_channels, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Dropout2d(0.1)
            )
            self.channel_restore = nn.Sequential(
                nn.Conv2d(d_model, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.channel_reduce = nn.Identity()
            self.channel_restore = nn.Identity()

        # ============ 两个独立的 Mamba2 分支 ============
        self.mamba_V = SimplifiedMamba2Block_V2(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand_factor=2,
            chunk_size=chunk_size
        )
        self.mamba_I = SimplifiedMamba2Block_V2(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand_factor=2,
            chunk_size=chunk_size
        )

        # ============ 融合参数 ============
        # alpha: 自己的输出 vs 交叉的输出
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # gate: 残差连接的强度
        self.gate = nn.Parameter(torch.tensor(0.01))

        # ============ 可选：学习交叉权重 ============
        self.cross_gate = nn.Parameter(torch.tensor(0.5))

    def feature_to_sequence(self, x):
        """特征图 -> 序列"""
        B, C, H, W = x.shape

        # 如果太大，下采样
        max_seq_len = 256
        if H * W > max_seq_len:
            scale = int(math.ceil(math.sqrt((H * W) / max_seq_len)))
            x = F.adaptive_avg_pool2d(x, (H // scale, W // scale))
            H, W = H // scale, W // scale

        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return x, H, W

    def sequence_to_feature(self, x, H, W):
        """序列 -> 特征图"""
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x_V, x_I):
        """
        x_V: RGB 特征 [B_V, C, H, W]
        x_I: IR 特征 [B_I, C, H, W]

        返回: (x_V_out, x_I_out)
        """
        # ============ 安全检查 ============
        if torch.isnan(x_V).any() or torch.isnan(x_I).any():
            print("⚠️  WARNING: NaN in MambaCross input!")
            return x_V.clone(), x_I.clone()

        # ============ 保存残差 ============
        residual_V = x_V.clone()
        residual_I = x_I.clone()

        try:
            # ============ 降维 ============
            x_V_reduced = self.channel_reduce(x_V)
            x_I_reduced = self.channel_reduce(x_I)

            # ============ 转序列 ============
            seq_V, H_V, W_V = self.feature_to_sequence(x_V_reduced)
            seq_I, H_I, W_I = self.feature_to_sequence(x_I_reduced)

            # ============ 前向传播获取 B, C 矩阵 ============
            # 我们需要从 in_proj 中提取 B 和 C

            # V 分支
            zxbcdt_V = self.mamba_V.in_proj(self.mamba_V.norm_input(seq_V))
            z_V, x_proj_V, dt_V = torch.split(
                zxbcdt_V,
                [self.mamba_V.d_inner, self.mamba_V.d_inner + 2 * self.mamba_V.d_state, self.mamba_V.nheads],
                dim=-1
            )
            x_ssm_V, B_V, C_V = torch.split(
                x_proj_V,
                [self.mamba_V.d_inner, self.mamba_V.d_state, self.mamba_V.d_state],
                dim=-1
            )

            # I 分支
            zxbcdt_I = self.mamba_I.in_proj(self.mamba_I.norm_input(seq_I))
            z_I, x_proj_I, dt_I = torch.split(
                zxbcdt_I,
                [self.mamba_I.d_inner, self.mamba_I.d_inner + 2 * self.mamba_I.d_state, self.mamba_I.nheads],
                dim=-1
            )
            x_ssm_I, B_I, C_I = torch.split(
                x_proj_I,
                [self.mamba_I.d_inner, self.mamba_I.d_state, self.mamba_I.d_state],
                dim=-1
            )

            # ============ 标准输出（使用自己的 B, C）============
            out_V_std = self.mamba_V(seq_V, B_override=None, C_override=None)
            out_I_std = self.mamba_I(seq_I, B_override=None, C_override=None)

            if torch.isnan(out_V_std).any() or torch.isnan(out_I_std).any():
                print("⚠️  WARNING: NaN in standard forward!")
                return residual_V, residual_I

            # ============ 交叉输出（交换 B 和 C 矩阵）============
            # V 使用 I 的 B, C
            B_I_detached = B_I.detach()
            C_I_detached = C_I.detach()
            out_V_cross = self.mamba_V(seq_V, B_override=B_I_detached, C_override=C_I_detached, freeze_params=True)

            # I 使用 V 的 B, C
            B_V_detached = B_V.detach()
            C_V_detached = C_V.detach()
            out_I_cross = self.mamba_I(seq_I, B_override=B_V_detached, C_override=C_V_detached, freeze_params=True)

            if torch.isnan(out_V_cross).any() or torch.isnan(out_I_cross).any():
                print("⚠️  WARNING: NaN in cross forward!")
                return residual_V, residual_I

            # ============ 融合（可学习权重）============
            alpha = torch.sigmoid(self.alpha)
            cross_gate = torch.sigmoid(self.cross_gate)

            # 融合策略：alpha * std + (1-alpha) * cross_gate * cross
            out_V_fused = alpha * out_V_std + (1 - alpha) * cross_gate * out_V_cross
            out_I_fused = alpha * out_I_std + (1 - alpha) * cross_gate * out_I_cross

            if torch.isnan(out_V_fused).any() or torch.isnan(out_I_fused).any():
                print(f"⚠️  WARNING: NaN after fusion! alpha={alpha:.4f}, cross_gate={cross_gate:.4f}")
                return residual_V, residual_I

            # ============ 转回特征图 ============
            feat_V = self.sequence_to_feature(out_V_fused, H_V, W_V)
            feat_I = self.sequence_to_feature(out_I_fused, H_I, W_I)

            # ============ 恢复空间分辨率 ============
            if feat_V.shape[2:] != x_V.shape[2:]:
                feat_V = F.interpolate(feat_V, size=x_V.shape[2:], mode='bilinear', align_corners=False)
            if feat_I.shape[2:] != x_I.shape[2:]:
                feat_I = F.interpolate(feat_I, size=x_I.shape[2:], mode='bilinear', align_corners=False)

            # ============ 升维 ============
            feat_V = self.channel_restore(feat_V)
            feat_I = self.channel_restore(feat_I)

            if torch.isnan(feat_V).any() or torch.isnan(feat_I).any():
                print("⚠️  WARNING: NaN after channel restore!")
                return residual_V, residual_I

            # ============ 残差连接 ============
            gate = torch.sigmoid(self.gate)
            out_V = residual_V + gate * feat_V
            out_I = residual_I + gate * feat_I

            # ============ 最终检查 ============
            if torch.isnan(out_V).any() or torch.isnan(out_I).any():
                print(f"⚠️  WARNING: NaN in final output! gate={gate:.6f}")
                return residual_V, residual_I

            return out_V, out_I

        except Exception as e:
            print(f"❌ ERROR in MambaCrossBlock_V2: {e}")
            import traceback
            traceback.print_exc()
            return residual_V, residual_I


# ==================== 保留原来的轻量级交叉注意力（未使用）====================

class LightweightCrossAttention(nn.Module):
    """
    轻量级交叉注意力（暂未使用，保留供未来扩展）
    """

    def __init__(self, d_model, num_heads=4, reduction=2):
        super(LightweightCrossAttention, self).__init__()

        self.d_model = d_model
        self.d_reduced = d_model // reduction
        self.num_heads = num_heads
        self.head_dim = self.d_reduced // num_heads

        assert self.d_reduced % num_heads == 0

        self.reduce = nn.Linear(d_model, self.d_reduced)
        self.q_proj = nn.Linear(self.d_reduced, self.d_reduced)
        self.k_proj = nn.Linear(self.d_reduced, self.d_reduced)
        self.v_proj = nn.Linear(self.d_reduced, self.d_reduced)
        self.out_proj = nn.Linear(self.d_reduced, d_model)

        self.scale = self.head_dim ** -0.5

    def forward(self, x1, x2):
        """
        x1, x2: (B, L, D)
        """
        B, L, D = x1.shape

        x1_reduced = self.reduce(x1)
        x2_reduced = self.reduce(x2)

        Q = self.q_proj(x1_reduced).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x2_reduced).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x2_reduced).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, L, self.d_reduced)
        out = self.out_proj(out)

        return out