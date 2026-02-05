# models/extension.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MambaSSMBlock(nn.Module):
    """
    完整的Mamba SSM实现

    符合论文标准：
    1. 时变参数Δ, B, C由输入动态生成
    2. 对角化A矩阵（负实数）
    3. 零阶保持离散化
    4. 支持外部注入B/C参数
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaSSMBlock, self).__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # A矩阵：对角矩阵，负实数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D：跳跃连接
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 时变参数投影（核心）
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)

        # dt投影
        self.dt_proj = nn.Linear(self.d_state * 2 + 1, self.d_inner, bias=True)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 归一化
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.D, -0.01, 0.01)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.5)

    def forward(self, x, BC_injection=None, return_BC=False):
        """
        前向传播

        Args:
            x: (B, L, D) 输入序列
            BC_injection: dict with keys 'B', 'C', 'alpha'
                - B: (B, L, D_inner, N) 外部B参数
                - C: (B, L, D_inner, N) 外部C参数
                - alpha: float, 混合系数
            return_BC: 是否返回生成的B/C参数（用于传递给另一分支）

        Returns:
            output: (B, L, D)
            BC_params: dict (if return_BC=True)
        """
        B_batch, L, D_model = x.shape

        # 归一化
        x_norm = self.norm(x)

        # 1. 输入投影
        xz = self.in_proj(x_norm)
        x_ssm, z = xz.chunk(2, dim=-1)  # 各 (B, L, D_inner)

        # 2. 1D卷积
        x_ssm = x_ssm.transpose(1, 2)  # (B, D_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, D_inner)
        x_ssm = F.silu(x_ssm)

        # 3. 生成时变参数
        x_proj_out = self.x_proj(x_ssm)  # (B, L, 2N+1)

        # 分离 dt_raw, B_self, C_self
        dt_raw, B_self, C_self = torch.split(
            x_proj_out,
            [1, self.d_state, self.d_state],
            dim=-1
        )

        # dt投影
        dt = self.dt_proj(x_proj_out)  # (B, L, D_inner)
        dt = F.softplus(dt) + 0.001  # 保证 dt > 0

        # 4. 决定使用哪个B和C
        if BC_injection is not None:
            # 混合自己的和外部的B/C
            alpha = BC_injection.get('alpha', 0.5)
            B_external = BC_injection['B']  # (B, L, D_inner, N)
            C_external = BC_injection['C']

            # 扩展B_self和C_self到相同形状
            B_self_expanded = B_self.unsqueeze(2).expand(-1, -1, self.d_inner, -1)
            C_self_expanded = C_self.unsqueeze(2).expand(-1, -1, self.d_inner, -1)

            # 混合
            B_used = alpha * B_self_expanded + (1 - alpha) * B_external
            C_used = alpha * C_self_expanded + (1 - alpha) * C_external
        else:
            # 只用自己的B/C
            B_used = B_self.unsqueeze(2).expand(-1, -1, self.d_inner, -1)
            C_used = C_self.unsqueeze(2).expand(-1, -1, self.d_inner, -1)

        # 5. 获取A（对角矩阵，负实数）
        A = -torch.exp(self.A_log).float()  # (D_inner, N)

        # 6. SSM递推
        h = torch.zeros(B_batch, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            x_t = x_ssm[:, t, :]  # (B, D_inner)
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D_inner, 1)
            B_t = B_used[:, t, :, :]  # (B, D_inner, N)
            C_t = C_used[:, t, :, :]  # (B, D_inner, N)

            # 离散化（零阶保持）
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, D_inner, N)
            dB = (1 - dA) / (A.unsqueeze(0) + 1e-8) * B_t

            # 状态更新
            h = dA * h + dB * x_t.unsqueeze(-1)

            # 数值稳定性
            h = torch.clamp(h, -10, 10)

            # 输出
            y_t = torch.sum(C_t * h, dim=-1) + self.D * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D_inner)

        # 7. 门控
        y = y * F.silu(z)

        # 8. 输出投影
        output = self.out_proj(y)

        # 9. 返回B/C参数（如果需要）
        if return_BC:
            BC_params = {
                'B': B_self.unsqueeze(2).expand(-1, -1, self.d_inner, -1),
                'C': C_self.unsqueeze(2).expand(-1, -1, self.d_inner, -1)
            }
            return output, BC_params
        else:
            return output


class MambaCrossBlock(nn.Module):
    """
    Mamba交叉注入模块 - 完整实现

    真正实现B/C矩阵交叉注入：
    1. 第一次前向：各自生成B/C
    2. 交叉传递：V使用I的B/C，I使用V的B/C
    3. 融合输出
    """

    def __init__(self, in_channels, d_model=512, d_state=16):
        super(MambaCrossBlock, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        # 通道调整
        if in_channels != d_model:
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(in_channels, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Dropout2d(0.05)
            )
            self.channel_restore = nn.Sequential(
                nn.Conv2d(d_model, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.channel_reduce = nn.Identity()
            self.channel_restore = nn.Identity()

        # 两个独立的Mamba块
        self.mamba_V = MambaSSMBlock(d_model, d_state, expand=2)
        self.mamba_I = MambaSSMBlock(d_model, d_state, expand=2)

        # 可学习的融合参数
        self.alpha_std = nn.Parameter(torch.tensor(0.6))  # 标准输出权重
        self.alpha_cross = nn.Parameter(torch.tensor(0.5))  # 交叉时的混合系数

        # 残差门控
        self.gate = nn.Parameter(torch.tensor(0.1))

    def feature_to_sequence(self, x):
        """特征图 -> 序列，带下采样"""
        B, C, H, W = x.shape

        max_seq_len = 196  # 14x14
        if H * W > max_seq_len:
            target_size = int(math.sqrt(max_seq_len))
            x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            H, W = target_size, target_size

        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x, H, W

    def sequence_to_feature(self, x, H, W):
        """序列 -> 特征图"""
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x_V, x_I):
        """
        交叉注入前向传播

        流程：
        1. V分支生成 out_V_std 和 BC_V
        2. I分支生成 out_I_std 和 BC_I
        3. V使用BC_I生成 out_V_cross
        4. I使用BC_V生成 out_I_cross
        5. 融合输出
        """
        # 输入检查
        if torch.isnan(x_V).any() or torch.isnan(x_I).any():
            print("[WARNING] NaN in input!")
            return x_V, x_I

        residual_V = x_V
        residual_I = x_I

        try:
            # === 1. 降维 ===
            x_V_reduced = self.channel_reduce(x_V)
            x_I_reduced = self.channel_reduce(x_I)

            # === 2. 转序列 ===
            seq_V, H_V, W_V = self.feature_to_sequence(x_V_reduced)
            seq_I, H_I, W_I = self.feature_to_sequence(x_I_reduced)

            # === 3. 第一次前向：生成标准输出和B/C参数 ===
            out_V_std, BC_V = self.mamba_V(seq_V, BC_injection=None, return_BC=True)
            out_I_std, BC_I = self.mamba_I(seq_I, BC_injection=None, return_BC=True)

            if torch.isnan(out_V_std).any() or torch.isnan(out_I_std).any():
                print("[WARNING] NaN in standard forward!")
                return residual_V, residual_I

            # === 4. 交叉前向：使用对方的B/C ===
            alpha_cross = torch.sigmoid(self.alpha_cross)

            # V使用I的B/C
            BC_injection_V = {
                'B': BC_I['B'].detach(),  # 阻断梯度避免循环依赖
                'C': BC_I['C'].detach(),
                'alpha': alpha_cross.item()
            }
            out_V_cross = self.mamba_V(seq_V, BC_injection=BC_injection_V, return_BC=False)

            # I使用V的B/C
            BC_injection_I = {
                'B': BC_V['B'].detach(),
                'C': BC_V['C'].detach(),
                'alpha': alpha_cross.item()
            }
            out_I_cross = self.mamba_I(seq_I, BC_injection=BC_injection_I, return_BC=False)

            if torch.isnan(out_V_cross).any() or torch.isnan(out_I_cross).any():
                print("[WARNING] NaN in cross forward!")
                return residual_V, residual_I

            # === 5. 融合标准输出和交叉输出 ===
            alpha_std = torch.sigmoid(self.alpha_std)

            out_V_fused = alpha_std * out_V_std + (1 - alpha_std) * out_V_cross
            out_I_fused = alpha_std * out_I_std + (1 - alpha_std) * out_I_cross

            # === 6. 转回特征图 ===
            feat_V = self.sequence_to_feature(out_V_fused, H_V, W_V)
            feat_I = self.sequence_to_feature(out_I_fused, H_I, W_I)

            # 上采样到原始尺寸
            if feat_V.shape[2:] != x_V.shape[2:]:
                feat_V = F.interpolate(feat_V, size=x_V.shape[2:],
                                       mode='bilinear', align_corners=False)
            if feat_I.shape[2:] != x_I.shape[2:]:
                feat_I = F.interpolate(feat_I, size=x_I.shape[2:],
                                       mode='bilinear', align_corners=False)

            # === 7. 升维 ===
            feat_V = self.channel_restore(feat_V)
            feat_I = self.channel_restore(feat_I)

            if torch.isnan(feat_V).any() or torch.isnan(feat_I).any():
                print("[WARNING] NaN after channel restore!")
                return residual_V, residual_I

            # === 8. 残差连接 ===
            gate_val = torch.sigmoid(self.gate)
            out_V = residual_V + gate_val * feat_V
            out_I = residual_I + gate_val * feat_I

            # 最终检查
            if torch.isnan(out_V).any() or torch.isnan(out_I).any():
                print(f"[WARNING] NaN in final output! gate={gate_val.item():.6f}")
                return residual_V, residual_I

            return out_V, out_I

        except Exception as e:
            print(f"[ERROR] Exception in MambaCrossBlock: {e}")
            import traceback
            traceback.print_exc()
            return residual_V, residual_I


# ==================== 轻量级交叉注意力（备用） ====================
class LightweightCrossAttention(nn.Module):
    """备用的轻量级交叉注意力"""

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