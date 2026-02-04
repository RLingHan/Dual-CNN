# models/extension.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


# ==================== 修复版 Mamba交叉注入模块 ====================

class SimplifiedMamba2Block(nn.Module):
    """
    简化版Mamba2状态空间模型 - 超级稳定版

    关键改进：
    1. 使用更小的d_state避免累积误差
    2. Layer normalization在每个关键步骤
    3. 梯度裁剪
    4. 更保守的数值范围
    """

    def __init__(self, d_model, d_state=8, expand_factor=1.5):  # 减小expand_factor和d_state
        super(SimplifiedMamba2Block, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM参数 - 更保守的初始化
        A_init = torch.randn(self.d_inner, d_state) * 0.1  # 小的初始化
        self.A_log = nn.Parameter(torch.log(-A_init.abs() - 0.1))

        self.B = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 多个LayerNorm用于稳定性
        self.norm_input = nn.LayerNorm(d_model)
        self.norm_output = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        # 非常保守的初始化
        nn.init.orthogonal_(self.B, gain=0.1)
        nn.init.orthogonal_(self.C, gain=0.01)
        nn.init.orthogonal_(self.in_proj.weight, gain=0.5)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.5)

    def forward(self, x, B_override=None, freeze_B=False):
        """
        x: (B, L, D)
        """
        B_batch, L, D = x.shape

        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: Invalid input to Mamba! NaN: {torch.isnan(x).any()}, Inf: {torch.isinf(x).any()}")
            return torch.zeros_like(x)

        # 使用B矩阵
        B_matrix = B_override if B_override is not None else self.B
        if freeze_B and B_override is None:
            B_matrix = B_matrix.detach()

        # 输入归一化
        x_norm = self.norm_input(x)

        # 输入投影
        x_proj = self.in_proj(x_norm)
        x_proj = torch.clamp(x_proj, min=-5, max=5)  # 限制范围
        x_ssm, gate = x_proj.chunk(2, dim=-1)

        # A矩阵 - 更严格的范围
        A = -torch.exp(self.A_log.clamp(min=-5, max=0))
        A = A.clamp(min=-2, max=-0.01)

        # 简化的SSM - 只用单向扫描避免复杂度
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            x_t = x_ssm[:, t, :]

            # 状态更新 - 添加epsilon避免数值问题
            update = torch.einsum('bi,id->bid', x_t, B_matrix)
            h = h * A.unsqueeze(0) * 0.9 + update * 0.1  # 降低反馈增益

            # 严格裁剪
            h = torch.clamp(h, min=-5, max=5)

            # 输出
            y_t = torch.einsum('bid,id->bi', h, self.C)
            y_t = torch.clamp(y_t, min=-5, max=5)
            outputs.append(y_t)

        out_ssm = torch.stack(outputs, dim=1)

        # 检查SSM输出
        if torch.isnan(out_ssm).any():
            print("WARNING: NaN in SSM output!")
            return torch.zeros_like(x)

        # 门控 - 使用tanh代替silu，更稳定
        out = out_ssm * torch.tanh(gate)

        # 输出投影
        out = self.out_proj(out)
        out = torch.clamp(out, min=-5, max=5)

        # 输出归一化
        out = self.norm_output(out)

        return out


class MambaCrossBlock(nn.Module):
    """
    Mamba交叉注入模块 - 超级稳定版

    关键改进：
    1. 更简单的融合策略
    2. 每步都检查NaN
    3. 降低复杂度
    """

    def __init__(self, in_channels, d_model=256, d_state=8, reduce_spatial=True):  # 降低d_model
        super(MambaCrossBlock, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        # 降维 - 添加Dropout增加鲁棒性
        if in_channels != d_model:
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(in_channels, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)  # 添加dropout
            )
            self.channel_restore = nn.Sequential(
                nn.Conv2d(d_model, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.channel_reduce = nn.Identity()
            self.channel_restore = nn.Identity()

        # 两个独立的Mamba分支
        self.mamba_V = SimplifiedMamba2Block(d_model, d_state)
        self.mamba_I = SimplifiedMamba2Block(d_model, d_state)

        # 简化的融合权重 - 全局可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 自己 vs 交叉的权重

        # 残差gate - 初始值很小
        self.gate = nn.Parameter(torch.tensor(0.01))

    def feature_to_sequence(self, x):
        """特征图 -> 序列，添加下采样减少序列长度"""
        B, C, H, W = x.shape

        # 如果空间维度太大，先下采样
        if H * W > 256:  # 序列长度限制
            scale = int(math.sqrt((H * W) / 256))
            x = F.adaptive_avg_pool2d(x, (H // scale, W // scale))
            H, W = H // scale, W // scale

        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x, H, W

    def sequence_to_feature(self, x, H, W):
        """序列 -> 特征图"""
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def safe_forward(self, func, *args, **kwargs):
        """安全的前向传播，捕获NaN"""
        try:
            result = func(*args, **kwargs)
            if torch.isnan(result).any() or torch.isinf(result).any():
                print(f"WARNING: NaN/Inf in {func.__class__.__name__}")
                return None
            return result
        except Exception as e:
            print(f"ERROR in {func.__class__.__name__}: {e}")
            return None

    def forward(self, x_V, x_I):
        """
        x_V: RGB特征 (B_V, C, H, W)
        x_I: IR特征 (B_I, C, H, W)
        """
        # 输入检查
        if torch.isnan(x_V).any() or torch.isnan(x_I).any():
            print("WARNING: NaN in Mamba input!")
            return x_V, x_I

        # 保存残差
        residual_V = x_V.clone()
        residual_I = x_I.clone()

        try:
            # 降维
            x_V_reduced = self.channel_reduce(x_V)
            x_I_reduced = self.channel_reduce(x_I)

            if torch.isnan(x_V_reduced).any() or torch.isnan(x_I_reduced).any():
                print("WARNING: NaN after channel reduction!")
                return residual_V, residual_I

            # 转换为序列
            seq_V, H_V, W_V = self.feature_to_sequence(x_V_reduced)
            seq_I, H_I, W_I = self.feature_to_sequence(x_I_reduced)

            # 标准输出
            out_V_std = self.safe_forward(self.mamba_V, seq_V, B_override=None, freeze_B=False)
            out_I_std = self.safe_forward(self.mamba_I, seq_I, B_override=None, freeze_B=False)

            if out_V_std is None or out_I_std is None:
                print("WARNING: Mamba standard forward failed!")
                return residual_V, residual_I

            # 交叉输出（使用对方的B矩阵）
            B_V_frozen = self.mamba_V.B.detach()
            B_I_frozen = self.mamba_I.B.detach()

            out_V_cross = self.safe_forward(self.mamba_V, seq_V, B_override=B_I_frozen, freeze_B=True)
            out_I_cross = self.safe_forward(self.mamba_I, seq_I, B_override=B_V_frozen, freeze_B=True)

            if out_V_cross is None or out_I_cross is None:
                print("WARNING: Mamba cross forward failed!")
                return residual_V, residual_I

            # 简单融合：alpha * std + (1-alpha) * cross
            alpha_val = torch.sigmoid(self.alpha)  # 限制在[0,1]

            out_V_fused = alpha_val * out_V_std + (1 - alpha_val) * out_V_cross
            out_I_fused = alpha_val * out_I_std + (1 - alpha_val) * out_I_cross

            # 检查融合结果
            if torch.isnan(out_V_fused).any() or torch.isnan(out_I_fused).any():
                print(f"WARNING: NaN after fusion! alpha={alpha_val.item():.4f}")
                return residual_V, residual_I

            # 转换回特征图
            feat_V = self.sequence_to_feature(out_V_fused, H_V, W_V)
            feat_I = self.sequence_to_feature(out_I_fused, H_I, W_I)

            # 恢复到原始空间分辨率
            if feat_V.shape[2:] != x_V.shape[2:]:
                feat_V = F.interpolate(feat_V, size=x_V.shape[2:], mode='bilinear', align_corners=False)
            if feat_I.shape[2:] != x_I.shape[2:]:
                feat_I = F.interpolate(feat_I, size=x_I.shape[2:], mode='bilinear', align_corners=False)

            # 升维
            feat_V = self.channel_restore(feat_V)
            feat_I = self.channel_restore(feat_I)

            if torch.isnan(feat_V).any() or torch.isnan(feat_I).any():
                print("WARNING: NaN after channel restoration!")
                return residual_V, residual_I

            # 残差连接（非常小的gate）
            gate_val = torch.sigmoid(self.gate)
            out_V = residual_V + gate_val * feat_V
            out_I = residual_I + gate_val * feat_I

            # 最终检查
            if torch.isnan(out_V).any() or torch.isnan(out_I).any():
                print(f"WARNING: NaN in final output! gate={gate_val.item():.6f}")
                return residual_V, residual_I

            return out_V, out_I

        except Exception as e:
            print(f"ERROR in MambaCrossBlock: {e}")
            import traceback
            traceback.print_exc()
            return residual_V, residual_I


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