# models/extension.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


# ==================== 修复版 Mamba交叉注入模块 ====================

class SimplifiedMamba2Block(nn.Module):
    """
    简化版Mamba2状态空间模型
    核心：h[t] = A × h[t-1] + B × x[t]
          y[t] = C × h[t]

    修复：使用函数式调用而非直接修改参数
    """

    def __init__(self, d_model, d_state=16, expand_factor=2):
        super(SimplifiedMamba2Block, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM参数
        A = torch.randn(self.d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(-A.abs() - 1.0))

        self.B = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.C = nn.Parameter(torch.zeros(self.d_inner, d_state))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.B)
        nn.init.constant_(self.C, 0.01)  # 小的非零初始化
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, B_override=None, freeze_B=False):
        """
        x: (B, L, D)
        B_override: 可选的替代B矩阵（用于交叉注入）
        freeze_B: 是否冻结B矩阵的梯度
        """
        B_batch, L, D = x.shape

        # 使用提供的B矩阵或默认的B
        B_matrix = B_override if B_override is not None else self.B

        # 如果需要冻结，detach
        if freeze_B and B_override is None:
            B_matrix = B_matrix.detach()

        # 输入投影
        x_proj = self.in_proj(self.norm(x))
        x_ssm, gate = x_proj.chunk(2, dim=-1)

        # A矩阵
        A = -torch.exp(self.A_log).clamp(min=-10, max=-0.01)  # 防止数值溢出

        # SSM前向传播
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        # 双向扫描 - Forward
        for t in range(L):
            x_t = x_ssm[:, t, :]  # (B, d_inner)
            # h[t] = A * h[t-1] + B * x[t]
            h = h * A.unsqueeze(0) + torch.einsum('bi,id->bid', x_t, B_matrix)
            # 添加数值稳定性
            h = torch.clamp(h, min=-10, max=10)
            # y[t] = C * h[t]
            y_t = torch.einsum('bid,id->bi', h, self.C)
            outputs.append(y_t)

        out_forward = torch.stack(outputs, dim=1)

        # 双向扫描 - Backward
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs_back = []

        for t in range(L - 1, -1, -1):
            x_t = x_ssm[:, t, :]
            h = h * A.unsqueeze(0) + torch.einsum('bi,id->bid', x_t, B_matrix)
            h = torch.clamp(h, min=-10, max=10)
            y_t = torch.einsum('bid,id->bi', h, self.C)
            outputs_back.append(y_t)

        out_backward = torch.stack(outputs_back[::-1], dim=1)

        # 融合
        out_ssm = (out_forward + out_backward) * 0.5

        # 门控
        out = out_ssm * F.silu(gate)

        # 输出投影
        out = self.out_proj(out)

        return out


class MambaCrossBlock(nn.Module):
    """
    Mamba交叉注入模块 - 修复版

    关键修复：
    1. 不再直接修改参数 .data
    2. 使用函数参数传递B矩阵
    3. 添加数值稳定性检查
    4. 简化交叉逻辑
    """

    def __init__(self, in_channels, d_model=512, d_state=16, reduce_spatial=True):
        super(MambaCrossBlock, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        # 降维
        if in_channels != d_model:
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(in_channels, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
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

        # 动态λ权重预测器（简化版）
        self.lambda_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model * 2, d_model // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, 2, 1),
            nn.Softmax(dim=1)  # 两个权重：self和cross
        )

        # 可学习的残差gate
        self.gate = nn.Parameter(torch.tensor(0.1))

    def feature_to_sequence(self, x):
        """特征图 -> 序列"""
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x, H, W

    def sequence_to_feature(self, x, H, W):
        """序列 -> 特征图"""
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x_V, x_I):
        """
        x_V: RGB特征 (B_V, C, H, W)
        x_I: IR特征 (B_I, C, H, W)
        """
        # 检查输入
        if torch.isnan(x_V).any() or torch.isnan(x_I).any():
            print("WARNING: NaN detected in Mamba input!")
            return x_V, x_I

        B_V, _, H, W = x_V.shape
        B_I = x_I.shape[0]

        # 保存残差
        residual_V = x_V
        residual_I = x_I

        # 降维
        x_V_reduced = self.channel_reduce(x_V)
        x_I_reduced = self.channel_reduce(x_I)

        # 预测混合权重（对每个模态）
        # 拼接两个模态的特征来预测
        concat_V = torch.cat([x_V_reduced, F.adaptive_avg_pool2d(x_I_reduced, (H, W))[:B_V]], dim=1)
        concat_I = torch.cat([F.adaptive_avg_pool2d(x_V_reduced, x_I_reduced.shape[2:])[:B_I], x_I_reduced], dim=1)

        lambda_V = self.lambda_predictor(concat_V)  # (B_V, 2, 1, 1)
        lambda_I = self.lambda_predictor(concat_I)  # (B_I, 2, 1, 1)

        # 转换为序列
        seq_V, H_V, W_V = self.feature_to_sequence(x_V_reduced)
        seq_I, H_I, W_I = self.feature_to_sequence(x_I_reduced)

        # 标准输出（使用自己的B矩阵）
        out_V_std = self.mamba_V(seq_V, B_override=None, freeze_B=False)
        out_I_std = self.mamba_I(seq_I, B_override=None, freeze_B=False)

        # 交叉输出（使用对方的B矩阵，冻结梯度）
        B_V_frozen = self.mamba_V.B.detach()
        B_I_frozen = self.mamba_I.B.detach()

        out_V_cross = self.mamba_V(seq_V, B_override=B_I_frozen, freeze_B=True)
        out_I_cross = self.mamba_I(seq_I, B_override=B_V_frozen, freeze_B=True)

        # 自适应融合：λ[0] * std + λ[1] * cross
        lambda_V_self = lambda_V[:, 0:1, :, :].squeeze(-1).squeeze(-1).unsqueeze(1)  # (B_V, 1, 1)
        lambda_V_cross = lambda_V[:, 1:2, :, :].squeeze(-1).squeeze(-1).unsqueeze(1)  # (B_V, 1, 1)

        lambda_I_self = lambda_I[:, 0:1, :, :].squeeze(-1).squeeze(-1).unsqueeze(1)
        lambda_I_cross = lambda_I[:, 1:2, :, :].squeeze(-1).squeeze(-1).unsqueeze(1)

        out_V_fused = lambda_V_self * out_V_std + lambda_V_cross * out_V_cross
        out_I_fused = lambda_I_self * out_I_std + lambda_I_cross * out_I_cross

        # 检查融合后是否有nan
        if torch.isnan(out_V_fused).any() or torch.isnan(out_I_fused).any():
            print("WARNING: NaN detected after Mamba fusion! Returning residual.")
            return residual_V, residual_I

        # 转换回特征图
        feat_V = self.sequence_to_feature(out_V_fused, H_V, W_V)
        feat_I = self.sequence_to_feature(out_I_fused, H_I, W_I)

        # 升维
        feat_V = self.channel_restore(feat_V)
        feat_I = self.channel_restore(feat_I)

        # 残差连接（带可学习gate，限制幅度）
        gate_value = torch.sigmoid(self.gate)  # 限制在[0,1]
        out_V = residual_V + gate_value * feat_V
        out_I = residual_I + gate_value * feat_I

        # 最终检查
        if torch.isnan(out_V).any() or torch.isnan(out_I).any():
            print("WARNING: NaN in Mamba output! Using residual only.")
            return residual_V, residual_I

        return out_V, out_I


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