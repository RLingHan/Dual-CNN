import torch.nn as nn
from torch.nn import functional as F
# from torchvision.models.utils import load_state_dict_from_url   #原来
from torch.hub import load_state_dict_from_url
import torch
import math

# ==================== 新增模块：Mamba交叉注入 ====================

class SimplifiedMamba2Block(nn.Module):
    """
    简化版Mamba2状态空间模型
    核心：h[t] = A × h[t-1] + B × x[t]
          y[t] = C × h[t]
    """

    def __init__(self, d_model, d_state=16, expand_factor=2):
        super(SimplifiedMamba2Block, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM参数
        # A: 状态转移矩阵 (d_inner, d_state)
        # B: 输入矩阵 (d_inner, d_state)
        # C: 输出矩阵 (d_inner, d_state)

        A = torch.randn(self.d_inner, d_state)
        # A初始化为稍微负的值保证稳定性
        self.A_log = nn.Parameter(torch.log(-A.abs() - 1.0))

        self.B = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.C = nn.Parameter(torch.zeros(self.d_inner, d_state))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        # Xavier初始化B矩阵
        nn.init.xavier_uniform_(self.B)
        # C矩阵保持零初始化
        nn.init.constant_(self.C, 0)
        # 投影层
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        """
        x: (B, L, D) - L是序列长度
        """
        B, L, D = x.shape

        # 输入投影并分割
        x_proj = self.in_proj(self.norm(x))
        x_ssm, gate = x_proj.chunk(2, dim=-1)  # 各自 (B, L, d_inner)

        # 应用SSM
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 初始化隐状态
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        outputs = []

        # 双向扫描：forward
        for t in range(L):
            x_t = x_ssm[:, t, :]  # (B, d_inner)
            # 状态更新: h[t] = A × h[t-1] + B × x[t]
            h = h * A.unsqueeze(0) + torch.einsum('bi,id->bid', x_t, self.B)
            # 输出: y[t] = C × h[t]
            y_t = torch.einsum('bid,id->bi', h, self.C)
            outputs.append(y_t)

        out_forward = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        # 双向扫描：backward
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        outputs_back = []

        for t in range(L - 1, -1, -1):
            x_t = x_ssm[:, t, :]
            h = h * A.unsqueeze(0) + torch.einsum('bi,id->bid', x_t, self.B)
            y_t = torch.einsum('bid,id->bi', h, self.C)
            outputs_back.append(y_t)

        out_backward = torch.stack(outputs_back[::-1], dim=1)

        # 融合双向输出
        out_ssm = (out_forward + out_backward) * 0.5

        # 门控
        out = out_ssm * F.silu(gate)

        # 输出投影
        out = self.out_proj(out)

        return out


class MambaCrossBlock(nn.Module):
    """
    Mamba交叉注入模块
    使用自适应融合策略（方案3）
    """

    def __init__(self, in_channels, d_model=512, d_state=16, reduce_spatial=True):
        super(MambaCrossBlock, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model
        self.reduce_spatial = reduce_spatial

        # 降维（如果需要）
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

        # 两个独立的Mamba分支（RGB和IR）
        self.mamba_V = SimplifiedMamba2Block(d_model, d_state)
        self.mamba_I = SimplifiedMamba2Block(d_model, d_state)

        # 轻量级交叉注意力模块
        self.cross_attention = LightweightCrossAttention(d_model)

        # 动态λ权重预测器
        self.lambda_predictor_V = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, 1, 1),
            nn.Sigmoid()
        )

        self.lambda_predictor_I = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, 1, 1),
            nn.Sigmoid()
        )

        # 自适应A矩阵融合权重
        self.A_attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
            nn.Softmax(dim=-1)
        )

    def feature_to_sequence(self, x):
        """
        将特征图转换为序列
        x: (B, C, H, W) -> (B, H*W, C)
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x

    def sequence_to_feature(self, x, H, W):
        """
        将序列转换回特征图
        x: (B, H*W, C) -> (B, C, H, W)
        """
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x_V, x_I):
        """
        x_V: RGB特征 (B_V, C, H, W)
        x_I: IR特征 (B_I, C, H, W)
        """
        B_V, _, H, W = x_V.shape
        B_I = x_I.shape[0]

        # 保存残差
        residual_V = x_V
        residual_I = x_I

        # 降维
        x_V_reduced = self.channel_reduce(x_V)
        x_I_reduced = self.channel_reduce(x_I)

        # 预测动态λ权重
        lambda_V = self.lambda_predictor_V(x_V_reduced)  # (B_V, 1, 1, 1)
        lambda_I = self.lambda_predictor_I(x_I_reduced)  # (B_I, 1, 1, 1)

        # 转换为序列
        seq_V = self.feature_to_sequence(x_V_reduced)  # (B_V, H*W, d_model)
        seq_I = self.feature_to_sequence(x_I_reduced)  # (B_I, H*W, d_model)

        # 通过各自的Mamba获得标准输出
        # 注意：这里需要使用交叉的B矩阵
        out_V_std = self.mamba_V(seq_V)  # 使用B_V
        out_I_std = self.mamba_I(seq_I)  # 使用B_I

        # 交叉B矩阵的输出（实现方式：交换B参数）
        # 临时交换B矩阵
        B_V_original = self.mamba_V.B.data.clone()
        B_I_original = self.mamba_I.B.data.clone()

        self.mamba_V.B.data = B_I_original
        self.mamba_I.B.data = B_V_original

        out_V_cross = self.mamba_V(seq_V)  # 使用B_I
        out_I_cross = self.mamba_I(seq_I)  # 使用B_V

        # 恢复B矩阵
        self.mamba_V.B.data = B_V_original
        self.mamba_I.B.data = B_I_original

        # 自适应融合
        # λ_V × out_std + (1-λ_V) × out_cross
        lambda_V_seq = lambda_V.flatten(1).unsqueeze(1)  # (B_V, 1, 1)
        lambda_I_seq = lambda_I.flatten(1).unsqueeze(1)  # (B_I, 1, 1)

        out_V_fused = lambda_V_seq * out_V_std + (1 - lambda_V_seq) * out_V_cross
        out_I_fused = lambda_I_seq * out_I_std + (1 - lambda_I_seq) * out_I_cross

        # 转换回特征图
        feat_V = self.sequence_to_feature(out_V_fused, H, W)
        feat_I = self.sequence_to_feature(out_I_fused, H, W)

        # 升维回原始通道数
        feat_V = self.channel_restore(feat_V)
        feat_I = self.channel_restore(feat_I)

        # 残差连接
        out_V = residual_V + feat_V
        out_I = residual_I + feat_I

        return out_V, out_I


class LightweightCrossAttention(nn.Module):
    """
    轻量级交叉注意力：降维 -> cross-attention -> 升维
    """

    def __init__(self, d_model, num_heads=4, reduction=2):
        super(LightweightCrossAttention, self).__init__()

        self.d_model = d_model
        self.d_reduced = d_model // reduction
        self.num_heads = num_heads
        self.head_dim = self.d_reduced // num_heads

        assert self.d_reduced % num_heads == 0

        # 降维
        self.reduce = nn.Linear(d_model, self.d_reduced)

        # Q, K, V投影
        self.q_proj = nn.Linear(self.d_reduced, self.d_reduced)
        self.k_proj = nn.Linear(self.d_reduced, self.d_reduced)
        self.v_proj = nn.Linear(self.d_reduced, self.d_reduced)

        # 输出投影 + 升维
        self.out_proj = nn.Linear(self.d_reduced, d_model)

        self.scale = self.head_dim ** -0.5

    def forward(self, x1, x2):
        """
        x1, x2: (B, L, D)
        计算x1 attend to x2
        """
        B, L, D = x1.shape

        # 降维
        x1_reduced = self.reduce(x1)
        x2_reduced = self.reduce(x2)

        # 投影
        Q = self.q_proj(x1_reduced).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x2_reduced).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x2_reduced).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 输出
        out = (attn @ V).transpose(1, 2).reshape(B, L, self.d_reduced)
        out = self.out_proj(out)

        return out