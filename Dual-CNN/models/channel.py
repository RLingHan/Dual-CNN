from torch import nn
import torch
import torch.nn.functional as F


class MUMModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(MUMModule, self).__init__()
        # 通道级判别器：输入 GAP 后的特征 [B, C]
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()  # 输出 P_c: 每个通道属于 RGB 的概率
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape

        # 1. 全局平均池化得到通道描述符
        feat_gap = F.avg_pool2d(x, (h, w)).view(b, c)

        # 2. 预测模态概率 P
        # P ≈ 1: 强偏向可见光; P ≈ 0: 强偏向红外; P ≈ 0.5: 模态不确定 (Shared)
        P = self.discriminator(feat_gap)

        # 3. 生成软掩码 (Soft Mask)
        # Shared Mask: 采用三角波函数，在 0.5 处取得极大值 1
        mask_sh = 1 - torch.abs(2 * P - 1)

        # Specific Mask: 在 0 或 1 处取得极大值 1
        mask_sp = torch.abs(2 * P - 1)

        # 4. 调整维度以便与特征图相乘 [B, C, 1, 1]
        mask_sh = mask_sh.view(b, c, 1, 1)
        mask_sp = mask_sp.view(b, c, 1, 1)

        return mask_sh, mask_sp, P


class MDIA(nn.Module):
    """
    Modal Disentanglement and Interaction Adapter
    三位一体：分离 + 交互对齐 + 自适应幻觉增强
    """

    def __init__(self, in_channels=1024, modal_classes=2):
        super().__init__()

        # ── 1. Disentangle 分支 ──────────────────────────────
        self.modal_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        # 有监督辅助分类头（二分类：RGB vs IR）
        self.modal_cls_head = nn.Linear(in_channels, modal_classes)

        # ── 2. Cross-Modal Interaction ───────────────────────
        self.q_proj = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.k_proj = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.interact_norm = nn.GroupNorm(32, in_channels)

        # ── 3. Adaptive Hallucination ────────────────────────
        # 输入是两个 GAP 特征拼接 (B, in_channels*2)，不含池化层
        self.lambda_predictor = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # λ ∈ (0, 1)
        )

        self.out_norm = nn.GroupNorm(32, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def disentangle(self, x):
        """返回共享特征、特有特征、以及用于有监督的modal_feat"""
        P = self.modal_discriminator(x)          # (B, C)，已经过池化+flatten
        modal_feat = P                            # 用于 modal_cls_head
        P = P.view(P.size(0), P.size(1), 1, 1)  # (B, C, 1, 1)
        mask_sh = 1 - torch.abs(2 * P - 1)       # 三角波，0.5处最大
        mask_sp = torch.abs(2 * P - 1)           # 两端最大
        f_sh = x * mask_sh
        f_sp = x * mask_sp
        return f_sh, f_sp, modal_feat

    def cross_interact(self, f_sh, sub):
        """
        跨模态共享特征交互对齐
        只在双模态都存在时做，否则走 identity
        """
        has_v = (sub == 0).any()
        has_i = (sub == 1).any()

        if not (has_v and has_i):
            return f_sh

        f_v = f_sh[sub == 0]  # (Bv, C, H, W)
        f_i = f_sh[sub == 1]  # (Bi, C, H, W)

        def attend(query_feat, kv_feat):
            dtype = query_feat.dtype
            B, C, H, W = query_feat.shape

            q = self.q_proj(query_feat)                                   # (B, C/4, H, W)
            k = self.k_proj(kv_feat.mean(0, keepdim=True).expand_as(kv_feat))  # (B, C/4, H, W)
            v = self.v_proj(kv_feat.mean(0, keepdim=True).expand_as(kv_feat))  # (B, C, H, W)

            q_flat = q.flatten(2)                                          # (B, C/4, HW)
            k_flat = k.flatten(2).mean(0, keepdim=True).expand(B, -1, -1) # (B, C/4, HW)
            v_flat = v.flatten(2).mean(0, keepdim=True).expand(B, -1, -1) # (B, C, HW)

            scale = (C // 4) ** 0.5
            attn = torch.softmax(
                torch.bmm(q_flat.transpose(1, 2), k_flat) / scale,
                dim=-1
            )  # (B, HW, HW)

            out = torch.bmm(v_flat, attn.transpose(1, 2))  # (B, C, HW)
            out = out.view(B, C, H, W).to(dtype)

            return query_feat + self.gamma * self.interact_norm(out).to(dtype)

        out_v = attend(f_v, f_i)
        out_i = attend(f_i, f_v)

        f_sh_new = torch.zeros_like(f_sh)
        f_sh_new[sub == 0] = out_v.to(f_sh.dtype)
        f_sh_new[sub == 1] = out_i.to(f_sh.dtype)
        return f_sh_new

    def adaptive_hallucination(self, f_sh, f_sp, labels, sub):
        """自适应幻觉注入：λ 由模态差异动态预测"""
        B = f_sh.size(0)
        device = f_sh.device

        # 找跨模态、不同ID的负样本特有特征
        sub_diff   = sub.unsqueeze(1) != sub.unsqueeze(0)      # (B, B)
        label_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
        valid = (sub_diff & label_diff).float()

        rand_w  = torch.rand(B, B, device=device) * valid
        selected = rand_w.argmax(1)                             # (B,)
        f_sp_cross = f_sp[selected]                             # (B, C, H, W)

        # GAP 后拼接，输入 lambda_predictor
        sh_gap = F.adaptive_avg_pool2d(f_sh, 1).flatten(1).float()       # (B, C)
        sp_gap = F.adaptive_avg_pool2d(f_sp_cross, 1).flatten(1).float() # (B, C)
        lam = self.lambda_predictor(torch.cat([sh_gap, sp_gap], dim=1))  # (B, 1)
        lam = lam.to(f_sh.dtype).view(B, 1, 1, 1)

        f_hallu = f_sh + lam * f_sp_cross
        return f_hallu, lam.squeeze()

    def forward(self, x, sub, labels=None):
        # 1. Disentangle
        f_sh, f_sp, modal_feat = self.disentangle(x)

        # 有监督模态分类 logits
        modal_logits = self.modal_cls_head(modal_feat.float())  # (B, 2)

        if self.training and labels is not None:
            # 2. Cross-Modal Interaction
            f_sh = self.cross_interact(f_sh, sub)
            # 3. Adaptive Hallucination
            f_out, lam = self.adaptive_hallucination(f_sh, f_sp, labels, sub)
        else:
            # 测试时只用对齐后的共享特征
            f_out = f_sh
            lam = None

        return f_out, f_sp, modal_logits, lam


    def adaptive_hallucination(self, f_sh, f_sp, labels, sub):
        """
        自适应幻觉注入：λ 由模态差异动态预测
        """
        B = f_sh.size(0)
        device = f_sh.device

        # 找跨模态、不同ID的负样本特有特征
        sub_diff = sub.unsqueeze(1) != sub.unsqueeze(0)  # (B, B)
        label_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
        valid = (sub_diff & label_diff).float()

        rand_w = torch.rand(B, B, device=device) * valid
        row_sum = rand_w.sum(1, keepdim=True).clamp(min=1e-8)
        selected = rand_w.argmax(1)  # (B,)
        f_sp_cross = f_sp[selected]  # (B, C, H, W)

        # 动态预测 λ：输入是共享特征和跨模态特有特征的拼接
        sh_gap = F.adaptive_avg_pool2d(f_sh, 1).flatten(1)  # (B, C)
        sp_gap = F.adaptive_avg_pool2d(f_sp_cross, 1).flatten(1)
        lam = self.lambda_predictor(
            torch.cat([sh_gap, sp_gap], dim=1))  # (B, 1)
        lam = lam.view(B, 1, 1, 1)  # broadcast

        f_hallu = f_sh + lam * f_sp_cross
        return f_hallu, lam.squeeze()

    def forward(self, x, sub, labels=None):
        # 1. Disentangle
        f_sh, f_sp, modal_feat = self.disentangle(x)

        # 有监督模态分类（训练时返回logits用于计算loss）
        modal_logits = self.modal_cls_head(modal_feat)  # (B, 2)

        if self.training and labels is not None:
            # 2. Cross-Modal Interaction
            f_sh = self.cross_interact(f_sh, sub)

            # 3. Adaptive Hallucination
            f_out, lam = self.adaptive_hallucination(f_sh, f_sp, labels, sub)
        else:
            # 测试时：只用对齐后的共享特征，不注入噪声
            f_out = f_sh
            lam = None

        return f_out, f_sp, modal_logits, lam

class FeatureDecomposition(nn.Module):
    """将特征分解为模态共享和模态特定"""

    def __init__(self, dim, reduction=16):
        super().__init__()

        # 学习分解mask
        self.shared_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 生成共享/特定mask
        shared_mask = self.shared_gate(x)  # (B, C, 1, 1)
        specific_mask = 1 - shared_mask

        feat_shared = x * shared_mask
        feat_specific = x * specific_mask

        return feat_shared, feat_specific

class GlobalContextBlock(nn.Module):
    """全局上下文注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # 全局上下文建模
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # 通道变换
        self.channel_add = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 全局上下文聚合
        context_mask = self.conv_mask(x)  # (B, 1, H, W)
        context_mask = context_mask.view(B, 1, H * W)
        context_mask = self.softmax(context_mask)  # (B, 1, H*W)

        context = torch.matmul(
            x.view(B, C, H * W),
            context_mask.transpose(1, 2)
        )  # (B, C, 1)
        context = context.unsqueeze(-1)  # (B, C, 1, 1)

        # 通道增强
        transform = self.channel_add(context)
        return x + transform


class LargeKernelAttention(nn.Module):
    """大核注意力"""

    def __init__(self, channels, kernel_size=7):
        super().__init__()

        # 深度可分离大核卷积
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels
        )

        # 1×1逐点卷积
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 大感受野特征
        attn = self.dw_conv(x)
        attn = self.pw_conv(attn)

        # 空间门控
        gate = self.spatial_gate(attn)

        return x * gate + x


class AdaptiveGlobalModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_attn = GlobalContextBlock(channels)
        self.lka = LargeKernelAttention(channels)

        # 门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

        # 缩放因子
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 全局特征
        global_feat = self.global_attn(x)
        global_feat = self.lka(global_feat)

        # 自适应门控
        gate = self.gate(x)

        # 小权重残差
        out = x + self.scale * gate * global_feat
        return out