import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss for VI-ReID (修正版)
    """

    def __init__(self, m=0.25, gamma=128):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

        # 定义两个目标阈值：
        # 正样本目标： > 1-m (例如 > 0.75)
        # 负样本目标： < m   (例如 < 0.25)
        self.delta_p = 1 - m
        self.delta_n = m

    def forward(self, features, labels, modalities=None):
        # 归一化特征 (必须)
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)

        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)

        # 基础Mask
        mask_pos = torch.eq(labels, labels.T).float()
        mask_neg = torch.eq(labels, labels.T).logical_not().float()

        # 去除自身 (对角线)
        mask_pos = mask_pos - torch.eye(batch_size, device=features.device)

        # ========== 模态感知优化 ==========
        if modalities is not None:
            modalities = modalities.contiguous().view(-1, 1)

            mask_same_mod = torch.eq(modalities, modalities.T).float()
            mask_diff_mod = torch.eq(modalities, modalities.T).logical_not().float()

            # 修正：对角线只在 mask_same_mod 中存在，需要再次确保去除
            mask_same_mod = mask_same_mod - torch.eye(batch_size, device=features.device)

            # 分解Mask
            mask_intra_pos = mask_pos * mask_same_mod
            mask_intra_neg = mask_neg * mask_same_mod

            mask_cross_pos = mask_pos * mask_diff_mod
            mask_cross_neg = mask_neg * mask_diff_mod

            # 计算两部分Loss
            loss_intra = self._compute_circle_loss(
                similarity_matrix, mask_intra_pos, mask_intra_neg
            )
            loss_cross = self._compute_circle_loss(
                similarity_matrix, mask_cross_pos, mask_cross_neg
            )

            # 建议：返回平均值或者总和，这里返回总和
            return (loss_intra + loss_cross) * 0.5

            # ========== 标准 Circle Loss ==========
        else:
            return self._compute_circle_loss(similarity_matrix, mask_pos, mask_neg)

    def _compute_circle_loss(self, sim_matrix, mask_pos, mask_neg):
        # 如果没有有效的正样本对（极端情况），返回0
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=sim_matrix.device, requires_grad=True)

        # 提取相似度
        sim_pos = sim_matrix * mask_pos
        sim_neg = sim_matrix * mask_neg

        # 1. 计算自适应权重 Alpha (论文公式)
        # detach() 很重要，防止梯度回传到权重计算中
        alpha_pos = torch.clamp_min(1 + self.m - sim_pos.detach(), min=0.)
        alpha_neg = torch.clamp_min(sim_neg.detach() + self.m, min=0.)

        # 2. 计算加权 Logits (关键修正处!)
        # 正样本: 目标是 sim_pos > (1-m)
        # 原始公式: - gamma * alpha * (s_p - delta_p)
        logits_pos = -alpha_pos * (sim_pos - self.delta_p) * self.gamma

        # 负样本: 目标是 sim_neg < m
        # 原始公式: gamma * alpha * (s_n - delta_n)
        logits_neg = alpha_neg * (sim_neg - self.delta_n) * self.gamma

        # 3. 各种Mask处理 (保持原逻辑，利用logsumexp技巧)
        # 对正样本: 为了数值稳定，无效位置设为负无穷
        logits_pos = logits_pos * mask_pos + (1 - mask_pos) * (-1e12)
        # 对负样本: 无效位置设为负无穷
        logits_neg = logits_neg * mask_neg + (1 - mask_neg) * (-1e12)

        # 4. LogSumExp
        # 找出每行的最大值用于数值稳定
        max_pos = logits_pos.max(dim=1, keepdim=True)[0].detach()
        max_neg = logits_neg.max(dim=1, keepdim=True)[0].detach()

        # LogSumExp 计算
        # exp(x - max) 防止溢出
        pos_exp = torch.exp(logits_pos - max_pos) * mask_pos
        neg_exp = torch.exp(logits_neg - max_neg) * mask_neg

        logsumexp_pos = max_pos + torch.log(pos_exp.sum(dim=1, keepdim=True) + 1e-12)
        logsumexp_neg = max_neg + torch.log(neg_exp.sum(dim=1, keepdim=True) + 1e-12)

        # 5. Final Loss
        # loss = softplus(logsumexp_neg + logsumexp_pos)
        loss = self.soft_plus(logsumexp_neg + logsumexp_pos)

        # 6. Mean Loss
        # 只计算那些确实有正样本对的行
        valid_rows = (mask_pos.sum(dim=1) > 0).float()
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=sim_matrix.device, requires_grad=True)

        return (loss * valid_rows.view(-1, 1)).sum() / valid_rows.sum()