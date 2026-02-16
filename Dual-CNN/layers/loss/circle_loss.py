import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss for VI-ReID (跨模态行人重识别)
    可作为主损失函数使用
    论文: Circle Loss: A Unified Perspective of Pair Similarity Optimization
    参数:
        m: 边界参数，默认0.25（论文推荐值）
        gamma: 缩放因子，默认128（论文推荐值，可尝试256获得更强优化）
    """
    def __init__(self, m=0.25, gamma=128):
        super(CircleLoss, self).__init__()
        self.m = m  # margin边界
        self.gamma = gamma  # scale缩放因子
        self.soft_plus = nn.Softplus()  # 用于数值稳定的log(1+exp(x))

    def forward(self, features, labels, modalities=None):
        """
        前向传播

        参数:
            features: 特征张量 shape=(batch_size, feature_dim)
            labels: ID标签 shape=(batch_size,)
            modalities: 模态标签 shape=(batch_size,)，0=可见光 1=红外
                       如果为None，视为标准Circle Loss
                       如果提供，则分别优化模态内和跨模态匹配

        返回:
            loss: Circle Loss损失值（标量）
        """
        # 归一化特征到单位球面（强烈推荐，使用余弦相似度）
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵：features @ features^T
        # shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(features, features.T)

        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)

        # 创建正负样本mask
        # mask_pos[i,j] = 1 表示i和j是同一个ID（正样本对）
        mask_pos = torch.eq(labels, labels.T).float()
        # mask_neg[i,j] = 1 表示i和j是不同ID（负样本对）
        mask_neg = torch.eq(labels, labels.T).logical_not().float()

        # 去除对角线（自己和自己的相似度）
        mask_pos = mask_pos - torch.eye(batch_size, device=features.device)

        # ========== 如果提供了模态信息，则进行模态感知优化 ==========
        if modalities is not None:
            modalities = modalities.contiguous().view(-1, 1)

            # 创建模态mask
            # mask_same_mod[i,j] = 1 表示i和j来自同一模态
            mask_same_mod = torch.eq(modalities, modalities.T).float()
            mask_diff_mod = torch.eq(modalities, modalities.T).logical_not().float()

            # 去除对角线
            mask_same_mod = mask_same_mod - torch.eye(batch_size, device=features.device)

            # 分解为4种情况：
            # 1. 模态内正样本：同模态 + 同ID
            mask_intra_pos = mask_pos * mask_same_mod
            # 2. 模态内负样本：同模态 + 不同ID
            mask_intra_neg = mask_neg * mask_same_mod
            # 3. 跨模态正样本：不同模态 + 同ID
            mask_cross_pos = mask_pos * mask_diff_mod
            # 4. 跨模态负样本：不同模态 + 不同ID
            mask_cross_neg = mask_neg * mask_diff_mod

            # 分别计算模态内和跨模态的Circle Loss
            loss_intra = self._compute_circle_loss(
                similarity_matrix, mask_intra_pos, mask_intra_neg
            )
            loss_cross = self._compute_circle_loss(
                similarity_matrix, mask_cross_pos, mask_cross_neg
            )

            # 总损失：模态内 + 跨模态（权重相等）
            return loss_intra + loss_cross

        # ========== 标准Circle Loss（不区分模态） ==========
        else:
            return self._compute_circle_loss(similarity_matrix, mask_pos, mask_neg)

    def _compute_circle_loss(self, sim_matrix, mask_pos, mask_neg):
        """
        计算Circle Loss的核心函数

        参数:
            sim_matrix: 相似度矩阵 shape=(B, B)
            mask_pos: 正样本对mask shape=(B, B)
            mask_neg: 负样本对mask shape=(B, B)

        返回:
            loss: Circle Loss值
        """
        # 提取正负样本的相似度
        sim_pos = sim_matrix * mask_pos  # f_j^{t+}
        sim_neg = sim_matrix * mask_neg  # f_i^{t-}

        # 计算自适应权重（detach阻止梯度回传到权重）
        # α_pos = [1 + m - sim_pos]_+  （正样本权重）
        alpha_pos = torch.clamp_min(1 + self.m - sim_pos.detach(), min=0.)
        # α_neg = [sim_neg + m]_+  （负样本权重）
        alpha_neg = torch.clamp_min(sim_neg.detach() + self.m, min=0.)

        # 计算加权后的logits
        # 正样本项: -α_pos * (sim_pos - m)
        logits_pos = -alpha_pos * (sim_pos - self.m) * self.gamma
        # 负样本项: α_neg * (sim_neg + m)
        logits_neg = alpha_neg * (sim_neg + self.m) * self.gamma

        # 只保留有效的正负样本对
        logits_pos = logits_pos * mask_pos
        logits_neg = logits_neg * mask_neg

        # 计算损失：log[1 + Σexp(logits_neg) · Σexp(logits_pos)]
        # 使用LogSumExp技巧保证数值稳定
        neg_exp = torch.exp(logits_neg) * mask_neg
        pos_exp = torch.exp(logits_pos) * mask_pos

        # 对每个anchor，求和所有负样本
        neg_term = neg_exp.sum(dim=1, keepdim=True)  # shape: (B, 1)
        # 对每个anchor，求和所有正样本
        pos_term = pos_exp.sum(dim=1)  # shape: (B,)

        # soft_plus(log(1 + x)) = log(1 + exp(log(1 + x)))
        loss = self.soft_plus(torch.log(1 + neg_term * pos_term + 1e-8))

        # 归一化：除以每个样本的正样本对数量
        num_valid_pairs = mask_pos.sum(dim=1).clamp(min=1.0)
        loss = loss / num_valid_pairs.unsqueeze(1)

        # 返回batch平均损失
        return loss.mean()