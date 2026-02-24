import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    VI-ReID 适配版 ArcFace
    支持双模态独立计算或合并计算
    """
    def __init__(self, in_features, num_classes, s=64.0, m=0.5,
                 easy_margin=False, label_smoothing=0.0):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s          # 超球面半径，通常 30~64
        self.m = m          # angular margin，通常 0.3~0.5
        self.label_smoothing = label_smoothing

        # 权重矩阵，等价于原来的 classifier.weight
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)   # cos(π - m)
        self.mm = math.sin(math.pi - m) * m

        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-1
        )

    def forward(self, feat, labels):
        """
        feat:   [B, C]  已经过 bn_neck 的特征（不需要再 normalize）
        labels: [B]
        """
        # 特征和权重都 L2 归一化，投影到超球面
        cosine = F.linear(F.normalize(feat), F.normalize(self.weight))
        # cosine: [B, num_classes]

        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        # cos(θ + m) = cosθ·cosm - sinθ·sinm
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 当 θ + m > π 时退化，防止梯度异常
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot：目标类用 phi，其余类用原始 cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = self.ce(output, labels)
        return loss, output  # 同时返回 logits 方便计算 acc