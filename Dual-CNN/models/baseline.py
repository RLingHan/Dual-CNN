#models/baseline.py
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

from layers.module.reverse_grad import ReverseGrad
from models.resnet import resnet50, embed_net, convDiscrimination, Discrimination
from utils.calc_acc import calc_acc

from layers.loss.rerank_loss import  RerankLoss
from layers.loss.triplet_loss import TripletLoss
from layers.loss.local_center_loss import CenterTripletLoss
from layers.loss.center_loss import CenterLoss
from layers.loss.circle_loss import CircleLoss
# from layers import cbam
# from layers import NonLocalBlockND
from utils.rerank import re_ranking, pairwise_distance

def intersect1d(tensor1, tensor2):
    #找出 tensor1 和 tensor2 中的共有元素
    return torch.unique(torch.cat([tensor1[tensor1 == val] for val in tensor2]))

#斯皮尔曼损失函数
def spearman_loss(dist_matrix, rerank_matrix):
    #dist_matrix 是原始特征的距离矩阵 rerank_matrix 是重排序（Re-ranking）之后的距离矩阵
    sorted_idx_dist = torch.argsort(dist_matrix, dim=1)
    sorted_idx_rerank = torch.argsort(rerank_matrix, dim=1)

    rank_corr = 0
    n = dist_matrix.size(1)
    for i in range(dist_matrix.size(0)):
        diff = sorted_idx_dist[i] - sorted_idx_rerank[i]
        rank_corr += 1 - (6 * torch.sum(diff * diff) / (n * (n**2 - 1)))

    rank_corr /= dist_matrix.size(0)

    return 1 - rank_corr #如果作为损失函数  为了确保网络提取的特征所计算的原始距离矩阵本身就具有很高的质量。

#负对数似然损失 鼓励同类样本之间距离更小，不同类样本距离更大
def Fb_dt(feat, labels):
    feat_dt = feat
    n_ft = feat_dt.size(0)
    dist_f = torch.pow(feat_dt, 2).sum(dim=1, keepdim=True).expand(n_ft, n_ft)
    dist_f = dist_f + dist_f.t()
    dist_f.addmm_(1, -2, feat_dt, feat_dt.t())
    dist_f = dist_f.clamp(min=1e-12).sqrt()
    mask_ft = labels.expand(n_ft, n_ft).eq(labels.expand(n_ft, n_ft).t())
    mask_ft_1 = torch.ones(n_ft, n_ft, dtype=bool)
    for i in range(n_ft):
        mask_ft_1[i, i] = 0
    mask_ft_2 = []
    for i in range(n_ft):

        mask_ft_2.append(mask_ft[i][mask_ft_1[i]])
    mask_ft_2 = torch.stack(mask_ft_2)
    dist_f_2 = []
    for i in range(n_ft):

        dist_f_2.append(dist_f[i][mask_ft_1[i]])
    dist_f_2 = torch.stack(dist_f_2)
    dist_f_2 = F.softmax(-(dist_f_2 - 1), 1)
    cN_ft = (mask_ft_2[0] == True).sum()
    f_d_ap = []
    for i in range(n_ft):

        f_d_ap.append(dist_f_2[i][mask_ft_2[i]])
    f_d_ap = torch.stack(f_d_ap).flatten()
    loss_f_d_ap = []
    xs_ft = 1
    m_ft = f_d_ap.size(0)
    for i in range(m_ft):
        loss_f_d_ap.append(
            -xs_ft * (1 / cN_ft) * torch.log(xs_ft * cN_ft * f_d_ap[i]))
    loss_f_d_ap = torch.stack(loss_f_d_ap).clamp(max=1e+3).sum() / n_ft
    return loss_f_d_ap


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

# 就是把池化后的张量展平成特征向量
def gem_p(x):
    ss = gem(x).squeeze()  # Gem池化
    ss= ss.view(ss.size(0), -1)  # Gem池化
    return ss

#标准的欧式距离矩阵计算
def pairwise_dist(x, y):
    # Compute pairwise distance of vectors
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2.0 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist

# 移除了所有自己与自己比较的距离
def kl_soft_dist(feat1,feat2):
    n_st = feat1.size(0)
    dist_st = pairwise_dist(feat1, feat2)
    mask_st_1 = torch.ones(n_st, n_st, dtype=bool)
    for i in range(n_st):  # 将同一类样本中自己与自己的距离舍弃
        mask_st_1[i, i] = 0
    dist_st_2 = []
    for i in range(n_st):
        dist_st_2.append(dist_st[i][mask_st_1[i]])
    dist_st_2 = torch.stack(dist_st_2)
    return dist_st_2

# 对称 KL 散度
def Bg_kl(logits1, logits2):####输入:(60,206),(60,206)
    KL = nn.KLDivLoss(reduction='batchmean')
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    bg_loss_kl = kl_loss_12 + kl_loss_21
    return kl_loss_12, bg_loss_kl


'''
计算了两个 Logits（分类得分）张量之间的对称 KL 散度（Symmetric KL Divergence），
但其独特之处在于它在计算之前对输入 Logits 进行了按身份分组和特征维度拼接，
使其能捕捉到更复杂的批次级别语义关系
'''
def Sm_kl(logits1, logits2, labels):
    KL = nn.KLDivLoss(reduction='batchmean')
    m_kl = torch.div((labels == labels[0]).sum(), 2, rounding_mode='floor')
    v_logits_s = logits1.split(m_kl, 0)
    i_logits_s = logits2.split(m_kl, 0)
    sm_v_logits = torch.cat(v_logits_s, 1)  # .t()  # 5,206*12->206*12,5
    sm_i_logits = torch.cat(i_logits_s, 1)  # .t()
    sm_kl_loss_vi = KL(F.log_softmax(sm_v_logits, 1), F.softmax(sm_i_logits, 1))
    sm_kl_loss_iv = KL(F.log_softmax(sm_i_logits, 1), F.softmax(sm_v_logits, 1))
    sm_kl_loss = sm_kl_loss_vi + sm_kl_loss_iv
    return sm_kl_loss_vi, sm_kl_loss

# 每个样本的预测熵 熵高 → 模型不确定
def samplewise_entropy(logits):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    entropies = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropies

# 让两个模态输出的置信度水平相似。
def entropy_margin_loss(logits1, logits2, margin):
    entropy1 = samplewise_entropy(logits1)
    entropy2 = samplewise_entropy(logits2)
    losses = torch.exp(F.relu(entropy2 - entropy1 + margin)) - 1
    return losses.mean()


def compute_centroid_distance(features, labels, modalities):
    """
    计算每个类别不同模态的中心特征的距离。
    计算每个 ID 在两种模态（红外 / 可见光）下的中心特征之间的距离。
    距离越大，说明模态间特征差异越大。
    参数:
    features -- 特征矩阵，形状为(B, C)。
    labels -- 类别标签，形状为(B,)。
    modalities -- 模态标签，形状为(B,)。

    返回:
    distances -- 每个类别模态中心距离的列表。
    """
    unique_labels = torch.unique(labels)
    distances = []
    for label in unique_labels:
        # 分别获取当前类别下的两种模态的特征
        features_modality_0 = features[(labels == label) & (modalities == 0)]
        features_modality_1 = features[(labels == label) & (modalities == 1)]

        # 计算中心特征
        centroid_modality_0 = features_modality_0.mean(dim=0)
        centroid_modality_1 = features_modality_1.mean(dim=0)

        # 计算两个中心特征之间的距离，这里使用欧氏距离
        distance = F.pairwise_distance(centroid_modality_0.unsqueeze(0), centroid_modality_1.unsqueeze(0))
        distances.append(distance)


    return torch.stack(distances)


def modal_centroid_loss(F1, F2, labels, modalities, margin):
    """
    计算损失函数，要求F2中每个类别不同模态的中心距离比F1更小，并施加一个margin。

    参数:
    F1 -- 第一组特征，形状为(B, C)。
    F2 -- 第二组特征，经过网络结构优化，形状为(B, C)。
    labels -- 类别标签，形状为(B,)。
    modalities -- 模态标签，形状为(B,)。
    margin -- 施加的margin值。

    返回:
    loss -- 计算的损失值。
    """
    # 计算F1和F2的中心距离
    distances_F1 = compute_centroid_distance(F1, labels, modalities)
    distances_F2 = compute_centroid_distance(F2, labels, modalities)

    # 计算带margin的损失
    losses = F.relu(distances_F2 - distances_F1 + margin)

    # 返回损失的平均值
    return losses.mean()


class ModalityAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, v_feat, i_feat, v_labels, i_labels):
        # 此时 v_feat 和 v_labels 都是 N=30
        v_feat = F.normalize(v_feat, dim=1)
        i_feat = F.normalize(i_feat, dim=1)

        # sim 形状为 (30, 30)
        sim = torch.mm(v_feat, i_feat.t()) / self.temp

        # v_labels.unsqueeze(1) -> (30, 1)
        # i_labels.unsqueeze(0) -> (1, 30)
        labels_eq = v_labels.unsqueeze(1) == i_labels.unsqueeze(0)

        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * labels_eq.float()).sum(1)
        all_sum = exp_sim.sum(1)

        # 增加 1e-8 防止 log(0)
        loss = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
        return loss


class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.decompose = decompose
        self.backbone = embed_net(drop_last_stride=drop_last_stride, decompose=decompose)

        self.base_dim = 2048
        self.dim = 0
        self.part_num = kwargs.get('num_parts', 0)


        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck_sp = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck_sp.bias, 0)
        self.bn_neck_sp.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)

        self.modality_align_loss = ModalityAlignmentLoss()
        self.align = False

        self.circle_criterion = CircleLoss(
            m=kwargs.get('circle_margin', 0.25),
            gamma=kwargs.get('circle_gamma', 128)
        )

        # 消融实验
        self.CSA1 = kwargs.get('bg_kl', False)
        self.CSA2 = kwargs.get('sm_kl', False)
        self.TGSA = kwargs.get('distalign', False)


        self.IP = kwargs.get('IP', False)
        self.fb_dt = kwargs.get('fb_dt', False)
        self.mutual_learning = kwargs.get('mutual_learning', False)

        self.D_shared_pseu = Discrimination()  # 伪模态分类器（共享特征分支）
        self.special_D = convDiscrimination(1024)

        if self.decompose:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False) # 主分类器（共享特征）
            self.classifier_sp = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False) # 特有模态分类器
            self.D_special = Discrimination() # 判别当前特征来自哪种模态
            self.C_sp_f = nn.Linear(self.base_dim, num_classes, bias=False) # 用于熵边界约束

        else:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        if self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)
            self.rerank_loss = RerankLoss(margin=0.7)
        if self.center_cluster:
            k_size = kwargs.get('k_size', 8)
            self.center_cluster_loss = CenterTripletLoss(k_size=k_size, margin=self.margin)
        if self.center_loss:
            self.center_loss = CenterLoss(num_classes, self.base_dim + self.dim * self.part_num)

    def forward(self, inputs, labels=None, **kwargs):

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6) #0 Visible 1 Infrared
        #epoch = kwargs.get('epoch')
        # CNN
        #layer4输出  layer4的语义特征  相互调节后的语义特征 mask前/后模态无关特征 mask前/后特别特征
        sh_pl, alpha, f_sh, f_sp, p_mod, sp_pl = self.backbone(inputs,sub=sub,labels=labels)
        #提取特征

        feats = sh_pl #layer4的语义输出

        if not self.training:
            if feats.size(0) == 2048:
                feats = self.bn_neck(feats.permute(1, 0))
                logits = self.classifier(feats)
                return logits  # feats #


            else:
                feats = self.bn_neck(
                    feats) # 归一化
                return feats

        else:
            return self.train_forward(feats, alpha, f_sh, f_sp,p_mod, sp_pl, labels,sub, **kwargs)



    def train_forward(self, feat, alpha, f_sh, f_sp, p_mod, sp_pl ,labels,sub, **kwargs):
        epoch = kwargs.get('epoch')
        metric = {}
        loss = 0

        metric.update({'alpha': alpha.data})
        t_sub = sub.long()

        sp_logits = self.special_D(f_sp) #F_sh
        sp_loss = self.id_loss(sp_logits.float(), t_sub) #鼓励判别器识别不出sh
        loss += sp_loss
        metric.update({'sp_loss': sp_loss.data})

        # P_logits_mean = p_mod.mean(dim=1)  # [B] logits的均值
        # is_rgb = (sub == 0).float()  # RGB=1, IR=0
        #
        # # ✅ 关键: 用 with_logits 版本 (支持AMP)
        # L_mc = F.binary_cross_entropy_with_logits(P_logits_mean, is_rgb)
        # loss += 1.0 * L_mc
        # metric.update({'L_mc': L_mc.data})

        if self.triplet:

            triplet_loss, _, _, _ = self.triplet_loss(feat.float(), labels) # 共享特征
            trip_loss = triplet_loss
            if self.decompose:
                triplet_loss_im, _, _, _ = self.triplet_loss(sp_pl.float(), labels)
                trip_loss += triplet_loss_im
            loss += trip_loss
            metric.update({'tri': trip_loss.data})

        if hasattr(self, 'circle_criterion'):
            circle_loss = self.circle_criterion(
                feat.float(),  # 共享特征
                labels,  # ID标签
                sub.long()  # 模态标签（会自动区分模态内/跨模态）
            )
            loss += circle_loss
            circle_loss = circle_loss * 0.15
            metric.update({'circle': circle_loss.data})

        if self.align:
            v_labels = labels[sub == 0]
            i_labels = labels[sub == 1]

            align_loss = self.modality_align_loss(feat[sub == 0], feat[sub == 1], v_labels,i_labels)
            align_loss = align_loss * 0.2
            loss += align_loss
            metric.update({'align_loss': align_loss.data})

        bb = 120  #90

        if self.TGSA:
            sf_sh_dist_v = kl_soft_dist(feat[sub == 0], feat[sub == 0])
            sf_sh_dist_i = kl_soft_dist(feat[sub == 1], feat[sub == 1])
            #可见光和红外光共享各取一半拼接
            half_B0 = feat[sub == 0].shape[0] // 2
            feat_half0 = feat[sub == 0][:half_B0]
            half_B1 = feat[sub == 1].shape[0] // 2
            feat_half1 = feat[sub == 1][:half_B1]
            feat_cross = torch.cat((feat_half0, feat_half1), dim=0)
            sf_sh_dist_vi = kl_soft_dist(feat_cross, feat_cross) #得到跨模态共享特征的距离分布
            _, kl_intra1 = Bg_kl(sf_sh_dist_v, sf_sh_dist_i)
            _, kl_intra2 = Bg_kl(sf_sh_dist_v, sf_sh_dist_vi)
            _, kl_intra3 = Bg_kl(sf_sh_dist_vi, sf_sh_dist_i)
            kl_intra = kl_intra1 + kl_intra2 + kl_intra3
            if self.decompose:
                sf_sp_dist_v = kl_soft_dist(sp_pl[sub == 0], sp_pl[sub == 0])  # 可见光模态特定特征的非对角距离分布
                sf_sp_dist_i = kl_soft_dist(sp_pl[sub == 1], sp_pl[sub == 1])
                _, kl_inter_v = Bg_kl(sf_sh_dist_v, sf_sp_dist_v) #模型特征模仿特定特征 对齐visible
                _, kl_inter_i = Bg_kl(sf_sh_dist_i, sf_sp_dist_i) #模型特征模仿特定特征 对齐infrared
                kl_inter = kl_inter_v + kl_inter_i
            # 如果当前批次是完整的（例如 120），使用正常的加权方式。
            if feat.size(0) == bb:
                soft_dt = kl_intra
                if self.decompose:
                    soft_dt += kl_inter * 0.6
            # 批次不完整采取更小的权重
            else:
                soft_dt = kl_intra1 * 0.1
                if self.decompose:
                    soft_dt += kl_inter * 0.1

            loss += soft_dt
            metric.update({'soft_dt': soft_dt.data})

        # 最小化每个样本与其自身类别中心之间的距离
        # 它没有明确惩罚不同类别中心之间的距离。如果单独使用，所有类别的中心可能会聚集在一起，导致类间分离度不足
        if self.center_loss:
            center_loss = self.center_loss(feat.float(), labels)
            loss += center_loss
            metric.update({'cen': center_loss.data})
        # 最大化不同类别中心之间的距离，从而优化类间分离
        if self.center_cluster:
            center_cluster_loss, _, _ = self.center_cluster_loss(feat.float(), labels)
            loss += center_cluster_loss
            metric.update({'cc': center_cluster_loss.data})

        # 同时约束模态共享特征（feat）和模态特定特征（sp_pl）的判别力
        if self.fb_dt and self.decompose:
            loss_f_d_ap = Fb_dt(feat, labels) #模态共享特征
            loss_Fb_im = Fb_dt(sp_pl, labels) #模态特定特征
            fb_loss = loss_f_d_ap + loss_Fb_im
            loss += fb_loss

            metric.update({'f_dt': fb_loss.data})

        feat = self.bn_neck(feat)
        if self.decompose:
            sp_pl = self.bn_neck_sp(sp_pl)
        sub_nb = sub + 0  ##模态标签

        if self.decompose:
            logits_sp = self.classifier_sp(sp_pl)  # self.bn_neck_un(sp_pl) sp分类得分
            loss_id_sp = self.id_loss(logits_sp.float(), labels) # 交叉熵损失
            sp_logits = self.D_special(sp_pl)  #F_sp输入给判别器
            discr_loss = self.id_loss(sp_logits.float(), sub_nb) #鼓励判别器识别sp
            metric.update({'discriminate_loss': discr_loss.data})
            metric.update({'loss_id_sp': loss_id_sp.data})
            loss += loss_id_sp + discr_loss

        pseu_sh_logits = self.D_shared_pseu(feat) #F_sh
        p_sub = sub_nb.chunk(2)[0].repeat_interleave(2) #构造标签
        pp_sub = torch.roll(p_sub, -1) #反转标签
        pseu_loss = self.id_loss(pseu_sh_logits.float(), pp_sub) #鼓励判别器识别不出sh
        loss += pseu_loss
        metric.update({'pseudo_loss': pseu_loss.data})

        if self.classification:
            logits = self.classifier(feat)
            if self.CSA1:
                _, intra_bg = Bg_kl(logits[sub == 0], logits[sub == 1])  # 共享和红外对齐
                bg_loss = intra_bg
            if self.CSA2:
                _, intra_Sm = Sm_kl(logits[sub == 0], logits[sub == 1], labels)  # 模态互相学习
                sm_kl_loss = intra_Sm
            if self.decompose:
                if self.CSA1:
                    _, inter_bg_v = Bg_kl(logits[sub == 0], logits_sp[sub == 0]) #强制共享可见光 Logits 模仿特定可见光 Logits
                    _, inter_bg_i = Bg_kl(logits[sub == 1], logits_sp[sub == 1]) #强制共享红外光 Logits 模仿特定可见光 Logits
                    inter_bg = inter_bg_v + inter_bg_i
                    if feat.size(0) == bb:
                        bg_loss += inter_bg * 0.8
                    else:
                        bg_loss += inter_bg * 0.3

                if self.CSA2:
                    _, inter_Sm_v = Sm_kl(logits[sub == 0], logits_sp[sub == 0], labels) # sh Logits 在批次内语义结构上模仿sp Logits
                    _, inter_Sm_i = Sm_kl(logits[sub == 1], logits_sp[sub == 1], labels)
                    inter_Sm = inter_Sm_v + inter_Sm_i # 隐式蒸馏
                    if feat.size(0) == bb:
                        sm_kl_loss += inter_Sm * 0.8
                    else:
                        sm_kl_loss += inter_Sm * 0.3

            if self.CSA1:
                loss += bg_loss
                metric.update({'bg_kl': bg_loss.data})
            if self.CSA2:
                sm_kl_loss = sm_kl_loss
                loss += sm_kl_loss
                metric.update({'sm_kl': sm_kl_loss.data})
            cls_loss = self.id_loss(logits.float(), labels) # 基础识别id的能力
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'id_loss': cls_loss.data})

        return loss, metric #对应engine代码下的返回损失和指标