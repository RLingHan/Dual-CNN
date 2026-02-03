import numpy as np

import copy
from torch.utils.data import Sampler
from collections import defaultdict


class CrossModalityRandomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.rgb_list = []
        self.ir_list = []
        for i, cam in enumerate(dataset.cam_ids):
            if cam in [3, 6]:
                self.ir_list.append(i)
            else:
                self.rgb_list.append(i)
    #先对红外和可见光分类

    def __len__(self):
        #取两种模态中样本数较多的那个，乘以 2（确保两种模态的样本都能被充分利用）
        return max(len(self.rgb_list), len(self.ir_list)) * 2

    def __iter__(self):
        sample_list = []
        #对RGB和红外样本的索引进行随机打乱
        rgb_list = np.random.permutation(self.rgb_list).tolist()
        ir_list = np.random.permutation(self.ir_list).tolist()

        rgb_size = len(self.rgb_list)
        ir_size = len(self.ir_list)
        #让 RGB 和红外样本的数量完全一致，避免模型偏向样本多的模态。
        if rgb_size >= ir_size:
            diff = rgb_size - ir_size
            reps = diff // ir_size
            pad_size = diff % ir_size #还需要补充的数量
            for _ in range(reps):
                ir_list.extend(np.random.permutation(self.ir_list).tolist())
            ir_list.extend(np.random.choice(self.ir_list, pad_size, replace=False).tolist()) #不放回抽样，抽取pad_size次
        else:
            diff = ir_size - rgb_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                rgb_list.extend(np.random.permutation(self.rgb_list).tolist())
            rgb_list.extend(np.random.choice(self.rgb_list, pad_size, replace=False).tolist())

        assert len(rgb_list) == len(ir_list)

        half_bs = self.batch_size // 2
        #每次训练各占一半 前后顺序
        for start in range(0, len(rgb_list), half_bs):
            sample_list.extend(rgb_list[start:start + half_bs])
            sample_list.extend(ir_list[start:start + half_bs])

        return iter(sample_list) #sampler子类约定俗成


class CrossModalityIdentitySampler(Sampler):
    def __init__(self, dataset, p_size, k_size):
        self.dataset = dataset
        self.p_size = p_size #每个批次包含的身份数
        self.k_size = k_size // 2  #每个身份的单模态的数量
        self.batch_size = p_size * k_size * 2
        #当访问一个新键时，defaultdict 会自动调用 list() 函数，创建一个 空列表 [] 作为该键的值
        self.id2idx_rgb = defaultdict(list)
        self.id2idx_ir = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            if dataset.cam_ids[i] in [3, 6]:
                self.id2idx_ir[identity].append(i)
            else:
                self.id2idx_rgb[identity].append(i)

    def __len__(self):
        return self.dataset.num_ids * self.k_size * 2

    def __iter__(self):
        sample_list = []

        id_perm = np.random.permutation(self.dataset.num_ids) #打乱身份
        for start in range(0, self.dataset.num_ids, self.p_size):
            selected_ids = id_perm[start:start + self.p_size] #按身份选取批次，每次取p_size个身份

            sample = []
            for identity in selected_ids:
                replace = len(self.id2idx_rgb[identity]) < self.k_size
                s = np.random.choice(self.id2idx_rgb[identity], size=self.k_size, replace=replace)
                sample.extend(s) #允许重复抽样保证rgb数量达到k_size

            sample_list.extend(sample)

            sample.clear()
            for identity in selected_ids:
                replace = len(self.id2idx_ir[identity]) < self.k_size
                s = np.random.choice(self.id2idx_ir[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

        return iter(sample_list)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source  # 数据源（同dataset，含ids、cam_ids）
        self.batch_size = batch_size  # 批次总大小
        self.num_instances = num_instances  # 每个身份在批次中占的总样本数
        # 计算每个批次包含的身份数量（批次总样本数 ÷ 每个身份的样本数）
        self.num_pids_per_batch = self.batch_size // self.num_instances

        # 构建“身份→样本索引”的映射（分模态）
        self.index_dic_R = defaultdict(list)  # RGB模态：{身份ID: [样本索引列表]}
        self.index_dic_I = defaultdict(list)  # 红外模态：{身份ID: [样本索引列表]}
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:  # 红外样本
                self.index_dic_I[identity].append(i)
            else:  # RGB样本
                self.index_dic_R[identity].append(i)

        self.pids = list(self.index_dic_I.keys())  # 将id键提出来作为列表 以红外模态的身份为基准

        # 估算一个epoch的总样本数（避免批次不完整）
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_I[pid]  # 以红外样本数为参考
            num = len(idxs)
            # 若样本数不足num_instances，按num_instances算（后续会重复采样）
            if num < self.num_instances:
                num = self.num_instances
            # 确保总样本数是num_instances的倍数（每个身份的样本能完整组成批次）
            self.length += num - num % self.num_instances

    def __len__(self):
        return self.length

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)  # 存储每个身份的“完整样本组”：{身份ID: [样本组列表]}

        for pid in self.pids:  # 遍历每个身份 身份是捆绑的
            # 深拷贝该身份的两种模态样本索引（避免修改原始映射）
            idxs_I = copy.deepcopy(self.index_dic_I[pid])  # 红外样本索引
            idxs_R = copy.deepcopy(self.index_dic_R[pid])  # RGB样本索引

            # 情况1：两种模态样本都不足 → 都重复采样（凑够每个模态所需数量）
            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                # 每个模态需要的样本数 = 总实例数 ÷ 2（严格1:1）
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)

            # 情况2：模态数量不平衡 → 裁剪多的一方（确保两种模态数量相等）
            if len(idxs_I) > len(idxs_R):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)  # 裁剪红外
            if len(idxs_R) > len(idxs_I):
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)  # 裁剪RGB

            # 打乱两种模态的样本顺序（增加随机性）
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)

            # 交替组合两种模态，生成“单身份样本组”
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I, idxs_R):
                batch_idxs.append(idx_I)  # 先加1个红外样本
                batch_idxs.append(idx_R)  # 再加1个RGB样本
                # 当样本组长度达到num_instances（每个身份的总样本数），保存并重置
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)  # 该身份的一个完整样本组
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)  # 可用身份列表（初始为所有身份）
        final_idxs = []  # 存储最终的采样索引

        # 循环生成批次，直到可用身份不足num_pids_per_batch（每个批次需的身份数）
        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机选num_pids_per_batch个身份（比如3个身份）
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                # 取出该身份的第一个样本组（FIFO原则）
                batch_idxs = batch_idxs_dict[pid].pop(0) #batch_idxs_dict是交替进入队伍
                final_idxs.extend(batch_idxs)  # 加入最终列表
                # 若该身份没有更多样本组，从可用列表中移除（避免重复使用）
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        # 更新实际总样本数（覆盖初始化时的估算值）
        self.length = len(final_idxs)
        return iter(final_idxs)  # 返回采样索引迭代器


class NormTripletSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.data_source.ids):
            self.index_dic[pid].append(index)  #加入对应分类
        self.pids = list(self.index_dic.keys())  #取出对应id

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length