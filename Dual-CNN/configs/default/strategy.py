from yacs.config import CfgNode

# 创建策略配置根节点
strategy_cfg = CfgNode()

# 实验标识前缀（用于区分不同实验版本）
strategy_cfg.prefix = "baseline"

# --------------------------
# 数据加载器（dataloader）设置
# --------------------------
strategy_cfg.sample_method = "random"  # 数据采样方法：随机采样
strategy_cfg.batch_size = 128          # 批次大小
strategy_cfg.p_size = 16               # 每个批次中的身份(ID)数量
strategy_cfg.k_size = 8                # 每个身份在批次中的样本数量

# --------------------------
# 损失函数设置
# --------------------------
strategy_cfg.classification = True     # 启用分类损失
strategy_cfg.triplet = False           # 禁用三元组损失
strategy_cfg.center_cluster = False    # 禁用中心聚类损失
strategy_cfg.center = False            # 禁用中心损失
strategy_cfg.sm_kl = False             # 禁用与softmax结合的KL损失
strategy_cfg.bg_kl = False             # 禁用背景相关的KL损失
strategy_cfg.IP = False                # 禁用内积损失
strategy_cfg.decompose = False         # 禁用特征分解损失
strategy_cfg.fb_dt = False             # 禁用前后景差异损失
strategy_cfg.distalign = False         # 禁用距离对齐损失



# --------------------------
# 度量学习设置
# --------------------------
strategy_cfg.margin = 0.3              # 度量学习中的间隔值
strategy_cfg.weight_KL = 3.0           # KL散度损失的权重
strategy_cfg.weight_sid = 1.0          # SID损失的权重
strategy_cfg.weight_sep = 1.0          # 分离损失的权重
strategy_cfg.update_rate = 1.0         # 动态参数更新速率

# --------------------------
# 优化器设置
# --------------------------
strategy_cfg.optimizer = "sgd"         # 使用SGD优化器
strategy_cfg.lr = 0.1                  # 初始学习率
strategy_cfg.wd = 5e-3                 # 权重衰减系数（L2正则化）
# 5e-4  # 备用权重衰减系数
strategy_cfg.lr_step = [40]            # 学习率衰减的epoch节点

# --------------------------
# 训练基本设置
# --------------------------
strategy_cfg.fp16 = False              # 禁用FP16混合精度训练
strategy_cfg.num_epoch = 60            # 总训练轮次

# --------------------------
# 数据集设置
# --------------------------
strategy_cfg.dataset = "sysu"          # 使用的数据集（SYSU-MM01）
strategy_cfg.image_size = (384, 128)   # 输入图像尺寸 (高度, 宽度)

# --------------------------
# 数据增强设置
# --------------------------
strategy_cfg.random_flip = True        # 启用随机翻转
strategy_cfg.random_crop = True        # 启用随机裁剪
strategy_cfg.random_erase = True       # 启用随机擦除
strategy_cfg.color_jitter = False      # 禁用颜色抖动
strategy_cfg.padding = 10              # 图像填充像素数

# --------------------------
# 网络结构设置
# --------------------------
strategy_cfg.drop_last_stride = False  # 不禁用最后一层的步长
strategy_cfg.pattern_attention = False # 禁用模式注意力机制
strategy_cfg.modality_attention = 0    # 模态注意力机制模式（0表示禁用）
strategy_cfg.mutual_learning = False   # 禁用互学习
strategy_cfg.rerank = True            # 禁用重排序后处理
strategy_cfg.num_parts = 6             # 图像分割的部分数量

# --------------------------
# 日志设置
# --------------------------
strategy_cfg.eval_interval = -1        # 评估间隔（-1表示不按固定间隔评估）
strategy_cfg.start_eval = 60           # 开始评估的epoch
strategy_cfg.log_period = 10           # 日志记录间隔（迭代次数）

# --------------------------
# 测试/恢复训练设置
# --------------------------
strategy_cfg.resume = ''               # 恢复训练的权重文件路径（空表示从头训练）
# 历史权重文件路径参考：
# /home/zhang/E/RKJ/MAPnet/MPA-LL2-cvpr/checkpoints/regdb/RegDB/model_best.pth
# /root/MPANet/MPA-LL2-cvpr/checkpoints/sysu/SYSU/model_best.pth
# /root/MPANet/MPA-LL2-cvpr/checkpoints/llcm/LLCM/model_best.pth
# /home/zhang/E/RKJ/MAPnet/MPA-cvpr/checkpoints/llcm/LLCM/model_best.pth