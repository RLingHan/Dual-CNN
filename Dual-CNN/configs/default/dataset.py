from yacs.config import CfgNode
#yacs库（一个用于配置管理的工具）定义不同数据集的配置参数，方便后续在程序中统一管理和使用这些配置

#CfgNode是yacs库的核心类，用于创建配置节点
dataset_cfg = CfgNode()

# config for dataset
dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
# dataset_cfg.sysu.data_root = "../dataset/SYSU-MM01"
dataset_cfg.sysu.data_root = "/kaggle/input/sysu-mm01/SYSU-MM01"


dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = "/kaggle/input/regdb001/RegDB"

dataset_cfg.llcm = CfgNode()
dataset_cfg.llcm.num_id = 713
dataset_cfg.llcm.num_cam = 2
dataset_cfg.llcm.data_root = "../dataset/LLCM"

#后续在程序中可以通过dataset_cfg.数据集名称.参数名的方式快速访问这些配置

