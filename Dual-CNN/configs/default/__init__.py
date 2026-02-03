from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg

#__init__.py 必须存在才能让目录被识别为包
# 定义公共接口（允许外部通过*导入的内容）
#只会导入__all__中列出的4个函数
__all__ = ["dataset_cfg", "strategy_cfg"]

#yaml文件在默认配置基础上定制化实验参数，通常会在程序启动时被加载并与config包中的默认配置合并
