import os
import re
import os.path as osp
from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
'''

'''
继承自Dataset方法
Dataset 类定义了 PyTorch 的 数据加载器 (DataLoader) 所期望的接口。
必需的方法：一个自定义的 Dataset 类通常需要实现（或重写）至少两个核心方法，这是 DataLoader 能够工作的基础：
__len__(self)：返回数据集的大小（样本总数）。
__getitem__(self, index)：根据给定的索引 index 返回单个样本（例如图像、标签等）。
'''
class SYSUDataset(Dataset):
    #构造函数
    def __init__(self, root, mode='train', transform=None) :
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            #处理ID字符串：去除换行符并按逗号分割成列表
            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids #合并列表
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        #将选中的id转化为整数形式  比如0001转化为1
        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        #**：这是最关键的部分，它代表零个或多个目录
        #只有当设置 recursive=True 时，glob 函数才会识别和处理模式中的
        #glob根据指定的模式（通配符）来查找文件和目录
        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        #路径格式假设为：.../身份ID/图像文件.jpg，通过split('/')[-2]获取身份ID
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        # 路径格式为：.../camX/身份ID/图像文件.jpg，split('/')[-3]获取摄像头目录名
        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

        #构造图库
        img_paths = sorted(img_paths)

        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]#图像对应的摄像头标号
        self.num_ids = num_ids #选中身份的数量
        self.transform = transform

        if mode == 'train':
            #range(num_ids) 生成一个从 0 开始，到 num_ids - 1 结束的连续整数序列
            #zip() 函数将这两个序列 按顺序配对 组合在一起
            #实例：id_map = {1001: 0, 1005: 1, 1012: 2, 1018: 3, 1020: 4}
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths] #对应人物身份的id
            #其中包含了 img_paths 中每张图片所对应人物的 新的、从 0 开始的整数标签
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        #转化为 torch.long 是为了满足 PyTorch 损失函数 和 网络层 对输入标签和索引的数据类型要求
        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths]
        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class LLCMData(Dataset):
    def __init__(self, root, mode='train', transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_vis.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_nir.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_vis.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_nir.txt','r'))


        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)
        # path = '/home/zhang/E/RKJ/MAPnet/dataset/LLCM/nir/0351/0351_c06_s200656_f4830_nir.jpg'
        # img = Image.open(path).convert('RGB')
        # img = np.array(img, dtype=np.uint8)
        # import pdb
        # pdb.set_trace()

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'nir') + 2 for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))

            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class MarketDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        self.transform = transform

        if mode == 'train':
            img_paths = glob(os.path.join(root, 'bounding_box_train/*.jpg'), recursive=True)
        elif mode == 'gallery':
            img_paths = glob(os.path.join(root, 'bounding_box_test/*.jpg'), recursive=True)
        elif mode == 'query':
            img_paths = glob(os.path.join(root, 'query/*.jpg'), recursive=True)

        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        relabel = mode == 'train'
        self.img_paths = []
        self.cam_ids = []
        self.ids = []
        for fpath in img_paths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            self.img_paths.append(fpath)
            self.ids.append(all_pids[pid])
            self.cam_ids.append(cam - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
