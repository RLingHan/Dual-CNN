import os

import torch
import torchvision.transforms as T

#引入了针对特定数据集（如 SYSU、RegDB 等）的处理类，以及不同的数据采样策略
from torch.utils.data import DataLoader
from data.dataset import SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import LLCMData
from data.dataset import MarketDataset

from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import RandomIdentitySampler
from data.sampler import NormTripletSampler
import random

#构建数据加载相关的工具函数

def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch)) #矩阵转置
    #对于剩下的字段，使用 PyTorch 的 stack 函数，沿着第 0 维（批次维度）将所有数据堆叠起来，形成一个大的 Tensor
    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data
    #将图像、标签等数值类型的数据堆叠（stack）成 PyTorch Tensor，方便 GPU 计算
    #将文件路径等字符串类型的数据保留为列表，因为字符串不能堆叠成 Tensor

#对彩色图像(RGB)的颜色通道进行随机变换或灰度化
class ChannelAdapGray(object):
    def __init__(self, probability=0.5):
        self.probability = probability
    #魔术方法 定义后类可以当函数一样调用
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :] #表示获取图像的第一个通道上的全部元素
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img

#根据输入的配置参数，构建并返回一个可直接用于模型训练的DataLoader
def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=4):
    if True==False: #tsne 注释
        #T为transform
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        # data pre-processing
        t = [T.Resize(image_size)] #这里将转换加入到列表，方便Compose后续
        # Compose的作用就是接收一个变换列表，并把它们按顺序应用到图像上

        if random_flip:
            t.append(T.RandomHorizontalFlip())

        if color_jitter:
            # 随机调整亮度、对比度、饱和度（幅度0.1）， hue=0表示不调整色调，避免颜色失真
            t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

        if random_crop:#t.extend(...): 表示调用列表t的扩展方法
            t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

        t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #在标准化过程中，函数会用图像的每个像素值 减去 对应通道的这个平均值
        #在减去均值之后，函数会用图像的每个像素值 除以 对应通道的这个标准差
        if random_erase:
            t.append(T.RandomErasing())
            #t.append(ChannelAdapGray(probability=0.5)) ###58
            # t.append(Jigsaw())

        transform = T.Compose(t)
    # # data pre-processing
    # t = [T.Resize(image_size)]
    #
    # if random_flip:
    #     t.append(T.RandomHorizontalFlip())
    #
    # if color_jitter:
    #     t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    #
    # if random_crop:
    #     t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])
    #
    # t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #
    # if random_erase:
    #     t.append(T.RandomErasing())
    #     # t.append(Jigsaw())
    #
    # transform = T.Compose(t)

    # dataset
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform)
    elif dataset == 'llcm':
        train_dataset = LLCMData(root, mode='train', transform=transform)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform)

    # sampler
    assert sample_method in ['random', 'identity_uniform', 'identity_random', 'norm_triplet']
    if sample_method == 'identity_uniform':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_random':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader


def get_test_loader(dataset, root, batch_size, image_size, num_workers=4):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform)
        query_dataset = RegDBDataset(root, mode='query', transform=transform)
    elif dataset == 'llcm':
        gallery_dataset = LLCMData(root, mode='gallery', transform=transform)
        query_dataset = LLCMData(root, mode='query', transform=transform)
    elif dataset == 'market':
        gallery_dataset = MarketDataset(root, mode='gallery', transform=transform)
        query_dataset = MarketDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
