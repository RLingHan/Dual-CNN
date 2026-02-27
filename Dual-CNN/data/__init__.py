import os

import torch
import torchvision.transforms as T

#引入了针对特定数据集（如 SYSU、RegDB 等）的处理类，以及不同的数据采样策略
from torch.utils.data import DataLoader

from data.ChannelAug import ChannelRandomErasing, ChannelExchange, ChannelAdapGray
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




#根据输入的配置参数，构建并返回一个可直接用于模型训练的DataLoader
def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=4):

    t = [T.Resize(image_size)]
    if random_flip:
        t.append(T.RandomHorizontalFlip())
    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    if random_crop:  # t.extend(...): 表示调用列表t的扩展方法
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])
    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if random_erase:
        t.append(T.RandomErasing())

    transform = T.Compose(t)

    # ── 可见光 transform（含通道增强）──────────────────────
    # 可见光：CRE + CA，不加GA
    t_color = [T.Resize(image_size)]
    if random_flip:
        t_color.append(T.RandomHorizontalFlip())
    if color_jitter:
        t_color.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    if random_crop:
        t_color.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])
    t_color.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if random_erase:
        t_color.append(T.RandomErasing())
    t_color.append(ChannelRandomErasing(probability=0.5))  # CRE
    t_color.append(ChannelExchange())  # CA：75%交换+25%保留
    # 不加 ChannelAdapGray
    transform_color = T.Compose(t_color)

    # 红外：CRE + GA，不加CA
    t_thermal = [T.Resize(image_size)]
    if random_flip:
        t_thermal.append(T.RandomHorizontalFlip())
    if random_crop:
        t_thermal.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])
    t_thermal.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if random_erase:
        t_thermal.append(T.RandomErasing())
    t_thermal.append(ChannelRandomErasing(probability=0.5))  # CRE
    t_thermal.append(ChannelAdapGray(probability=0.5))  # GA
    # 不加 ChannelExchange
    transform_thermal = T.Compose(t_thermal)
    # dataset
    # ── 构建dataset ────────────────────────────────────────
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train',
                                    transform=transform_color,
                                    transform_thermal=transform_thermal)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train',
                                     transform=transform,
                                     transform_thermal=transform)
    elif dataset == 'llcm':
        train_dataset = LLCMData(root, mode='train', transform=transform_color)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform_color)

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
