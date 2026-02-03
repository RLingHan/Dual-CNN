# engine/__init__.py
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import time

# 假设评估函数和配置已存在
from utils.eval_sysu import eval_sysu
from utils.eval_regdb import eval_regdb
from utils.eval_llcm import eval_llcm
from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg


# -------------------------- 自定义指标类 --------------------------
class ScalarMetric:
    def __init__(self):
        self.sum_metric = 0.0
        self.sum_inst = 0

    def update(self, value):
        self.sum_metric += value
        self.sum_inst += 1

    def reset(self):
        self.sum_metric = 0.0
        self.sum_inst = 0

    def compute(self):
        if self.sum_inst == 0:
            raise ValueError("No data to compute metric")
        return self.sum_metric / self.sum_inst


class IgnoreAccuracy:
    def __init__(self, ignore_index=-1):
        self.ignore_index = ignore_index
        self._num_correct = 0
        self._num_examples = 0

    def update(self, y_pred, y):
        if y_pred.dim() == 2:
            indices = torch.argmax(y_pred, dim=1)
        else:
            indices = torch.round(y_pred).type(y.type())

        correct = torch.eq(indices, y).view(-1)
        ignore = torch.eq(y, self.ignore_index).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0] - ignore.sum().item()

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def compute(self):
        if self._num_examples == 0:
            raise ValueError("No data to compute accuracy")
        return self._num_correct / self._num_examples


class AutoKVMetric:
    def __init__(self):
        self.kv_sum_metric = {}
        self.kv_sum_inst = {}

    def update(self, output):
        if not isinstance(output, dict):
            raise TypeError("Output must be a dictionary")
        for k, v in output.items():
            if k not in self.kv_sum_metric:
                self.kv_sum_metric[k] = 0.0
                self.kv_sum_inst[k] = 0
            self.kv_sum_metric[k] += v
            self.kv_sum_inst[k] += 1

    def reset(self):
        self.kv_sum_metric.clear()
        self.kv_sum_inst.clear()

    def compute(self):
        result = {}
        for k in self.kv_sum_metric:
            if self.kv_sum_inst[k] == 0:
                continue
            result[k] = self.kv_sum_metric[k] / self.kv_sum_inst[k]
        return result


# -------------------------- 核心训练/评估逻辑 --------------------------
def create_train_step(model, optimizer, scaler, device, non_blocking=False):
    def train_step(batch, epoch, iteration):
        model.train()
        data, labels, cam_ids, _, _ = batch  # 解析batch数据

        # 数据转移到设备
        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        # 学习率warmup和权重衰减调整（还原原逻辑）
        warmup = False  # 保持原代码的开关控制
        if warmup:
            # 学习率warmup
            warm_iteration = 30 * 213  # 原代码固定参数
            if epoch < 21:
                lr = 0.00035 * iteration / warm_iteration
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # 权重衰减分阶段调整
            def adjust_weight_decay(epoch, initial_wd):
                if epoch > 15:
                    return initial_wd / 100
                elif 5 < epoch <= 15:
                    return initial_wd / 10
                else:
                    return initial_wd

            new_wd = adjust_weight_decay(epoch, 0.5)  # 原代码初始权重衰减0.5
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = new_wd

        # mutual_learning参数更新（原逻辑）
        if hasattr(model, 'mutual_learning') and model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_

        # 前向传播与反向传播（PyTorch 2.x AMP）
        optimizer.zero_grad()
        with autocast():
            loss, metric = model(data, labels, cam_ids=cam_ids, epoch=epoch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return metric

    return train_step


def create_eval_step(model, device, non_blocking=False):
    def eval_step(batch):
        model.eval()
        data, labels, cam_ids, img_paths = batch[:4]

        # 数据转移到设备
        data = data.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        # 无梯度提取特征
        with torch.no_grad(), autocast():
            feat = model(data, cam_ids=cam_ids)

        return feat.data.float().cpu(), labels, cam_ids.cpu(), np.array(img_paths)

    return eval_step


# -------------------------- 主训练函数 --------------------------
def train(
        dataset,
        model,
        train_loader,
        optimizer,
        lr_scheduler=None,
        logger=None,
        writer=None,
        non_blocking=False,
        log_period=10,
        save_dir="checkpoints",
        prefix="model",
        gallery_loader=None,
        query_loader=None,
        eval_interval=1,
        start_eval=0,
        max_epochs=100,
        rerank=False
):
    # 设备初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = GradScaler()  # PyTorch 2.x原生AMP
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Using device: {device}")

    # 初始化训练/评估步骤
    train_step = create_train_step(model, optimizer, scaler, device, non_blocking)
    eval_step = create_eval_step(model, device, non_blocking)

    # 训练状态初始化
    best_rank1 = 0.0
    kv_metric = AutoKVMetric()

    # 主训练循环
    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        kv_metric.reset()
        batch_timer = time.time()
        total_iteration = (epoch - 1) * len(train_loader)  # 总迭代数计数

        # 训练迭代
        for iter_in_epoch, batch in enumerate(train_loader, 1):
            total_iteration += 1  # 更新总迭代数
            metric = train_step(batch, epoch, total_iteration)
            kv_metric.update(metric)

            # 日志打印
            if iter_in_epoch % log_period == 0:
                batch_size = batch[0].size(0)
                speed = log_period * batch_size / (time.time() - batch_timer)
                msg = f"Epoch[{epoch}] Batch [{iter_in_epoch}/{len(train_loader)}]\tSpeed: {speed:.2f} samples/sec"

                metric_dict = kv_metric.compute()
                for k in sorted(metric_dict.keys()):
                    msg += f"\t{k}: {metric_dict[k]:.4f}"
                    if writer:
                        writer.add_scalar(f'metric/{k}', metric_dict[k], total_iteration)

                logger.info(msg)
                kv_metric.reset()
                batch_timer = time.time()

        # Epoch结束处理
        logger.info(f"Epoch {epoch} finished in {time.time() - epoch_start_time:.2f}s")

        # 学习率调度
        if lr_scheduler:
            lr_scheduler.step()

        # 模型保存
        if epoch % 20 == 0:
            save_path = os.path.join(save_dir, f"{prefix}_model_{epoch}.pth")
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")

        # 评估逻辑
        if (gallery_loader and query_loader
                and epoch % eval_interval == 0
                and epoch >= start_eval
                and len(gallery_loader) > 0
                and len(query_loader) > 0):

            # 提取查询集特征
            q_feats, q_ids, q_cams, q_img_paths = [], [], [], []
            for batch in query_loader:
                feat, labels, cams, paths = eval_step(batch)
                q_feats.append(feat)
                q_ids.append(labels)
                q_cams.append(cams)
                q_img_paths.append(paths)
            q_feats = torch.cat(q_feats, dim=0)
            q_ids = torch.cat(q_ids, dim=0).numpy()
            q_cams = torch.cat(q_cams, dim=0).numpy()
            q_img_paths = np.concatenate(q_img_paths, axis=0)

            # 提取gallery集特征
            g_feats, g_ids, g_cams, g_img_paths = [], [], [], []
            for batch in gallery_loader:
                feat, labels, cams, paths = eval_step(batch)
                g_feats.append(feat)
                g_ids.append(labels)
                g_cams.append(cams)
                g_img_paths.append(paths)
            g_feats = torch.cat(g_feats, dim=0)
            g_ids = torch.cat(g_ids, dim=0).numpy()
            g_cams = torch.cat(g_cams, dim=0).numpy()
            g_img_paths = np.concatenate(g_img_paths, axis=0)

            # 数据集评估
            current_r1 = 0.0
            mAP = 0.0
            if dataset == 'sysu':
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                mAP, r1, r5, _, _ = eval_sysu(
                    q_feats, q_ids, q_cams, g_feats, g_ids, g_cams,
                    g_img_paths, perm, mode='all', num_shots=1, rerank=rerank
                )
                current_r1 = r1
            elif dataset == 'regdb':
                print('infrared to visible')
                mAP1, r1_1, _, _, _ = eval_regdb(
                    q_feats, q_ids, q_cams, g_feats, g_ids, g_cams,
                    g_img_paths, rerank=rerank
                )
                print('visible to infrared')
                mAP2, r1_2, _, _, _ = eval_regdb(
                    g_feats, g_ids, g_cams, q_feats, q_ids, q_cams,
                    q_img_paths, rerank=rerank
                )
                current_r1 = (r1_1 + r1_2) / 2
                mAP = (mAP1 + mAP2) / 2
            elif dataset == 'llcm':
                print('infrared to visible')
                mAP1, r1_1, _, _, _ = eval_llcm(
                    q_feats, q_ids, q_cams, g_feats, g_ids, g_cams,
                    g_img_paths, rerank=rerank  # 修复rerank参数引用错误
                )
                print('visible to infrared')
                mAP2, r1_2, _, _, _ = eval_llcm(
                    g_feats, g_ids, g_cams, q_feats, q_ids, q_cams,
                    q_img_paths, rerank=rerank
                )
                current_r1 = (r1_1 + r1_2) / 2
                mAP = (mAP1 + mAP2) / 2

            # 更新最佳模型
            if current_r1 > best_rank1:
                best_rank1 = current_r1
                best_path = os.path.join(save_dir, "model_best.pth")
                try:
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"Best model updated (Rank1: {best_rank1:.4f})")
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")

            # 记录TensorBoard
            if writer:
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', current_r1, epoch)

            # 清理内存
            del q_feats, g_feats, q_ids, g_ids, q_cams, g_cams
            torch.cuda.empty_cache()

    logger.info(f"Training completed. Best Rank1: {best_rank1:.4f}")
    return best_rank1  # 返回最佳指标用于同步


# -------------------------- 兼容层（模拟原get_trainer接口） --------------------------
class TrainerWrapper:
    """包装训练逻辑，模拟原Ignite Engine接口"""

    def __init__(self, train_func, *args, **kwargs):
        self.train_func = train_func
        self.args = args
        self.kwargs = kwargs
        # 模拟原Engine的state属性
        self.state = type('', (), {})()
        self.state.best_rank1 = 0.0  # 同步最佳指标

    def run(self, train_loader, max_epochs):
        """模拟原engine.run()方法"""
        best_rank1 = self.train_func(
            *self.args,
            train_loader=train_loader,
            max_epochs=max_epochs, **self.kwargs
        )
        self.state.best_rank1 = best_rank1  # 同步最佳指标到state


def get_trainer(
        dataset,
        model,
        optimizer,
        lr_scheduler=None,
        logger=None,
        writer=None,
        non_blocking=False,
        log_period=10,
        save_dir="checkpoints",
        prefix="model",
        gallery_loader=None,
        query_loader=None,
        eval_interval=None,
        start_eval=None,
        rerank=False
):
    """兼容原get_trainer接口"""
    # 参数校验（保持原逻辑）
    if not isinstance(eval_interval, int):
        raise TypeError("The parameter 'eval_interval' must be type INT.")
    if not isinstance(start_eval, int):
        raise TypeError("The parameter 'start_eval' must be type INT.")

    # 初始化日志（确保不为None）
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())

    return TrainerWrapper(
        train,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        writer=writer,
        non_blocking=non_blocking,
        log_period=log_period,
        save_dir=save_dir,
        prefix=prefix,
        gallery_loader=gallery_loader,
        query_loader=query_loader,
        eval_interval=eval_interval,
        start_eval=start_eval,
        rerank=rerank
    )


# -------------------------- 使用示例 --------------------------
"""
if __name__ == "__main__":
    # 示例：替换为实际模型和数据加载器
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 10)
        def forward(self, x, cam_ids=None, epoch=None):
            return torch.tensor(0.5), {"loss": 0.5, "acc": 0.8}

    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader([(torch.randn(3,256), torch.tensor([0]), torch.tensor([0]), "", "") for _ in range(10)])
    gallery_loader = train_loader
    query_loader = train_loader

    # 原代码调用方式
    trainer = get_trainer(
        dataset="sysu",
        model=model,
        optimizer=optimizer,
        gallery_loader=gallery_loader,
        query_loader=query_loader,
        eval_interval=2,
        start_eval=1
    )
    trainer.run(train_loader, max_epochs=5)
"""