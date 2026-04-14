import os
# import logging # <--- 移除 logging 模块
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader

from data import get_test_loader
from models.baseline import Baseline
from utils.eval_sysu import eval_sysu


def extract_features(model, data_loader, device):
    """提取特征"""
    model.eval()
    features_list = []
    ids_list = []
    cams_list = []
    img_paths_list = []

    with torch.no_grad():
        for batch in data_loader:
            data, labels, cam_ids, img_paths = batch[:4]
            data = data.to(device)
            cam_ids = cam_ids.to(device)

            # 提取特征
            feat = model(data, cam_ids=cam_ids)

            features_list.append(feat.cpu())
            ids_list.append(labels)
            cams_list.append(cam_ids.cpu())
            img_paths_list.append(np.array(img_paths))

    features = torch.cat(features_list, dim=0)
    ids = torch.cat(ids_list, dim=0).numpy()
    cams = torch.cat(cams_list, dim=0).numpy()
    img_paths = np.concatenate(img_paths_list, axis=0)

    return features, ids, cams, img_paths


def evaluate_sysu_all_modes(checkpoint_path, cfg):
    """
    对SYSU数据集进行四种模式的完整评估

    Args:
        checkpoint_path: 模型checkpoint路径
        cfg: 配置对象
    """
    # 移除日志设置
    # logging.basicConfig(
    #     format="%(asctime)s %(message)s",
    #     level=logging.INFO
    # )
    # logger = logging.getLogger()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}") # <--- 使用 print() 替代 logger.info()

    # 加载数据
    print("[INFO] Loading test data...") # <--- 使用 print() 替代 logger.info()
    gallery_loader, query_loader = get_test_loader(
        dataset=cfg.dataset,
        root=cfg.data_root,
        batch_size=64,
        image_size=cfg.image_size,
        num_workers=4
    )

    # 创建模型
    print("[INFO] Creating model...") # <--- 使用 print() 替代 logger.info()
    model = Baseline(
        num_classes=cfg.num_id,
        pattern_attention=cfg.get('pattern_attention', 0),
        modality_attention=cfg.get('modality_attention', 0),
        mutual_learning=cfg.get('mutual_learning', False),
        decompose=cfg.get('decompose', False),
        drop_last_stride=cfg.get('drop_last_stride', False),
        triplet=cfg.get('triplet', False),
        k_size=cfg.get('k_size', 8),
        center_cluster=cfg.get('center_cluster', False),
        center=cfg.get('center', False),
        margin=cfg.get('margin', 0.3),
        num_parts=cfg.get('num_parts', 0),
        weight_KL=cfg.get('weight_KL', 0),
        weight_sid=cfg.get('weight_sid', 0),
        weight_sep=cfg.get('weight_sep', 0),
        update_rate=cfg.get('update_rate', 0),
        classification=cfg.get('classification', False),
        bg_kl=cfg.get('bg_kl', False),
        sm_kl=cfg.get('sm_kl', False),
        fb_dt=cfg.get('fb_dt', False),
        IP=cfg.get('IP', False),
        distalign=cfg.get('distalign', False),
        use_gca=cfg.get('use_gca', False),
        gca_k=cfg.get('gca_k', 8),
        gca_temp=cfg.get('gca_temp', 0.07),
        use_lab=cfg.get('use_lab', False),
        eval=True  # 评估模式
    )

    # 加载checkpoint
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}") # <--- 使用 print() 替代 logger.info()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # 提取查询集特征
    print("[INFO] Extracting query features...") # <--- 使用 print() 替代 logger.info()
    q_feats, q_ids, q_cams, q_img_paths = extract_features(
        model, query_loader, device
    )

    # 提取gallery集特征
    print("[INFO] Extracting gallery features...") # <--- 使用 print() 替代 logger.info()
    g_feats, g_ids, g_cams, g_img_paths = extract_features(
        model, gallery_loader, device
    )

    # 加载随机排列矩阵
    perm_path = os.path.join(cfg.data_root, 'exp', 'rand_perm_cam.mat')
    perm = sio.loadmat(perm_path)['rand_perm_cam']

    # 定义评估配置
    eval_configs = [
        {'mode': 'all', 'num_shots': 1, 'name': 'All-Search (Single-Shot)'},
        {'mode': 'all', 'num_shots': 10, 'name': 'All-Search (Multi-Shot)'},
        {'mode': 'indoor', 'num_shots': 1, 'name': 'Indoor-Search (Single-Shot)'},
        {'mode': 'indoor', 'num_shots': 10, 'name': 'Indoor-Search (Multi-Shot)'}
    ]

    # 存储结果
    results = {}

    # 执行四种评估
    print("\n" + "=" * 80) # <--- 使用 print() 替代 logger.info()
    print("Starting SYSU-MM01 Evaluation") # <--- 使用 print() 替代 logger.info()
    print("=" * 80 + "\n") # <--- 使用 print() 替代 logger.info()

    for config in eval_configs:
        mode = config['mode']
        num_shots = config['num_shots']
        name = config['name']

        print(f"\n{'=' * 60}") # <--- 使用 print() 替代 logger.info()
        print(f"Evaluating: {name}") # <--- 使用 print() 替代 logger.info()
        print(f"{'=' * 60}") # <--- 使用 print() 替代 logger.info()

        # 不使用rerank的评估
        print(f"\n[Without Re-ranking]") # <--- 使用 print() 替代 logger.info()
        mAP, r1, r5, r10, r20 = eval_sysu(
            q_feats, q_ids, q_cams,
            g_feats, g_ids, g_cams,
            g_img_paths, perm,
            mode=mode,
            num_shots=num_shots,
            rerank=False,
        )

        results[f"{name}_no_rerank"] = {
            'mAP': mAP,
            'r1': r1,
            'r5': r5,
            'r10': r10,
            'r20': r20
        }

        # 使用rerank的评估
        print(f"\n[With Re-ranking]") # <--- 使用 print() 替代 logger.info()
        mAP_rr, r1_rr, r5_rr, r10_rr, r20_rr = eval_sysu(
            q_feats, q_ids, q_cams,
            g_feats, g_ids, g_cams,
            g_img_paths, perm,
            mode=mode,
            num_shots=num_shots,
            rerank=True
        )

        results[f"{name}_rerank"] = {
            'mAP': mAP_rr,
            'r1': r1_rr,
            'r5': r5_rr,
            'r10': r10_rr,
            'r20': r20_rr
        }

    # 打印总结
    print("\n" + "=" * 80) # <--- 使用 print() 替代 logger.info()
    print("EVALUATION SUMMARY") # <--- 使用 print() 替代 logger.info()
    print("=" * 80 + "\n") # <--- 使用 print() 替代 logger.info()

    for eval_name, metrics in results.items():
        print(f"{eval_name}:")
        print(f"  Rank-1: {metrics['r1']:.2f}%")
        print(f"  Rank-5: {metrics['r5']:.2f}%")
        print(f"  Rank-10: {metrics['r10']:.2f}%")
        print(f"  Rank-20: {metrics['r20']:.2f}%")
        print(f"  mAP: {metrics['mAP']:.2f}%")
        print("")

    return results


if __name__ == '__main__':
    # ... (保持不变, 但需要确保顶部移除了 `import logging`)
    import argparse
    import yaml
    import random
    from configs.default import strategy_cfg, dataset_cfg as ds_cfg

    parser = argparse.ArgumentParser(description='Evaluate SYSU-MM01 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., model_best.pth)')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file (e.g., configs/SYSU.yml)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device id')

    args = parser.parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 设置随机种子 (与训练保持一致)
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # 加载配置 (与train.py完全一致的方式)
    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    # 合并数据集配置
    dataset_cfg = ds_cfg.get(cfg.dataset)
    for k, v in dataset_cfg.items():
        cfg[k] = v

    cfg.freeze()

    print(f"\nDataset: {cfg.dataset}")
    print(f"Data root: {cfg.data_root}")
    print(f"Number of identities: {cfg.num_id}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # 执行评估
    results = evaluate_sysu_all_modes(args.checkpoint, cfg)