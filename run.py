"""
run.py — 横向对比多种聚类算法
"""
import logging
import numpy as np
from pipeline import (load_pcd, preprocess, estimate_resolution,
                       ransac_remove_planes, build_features, compute_miou)
from methods import (dbscan_xyz, dbscan_9d, hdbscan_xyz, hdbscan_9d,
                     region_growing)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 配置
ROOM_DIR = "Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1"
ROOM_PATH = f"{ROOM_DIR}/{ROOM_DIR.split('/')[-1]}.txt"

# ── 加载 + 预处理 + RANSAC (所有算法共享) ────────
logger.info("加载数据...")
pcd = load_pcd(ROOM_PATH)
logger.info(f"  原始: {len(pcd.points)} 点")

logger.info("预处理...")
pcd = preprocess(pcd, voxel_size=0.05)
logger.info(f"  预处理后: {len(pcd.points)} 点")

resolution = estimate_resolution(pcd)
normal_radius = resolution * 6.0
logger.info(f"  法线半径: {normal_radius:.4f}")

logger.info("RANSAC 去平面...")
pcd_objects, planes = ransac_remove_planes(pcd, resolution)

logger.info("构建 9D 特征...")
features = build_features(pcd_objects, normal_radius)

# ── 逐个算法跑 ──────────────────────────────────
results = []

# 基于 feature 数组的聚类
for name, fn in [
    ("DBSCAN XYZ",      lambda: dbscan_xyz(features)),
    ("DBSCAN 9D",       lambda: dbscan_9d(features)),
    ("HDBSCAN XYZ",     lambda: hdbscan_xyz(features)),
    ("HDBSCAN 9D",      lambda: hdbscan_9d(features)),
]:
    logger.info(f"\n{'='*50}")
    logger.info(f"算法: {name}")
    labels = fn()
    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = np.sum(labels == -1)
    logger.info(f"  簇: {n_clusters}, 噪声: {n_noise} "
                f"({n_noise/len(labels)*100:.1f}%)")

    miou = compute_miou(ROOM_DIR, labels, pcd_objects)
    results.append((name, miou, n_clusters))

# 区域生长 (需要点云对象，不一样)
logger.info(f"\n{'='*50}")
logger.info("算法: Region Growing")
rg_labels = region_growing(pcd_objects, normal_radius=normal_radius)
n_clusters = len(np.unique(rg_labels[rg_labels >= 0]))
n_noise = np.sum(rg_labels == -1)
logger.info(f"  簇: {n_clusters}, 噪声: {n_noise} "
            f"({n_noise/len(rg_labels)*100:.1f}%)")
miou = compute_miou(ROOM_DIR, rg_labels, pcd_objects)
results.append(("Region Growing", miou, n_clusters))

# ── 汇总 ────────────────────────────────────────
print(f"\n{'算法':<20} {'mIoU':>8} {'簇数':>6}")
print("-" * 37)
for name, miou, n_clusters in results:
    print(f"{name:<20} {miou:>7.1f}% {n_clusters:>5d}")
print("-" * 37)
