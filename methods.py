"""
methods.py — 聚类算法集合
每个函数签名统一: fn(features, **kwargs) -> labels
"""
import logging
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

logger = logging.getLogger(__name__)


# ── DBSCAN 系列 ───────────────────────────────────
def dbscan_xyz(features):
    """纯 XYZ DBSCAN"""
    return DBSCAN(eps=0.15, min_samples=10, n_jobs=-1).fit_predict(features[:, :3])

def dbscan_9d(features):
    """9D 特征 DBSCAN (XYZ + 法向量外积)"""
    return DBSCAN(eps=0.15, min_samples=10, n_jobs=-1).fit_predict(features)


# ── HDBSCAN 系列 ──────────────────────────────────
def hdbscan_xyz(features):
    """纯 XYZ HDBSCAN"""
    return HDBSCAN(min_cluster_size=10, min_samples=5, n_jobs=-1).fit_predict(features[:, :3])

def hdbscan_9d(features):
    """9D 特征 HDBSCAN (XYZ + 法向量外积)"""
    return HDBSCAN(min_cluster_size=10, min_samples=5, n_jobs=-1).fit_predict(features)


# ── 区域生长 ──────────────────────────────────────
def region_growing(pcd, angle_threshold_deg=15.0, curvature_threshold=0.05,
                   min_cluster_size=10, k=30, normal_radius=0.15):
    """基于法向夹角 + 曲率的区域生长"""
    import open3d as o3d

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 曲率
    curvatures = np.zeros(len(points))
    for i in range(len(points)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], k)
        cov = np.cov(points[idx].T)
        eigvals = np.linalg.eigvalsh(cov)
        curvatures[i] = eigvals[0] / (eigvals.sum() + 1e-10)

    angle_thresh = np.deg2rad(angle_threshold_deg)
    sorted_idx = np.argsort(curvatures)

    visited = np.zeros(len(points), dtype=bool)
    labels = np.full(len(points), -1, dtype=int)
    current_label = -1

    for seed_idx in sorted_idx:
        if visited[seed_idx]:
            continue
        if curvatures[seed_idx] > curvature_threshold:
            break

        current_label += 1
        seed_queue = [seed_idx]
        visited[seed_idx] = True
        labels[seed_idx] = current_label

        while seed_queue:
            cur = seed_queue.pop(0)
            _, neighbors, _ = pcd_tree.search_knn_vector_3d(points[cur], k)
            for nb in neighbors:
                if visited[nb]:
                    continue
                visited[nb] = True
                dot = np.abs(np.dot(normals[cur], normals[nb]))
                angle = np.arccos(np.clip(dot, 0.0, 1.0))
                if angle < angle_thresh:
                    labels[nb] = current_label
                    if curvatures[nb] < curvature_threshold:
                        seed_queue.append(nb)

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    for lbl, cnt in zip(unique, counts):
        if cnt < min_cluster_size:
            labels[labels == lbl] = -1

    return labels
