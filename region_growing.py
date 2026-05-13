"""
region_growing.py — 区域生长分割 (纯几何, 无监督)
流程: 滤波 → RANSAC去平面 → KNN+PCA算法向量 → 区域生长 → 可视化
"""
import logging
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 预处理
# ═══════════════════════════════════════════════════════
def preprocess(pcd, voxel_size=0.03):
    logger.info("预处理: 体素降采样 + 统计滤波")
    logger.info(f"  原始点数: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    logger.info(f"  预处理后: {len(pcd.points)} 点")
    return pcd


# ═══════════════════════════════════════════════════════
# 2. 分辨率估算
# ═══════════════════════════════════════════════════════
def estimate_resolution(pcd, k=20):
    logger.info("估算点云分辨率...")
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    n = min(500, len(pts))
    idx = np.random.choice(len(pts), n, replace=False)
    avg_dists = [np.mean(np.sqrt(tree.search_knn_vector_3d(pts[i], k)[2]))
                 for i in idx]
    res = np.median(avg_dists)
    logger.info(f"  分辨率: {res:.4f}")
    return res


# ═══════════════════════════════════════════════════════
# 3. RANSAC 去平面
# ═══════════════════════════════════════════════════════
def ransac_remove_planes(pcd, resolution, n_planes=6):
    ransac_thresh = resolution
    logger.info(f"RANSAC 去平面: {n_planes}轮, threshold={ransac_thresh:.4f}")

    rest = pcd
    planes = []
    for i in range(n_planes):
        plane_model, inliers = rest.segment_plane(
            distance_threshold=ransac_thresh, ransac_n=3, num_iterations=1000)
        plane_pcd = rest.select_by_index(inliers)
        planes.append(plane_pcd)
        rest = rest.select_by_index(inliers, invert=True)
        a, b, c, d = plane_model
        logger.info(f"  平面{i+1}: {a:.2f}x+{b:.2f}y+{c:.2f}z+{d:.2f}=0, "
                    f"内点{len(inliers)}")

    logger.info(f"  剩余物体点: {len(rest.points)}")
    return rest, planes


# ═══════════════════════════════════════════════════════
# 4. KNN + PCA 算法向量
# ═══════════════════════════════════════════════════════
def compute_normals_knn_pca(pcd, k=30):
    """
    对每个点: 找 K 近邻 → PCA 分解协方差矩阵
    → 最小特征值对应的特征向量 = 法向量
    → 最小特征值 / 特征值之和 = 曲率
    """
    logger.info(f"KNN+PCA 算法向量 (k={k})...")
    points = np.asarray(pcd.points)
    n_pts = len(points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    normals = np.zeros((n_pts, 3))
    curvatures = np.zeros(n_pts)

    for i in range(n_pts):
        _, idx, _ = tree.search_knn_vector_3d(points[i], k)
        neighbors = points[idx]                          # (k, 3)
        centered = neighbors - neighbors.mean(axis=0)    # 去中心化
        cov = centered.T @ centered / k                  # 协方差矩阵 (3,3)

        eigvals, eigvecs = np.linalg.eigh(cov)           # 升序: λ0 ≤ λ1 ≤ λ2
        normals[i] = eigvecs[:, 0]                       # 最小特征值 → 法向量
        curvatures[i] = eigvals[0] / (eigvals.sum() + 1e-10)

    pcd.normals = o3d.utility.Vector3dVector(normals)
    logger.info(f"  曲率范围: [{curvatures.min():.4f}, {curvatures.max():.4f}]")
    return pcd, curvatures


# ═══════════════════════════════════════════════════════
# 5. 区域生长
# ═══════════════════════════════════════════════════════
def region_growing(pcd, curvatures,
                   angle_threshold_deg=15.0,
                   curvature_threshold=0.05,
                   min_cluster_size=50,
                   max_cluster_size=100000,
                   k=30):
    """
    区域生长分割

    参数:
      angle_threshold_deg: 相邻点法向量夹角阈值 (度)
      curvature_threshold: 种子点最大曲率
      min_cluster_size:    最小簇点数 (过滤噪点簇)
      max_cluster_size:    最大簇点数 (防止面积过大)
      k:                   邻域搜索点数
    """
    logger.info(f"区域生长: angle={angle_threshold_deg}°, curv={curvature_threshold}, "
                f"min={min_cluster_size}, max={max_cluster_size}, k={k}")

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_pts = len(points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    angle_thresh = np.deg2rad(angle_threshold_deg)

    # 曲率排序: 低曲率优先做种子
    sorted_idx = np.argsort(curvatures)

    visited = np.zeros(n_pts, dtype=bool)
    labels = np.full(n_pts, -1, dtype=int)
    current_label = -1

    for seed_idx in sorted_idx:
        if visited[seed_idx]:
            continue
        if curvatures[seed_idx] > curvature_threshold:
            break  # 后面曲率都更大, 不再长新的

        current_label += 1
        seed_queue = [seed_idx]
        visited[seed_idx] = True
        labels[seed_idx] = current_label

        while seed_queue:
            cur = seed_queue.pop(0)
            _, neighbors, _ = tree.search_knn_vector_3d(points[cur], k)

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

    # 过滤簇
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    removed_small = 0
    removed_large = 0
    for lbl, cnt in zip(unique, counts):
        if cnt < min_cluster_size:
            labels[labels == lbl] = -1
            removed_small += 1
        elif cnt > max_cluster_size:
            labels[labels == lbl] = -1
            removed_large += 1

    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = np.sum(labels == -1)
    logger.info(f"  结果: {n_clusters} 个区域, {n_noise} 个噪声点 "
                f"({n_noise/n_pts*100:.1f}%), "
                f"过滤小簇{removed_small}个, 大簇{removed_large}个")
    return labels


# ═══════════════════════════════════════════════════════
# 6. 可视化
# ═══════════════════════════════════════════════════════
def colorize(pcd, labels):
    result = copy.deepcopy(pcd)
    valid = labels >= 0
    if not valid.any():
        result.paint_uniform_color([0, 0, 0])
        return result
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((len(labels), 3))
    for i in np.where(valid)[0]:
        colors[i] = cmap((labels[i] % 20) / 20)[:3]
    result.colors = o3d.utility.Vector3dVector(colors)
    return result


def visualize(pcd_raw, planes, pcd_objects, labels):
    offset = 8.0

    pcd_all_planes = o3d.geometry.PointCloud()
    plane_colors = plt.get_cmap("tab10")(np.linspace(0, 1, 6))
    for i, p in enumerate(planes):
        p_ = copy.deepcopy(p)
        p_.paint_uniform_color(plane_colors[i][:3])
        pcd_all_planes += p_

    pcd_raw_copy = copy.deepcopy(pcd_raw)
    pcd_all_planes.translate((offset, 0, 0))

    pcd_clusters = colorize(pcd_objects, labels)
    pcd_clusters.translate((offset * 2, 0, 0))

    logger.info("可视化: 原始 | 6平面 | 区域生长结果")
    o3d.visualization.draw_geometries(
        [pcd_raw_copy, pcd_all_planes, pcd_clusters],
        window_name="原始点云 | RANSAC 6平面 | 区域生长",
        width=1600, height=900)


# ═══════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    PCD_PATH = "Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt"

    # 1. 加载
    logger.info(f"加载: {PCD_PATH.split('/')[-1]}")
    data = np.loadtxt(PCD_PATH)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    if data.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)

    # 2. 预处理
    pcd = preprocess(pcd, voxel_size=0.03)

    # 3. 分辨率 + RANSAC
    resolution = estimate_resolution(pcd)
    pcd_objects, planes = ransac_remove_planes(pcd, resolution, n_planes=6)

    # 4. KNN+PCA 算法向量
    pcd_objects, curvatures = compute_normals_knn_pca(pcd_objects, k=30)

    # 5. 区域生长
    labels = region_growing(
        pcd_objects, curvatures,
        angle_threshold_deg=15.0,
        curvature_threshold=0.05,
        min_cluster_size=50,
        max_cluster_size=100000,
        k=30)

    # 6. 可视化
    visualize(pcd, planes, pcd_objects, labels)
    logger.info("完成")
