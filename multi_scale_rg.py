import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import deque
import copy
import time

# ═══════════════════════════════════════════════════════
# 0. 日志的配置
# ═══════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 核心特征与算法组件
# ═══════════════════════════════════════════════════════

def get_pcd_features(pcd, k=30):
    """计算法向量和曲率"""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(points))

    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(points[i], k)
        if len(idx) < 3: continue
        cov = np.cov(points[idx].T)
        eigvals, _ = np.linalg.eigh(cov)
        curvatures[i] = eigvals[0] / (np.sum(eigvals) + 1e-10)
    return curvatures


def standard_region_growing(pcd, curvatures, angle_deg, curv_thresh, min_size, k=30):
    """标准区域生长：用于第一阶段（严阈值）"""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree = o3d.geometry.KDTreeFlann(pcd)
    angle_thresh = np.deg2rad(angle_deg)

    labels = np.full(len(points), -1, dtype=int)
    visited = np.zeros(len(points), dtype=bool)
    current_label = 0
    seed_indices = np.argsort(curvatures)

    for s_idx in seed_indices:
        if visited[s_idx] or curvatures[s_idx] > curv_thresh: continue

        queue = deque([s_idx])
        visited[s_idx] = True
        labels[s_idx] = current_label
        cluster = [s_idx]

        while queue:
            curr = queue.popleft()
            _, neighbors, _ = tree.search_knn_vector_3d(points[curr], k)
            for nb in neighbors:
                if visited[nb]: continue
                dot = np.abs(np.dot(normals[curr], normals[nb]))
                if np.arccos(np.clip(dot, 0.0, 1.0)) < angle_thresh:
                    visited[nb] = True
                    labels[nb] = current_label
                    cluster.append(nb)
                    if curvatures[nb] < curv_thresh:
                        queue.append(nb)

        if len(cluster) < min_size:
            for idx in cluster: labels[idx] = -1
        else:
            current_label += 1
    return labels


def guided_region_growing(pcd, initial_labels, angle_deg, k=30):
    """引导生长：用于第二阶段（宽阈值）"""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree = o3d.geometry.KDTreeFlann(pcd)
    angle_thresh = np.deg2rad(angle_deg)

    labels = initial_labels.copy()
    queue = deque(np.where(labels > 0)[0])

    while queue:
        curr = queue.popleft()
        _, neighbors, _ = tree.search_knn_vector_3d(points[curr], k)
        for nb in neighbors:
            if labels[nb] == 0:  # 只向无人区生长
                dot = np.abs(np.dot(normals[curr], normals[nb]))
                if np.arccos(np.clip(dot, 0.0, 1.0)) < angle_thresh:
                    labels[nb] = labels[curr]
                    queue.append(nb)
    return labels


def colorize(pcd, labels):
    """上色逻辑：Label 0 设为灰色，其他彩色"""
    res = copy.deepcopy(pcd)
    max_label = labels.max()
    if max_label <= 0:
        res.paint_uniform_color([0.8, 0.8, 0.8])
        return res
    # 使用 tab20 颜色表
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        if labels[i] <= 0:
            colors[i] = [0.7, 0.7, 0.7]  # 灰色
        else:
            colors[i] = cmap((labels[i] % 20) / 20.0)[:3]
    res.colors = o3d.utility.Vector3dVector(colors)
    return res


# ═══════════════════════════════════════════════════════
# 2. 主流水线流程
# ═══════════════════════════════════════════════════════

def run_multi_scale_pipeline(path):
    start_time = time.time()
    logger.info("=" * 50)
    logger.info("开始多尺度区域生长流程")
    logger.info("=" * 50)

    # --- Step 0: 数据载入与洗数据 ---
    raw_data = np.loadtxt(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_data[:, :3])
    logger.info(f"0.1 加载数据成功: {len(pcd.points)} 点")

    fine_pcd = pcd.voxel_down_sample(0.02)
    fine_pcd, _ = fine_pcd.remove_statistical_outlier(20, 2.0)
    logger.info(f"0.2 精细采样完成 (2cm): {len(fine_pcd.points)} 点")

    # --- Step 1: 粗点云骨架提取 ---
    logger.info("1.1 执行粗采样 (6cm)...")
    coarse_pcd = fine_pcd.voxel_down_sample(0.06)
    curv_coarse = get_pcd_features(coarse_pcd, k=20)

    logger.info("1.2 粗分割阶段: 严阈值寻找骨架中心...")
    coarse_labels = standard_region_growing(coarse_pcd, curv_coarse, angle_deg=8.0, curv_thresh=0.04, min_size=50)
    num_seeds = coarse_labels.max() + 1
    logger.info(f"    --> 粗点云中发现种子簇: {num_seeds} 个")

    pcd_step1_coarse = colorize(coarse_pcd, coarse_labels)  # 骨架图 (粗)

    # --- Step 2: 标签映射 (增加距离阈值限制) ---
    logger.info("2.1 执行 KNN 映射 (增加 5cm 距离限制)...")
    coarse_tree = o3d.geometry.KDTreeFlann(coarse_pcd)
    fine_labels_skeleton = np.zeros(len(fine_pcd.points), dtype=int)
    fine_pts = np.asarray(fine_pcd.points)

    # 设定一个距离门槛：比如 0.05m (5cm)
    # 如果精细点离最近的粗点超过 5cm，说明它们可能不是同一个物体，保持 Label 0
    max_mapping_dist = 0.05

    for i in range(len(fine_pts)):
        # search_knn_vector_3d 返回: (返回码, 索引, 距离的平方)
        _, idx, dist_sq = coarse_tree.search_knn_vector_3d(fine_pts[i], 1)

        if np.sqrt(dist_sq[0]) < max_mapping_dist:
            res = coarse_labels[idx[0]]
            fine_labels_skeleton[i] = res + 2 if res >= 0 else 0
        else:
            fine_labels_skeleton[i] = 0  # 太远了，宁愿保持灰色

    for i in range(len(fine_pts)):
        _, idx, _ = coarse_tree.search_knn_vector_3d(fine_pts[i], 1)
        res = coarse_labels[idx[0]]
        fine_labels_skeleton[i] = res + 2 if res >= 0 else 0

    pcd_step2_fine_skeleton = colorize(fine_pcd, fine_labels_skeleton)  # 骨架映射图 (细)
    logger.info(f"    --> 映射完成，种子覆盖点数: {np.sum(fine_labels_skeleton > 0)}")

    # --- Step 3: 引导生长 (补全边缘) ---
    logger.info("3.1 引导生长阶段: 宽阈值修复边缘 (15度)...")
    _ = get_pcd_features(fine_pcd, k=10)
    fine_labels_guided = guided_region_growing(fine_pcd, fine_labels_skeleton, angle_deg=15.0)

    pcd_step3_guided = colorize(fine_pcd, fine_labels_guided)  # 生长补全图
    logger.info(
        f"    --> 生长完成，未分配点数从 {np.sum(fine_labels_skeleton == 0)} 降至 {np.sum(fine_labels_guided == 0)}")

    # --- Step 4: 碎点处理 (抓小件) ---
    logger.info("4.1 扫尾工作: 执行 DBSCAN 抓取独立小物件...")
    final_labels = fine_labels_guided.copy()
    zero_mask = (final_labels == 0)
    if zero_mask.any():
        zero_indices = np.where(zero_mask)[0]
        trash_cloud = fine_pcd.select_by_index(zero_indices)
        obj_labels = np.array(trash_cloud.cluster_dbscan(eps=0.08, min_points=20))

        max_lbl = final_labels.max()
        for i, o_lbl in enumerate(obj_labels):
            if o_lbl != -1:
                final_labels[zero_indices[i]] = max_lbl + 1 + o_lbl

    pcd_step4_final = colorize(fine_pcd, final_labels)  # 最终图
    logger.info(f"    --> 最终识别区域总数: {final_labels.max()}")

    # --- 可视化布局 ---
    offset = 6.0
    pcd_step1_coarse.translate((-offset, 0, 0))  # 放到最左边
    pcd_step2_fine_skeleton.translate((0, 0, 0))
    pcd_step3_guided.translate((offset, 0, 0))
    pcd_step4_final.translate((offset * 2, 0, 0))

    logger.info(f"总耗时: {time.time() - start_time:.2f} 秒")
    logger.info("展示顺序: [1.粗点云骨架] -> [2.映射至精细点云] -> [3.引导生长补全] -> [4.最终碎点处理]")

    o3d.visualization.draw_geometries(
        [pcd_step1_coarse, pcd_step2_fine_skeleton, pcd_step3_guided, pcd_step4_final],
        window_name="进化对比: 粗骨架 -> 映射 -> 生长 -> 补漏",
        width=1600, height=900
    )


if __name__ == "__main__":
    # 请修改为您电脑上的 S3DIS 路径
    FILE_PATH = "Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt"
    run_multi_scale_pipeline(FILE_PATH)