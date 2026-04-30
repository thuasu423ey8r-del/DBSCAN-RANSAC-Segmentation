"""
pipeline.py — 场景分割公共流程
加载 / 预处理 / RANSAC / 特征 / mIoU 评测 全部在这里
"""
import logging
import os
import open3d as o3d
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

SEMANTIC_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "bookcase",
    "sofa", "board", "clutter"
]
STRUCTURAL = {"ceiling", "floor", "wall"}
CLASS_TO_IDX = {c: i for i, c in enumerate(SEMANTIC_CLASSES)}


# ═══════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════
def load_pcd(pcd_path):
    """加载 S3DIS .txt 点云"""
    data = np.loadtxt(pcd_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    if data.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)
    return pcd


def load_gt(room_dir):
    """从 Annotations 构建 per-point GT (inst_labels, cls_labels)"""
    full_path = os.path.join(room_dir, os.path.basename(room_dir) + ".txt")
    full_xyz = np.loadtxt(full_path)[:, :3]

    lut = {tuple(np.round(xyz, 4)): i for i, xyz in enumerate(full_xyz)}
    inst_lbl = np.full(len(full_xyz), -1, dtype=int)
    cls_lbl = np.full(len(full_xyz), -1, dtype=int)
    anno_dir = os.path.join(room_dir, "Annotations")

    inst_id = 0
    for fname in sorted(os.listdir(anno_dir)):
        if not fname.endswith('.txt'):
            continue
        cls_name = fname.rsplit('_', 1)[0]
        if cls_name not in CLASS_TO_IDX:
            continue
        for xyz in np.loadtxt(os.path.join(anno_dir, fname))[:, :3]:
            key = tuple(np.round(xyz, 4))
            if key in lut:
                idx = lut[key]
                inst_lbl[idx] = inst_id
                cls_lbl[idx] = CLASS_TO_IDX[cls_name]
        inst_id += 1

    n_inst = len(np.unique(inst_lbl[inst_lbl >= 0]))
    logger.info(f"  GT 加载: {n_inst} 个实例")
    return full_xyz, inst_lbl, cls_lbl


# ═══════════════════════════════════════════════════════
# 2. 预处理 + 参数估算
# ═══════════════════════════════════════════════════════
def preprocess(pcd, voxel_size=0.03):
    """降采样 + 统计滤波"""
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def estimate_resolution(pcd, k=20):
    """估算点云分辨率 (平均近邻距离)"""
    pts = np.asarray(pcd.points)
    n = min(500, len(pts))
    tree = o3d.geometry.KDTreeFlann(pcd)
    idx = np.random.choice(len(pts), n, replace=False)
    avg_dists = [np.mean(np.sqrt(tree.search_knn_vector_3d(pts[i], k)[2]))
                 for i in idx]
    res = np.median(avg_dists)
    logger.info(f"  点云分辨率: {res:.4f}")
    return res


# ═══════════════════════════════════════════════════════
# 3. RANSAC 去平面
# ═══════════════════════════════════════════════════════
def ransac_remove_planes(pcd, resolution, n_planes=6):
    """循环 RANSAC 提取平面，返回 (物体点云, 平面列表)"""
    ransac_thresh = resolution * 2.0
    logger.info(f"  RANSAC: {n_planes} 个平面, thresh={ransac_thresh:.4f}")

    rest = pcd
    planes = []
    for i in range(n_planes):
        plane_model, inliers = rest.segment_plane(
            distance_threshold=ransac_thresh, ransac_n=3, num_iterations=1000)
        plane_pcd = rest.select_by_index(inliers)
        planes.append(plane_pcd)
        rest = rest.select_by_index(inliers, invert=True)
        a, b, c, d = plane_model
        logger.info(f"    平面 {i+1}: {a:.2f}x+{b:.2f}y+{c:.2f}z+{d:.2f}=0, "
                    f"内点 {len(inliers)}")

    logger.info(f"  物体点数: {len(rest.points)}")
    return rest, planes


# ═══════════════════════════════════════════════════════
# 4. 特征构建
# ═══════════════════════════════════════════════════════
def build_features(pcd, normal_radius):
    """构建 9D 特征: XYZ(3D) + 法向量外积矩阵(6D)"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))

    xyz = np.asarray(pcd.points)
    n = np.asarray(pcd.normals)
    w = 0.1
    tensor = np.column_stack((
        n[:, 0]**2, n[:, 1]**2, n[:, 2]**2,
        n[:, 0]*n[:, 1], n[:, 0]*n[:, 2], n[:, 1]*n[:, 2])) * w
    return np.concatenate((xyz, tensor), axis=1)


# ═══════════════════════════════════════════════════════
# 5. mIoU 评测
# ═══════════════════════════════════════════════════════
def compute_miou(room_dir, pred_labels, pcd_objects):
    """计算物体实例 mIoU (排除 wall/floor/ceiling)"""
    orig_xyz, gt_inst, gt_cls = load_gt(room_dir)

    # GT 对齐到物体点云 (最近邻)
    orig_pcd = o3d.geometry.PointCloud()
    orig_pcd.points = o3d.utility.Vector3dVector(orig_xyz)
    orig_tree = o3d.geometry.KDTreeFlann(orig_pcd)

    f_xyz = np.asarray(pcd_objects.points)
    gt_inst_f = np.full(len(f_xyz), -1, dtype=int)
    gt_cls_f = np.full(len(f_xyz), -1, dtype=int)
    for i, pt in enumerate(f_xyz):
        _, idx, _ = orig_tree.search_knn_vector_3d(pt, 1)
        gt_inst_f[i] = gt_inst[idx[0]]
        gt_cls_f[i] = gt_cls[idx[0]]

    # 只取非结构类
    obj_mask = np.zeros(len(gt_cls_f), dtype=bool)
    for cls_name in SEMANTIC_CLASSES:
        if cls_name not in STRUCTURAL:
            obj_mask |= (gt_cls_f == CLASS_TO_IDX[cls_name])

    logger.info(f"  物体点: {obj_mask.sum()} / {len(f_xyz)}")

    pred_v = pred_labels[obj_mask]
    gt_inst_v = gt_inst_f[obj_mask]

    gt_ids = sorted(set(gt_inst_v[gt_inst_v >= 0]))
    pred_ids = sorted(set(pred_v[pred_v >= 0]))
    logger.info(f"  GT 实例: {len(gt_ids)}, 预测簇: {len(pred_ids)}")

    # IoU 矩阵
    iou = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gid in enumerate(gt_ids):
        gm = gt_inst_v == gid
        for j, pid in enumerate(pred_ids):
            pm = pred_v == pid
            inter = np.sum(gm & pm)
            union = np.sum(gm | pm)
            iou[i, j] = inter / union if union > 0 else 0

    row, col = linear_sum_assignment(-iou)

    ious = [iou[r, c] for r, c in zip(row, col) if iou[r, c] > 0]
    miou = np.mean(ious) * 100 if ious else 0

    logger.info(f"  mIoU: {miou:.1f}% ({len(ious)}/{len(gt_ids)} 匹配)")
    if ious:
        logger.info(f"  IoU 范围: {min(ious)*100:.0f}% ~ {max(ious)*100:.0f}%, "
                    f"中位数: {np.median(ious)*100:.0f}%")
    return miou
