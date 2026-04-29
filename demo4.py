import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import DBSCAN


def pipeline_with_structure_tensor(pcd_path):
    # 1. 加载 S3DIS .txt 点云
    print("-> 正在读取 S3DIS 原始数据...")
    data = np.loadtxt(pcd_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    if data.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)

    # 2. 预处理
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 3. 循环 RANSAC 提取 6 个平面
    print("-> 开始循环提取 6 个平面...")
    rest = pcd
    planes_list = []
    plane_colors = plt.get_cmap("tab10")(np.linspace(0, 1, 6))

    for i in range(6):
        plane_model, inliers = rest.segment_plane(distance_threshold=0.03,
                                                  ransac_n=3,
                                                  num_iterations=1000)
        plane_pcd = rest.select_by_index(inliers)
        plane_pcd.paint_uniform_color(plane_colors[i][:3])
        planes_list.append(plane_pcd)
        rest = rest.select_by_index(inliers, invert=True)

    # --- 关键步骤：估计法向量并计算外积矩阵 ---
    print("-> 正在估计法向量并构建外积特征...")
    # 适当加大半径可以让法线更平滑，减少由于噪声引起的微小方向跳变
    rest.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50))

    xyz = np.asarray(rest.points)
    normals = np.asarray(rest.normals)

    # 计算外积矩阵的 6 个独立分量: n*n^T 是对称矩阵
    nx = normals[:, 0]
    ny = normals[:, 1]
    nz = normals[:, 2]

    # 这 6 个量对于 n 和 -n 是完全一致的
    nxx = nx * nx
    nyy = ny * ny
    nzz = nz * nz
    nxy = nx * ny
    nxz = nx * nz
    nyz = ny * nz

    # 4. 构建 9D 特征空间
    # normal_weight 建议设在 0.1 左右。如果物体依然连在一起，调大它；如果桌面碎裂，调小它。
    normal_weight = 0.1

    # 拼接坐标(3D) + 外积特征(6D) = 9D 特征向量
    # 这里我们给 6 个法向分量都统一乘以权重
    tensor_features = np.column_stack((nxx, nyy, nzz, nxy, nxz, nyz)) * normal_weight
    features = np.concatenate((xyz, tensor_features), axis=1)

    # 5. 运行 Sklearn DBSCAN
    print("-> 正在运行 9D 空间 DBSCAN (外积矩阵法)...")
    # 因为维度增加了，eps 可能需要稍微调大一点点，比如 0.12 -> 0.15
    db = DBSCAN(eps=0.15, min_samples=10, n_jobs=-1).fit(features)
    labels = db.labels_

    max_label = labels.max()
    print(f"检测到 {max_label + 1} 个聚类物体")

    # 为聚类物体染色
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        if labels[i] >= 0:
            colors[i] = cmap(labels[i] / (max_label if max_label > 0 else 1))[:3]
        else:
            colors[i] = [0, 0, 0]  # 噪声设为黑色
    rest.colors = o3d.utility.Vector3dVector(colors)

    # 6. 可视化准备
    offset = 8.0
    pcd_raw = copy.deepcopy(pcd)
    pcd_all_planes = o3d.geometry.PointCloud()
    for p in planes_list:
        pcd_all_planes += p
    pcd_all_planes.translate((offset, 0, 0))
    pcd_objects = copy.deepcopy(rest)
    pcd_objects.translate((offset * 2, 0, 0))

    print("-> 正在显示结果...")
    # 第一个窗口显示带法线的剩余点（检查法线是否平滑）
    o3d.visualization.draw_geometries([rest], point_show_normal=True, window_name="检查法线")

    # 第二个窗口显示最终对比
    o3d.visualization.draw_geometries([pcd_raw, pcd_all_planes, pcd_objects],
                                      window_name="外积矩阵 DBSCAN 聚类结果",
                                      width=1600, height=900)


# 执行
pipeline_with_structure_tensor("Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt")