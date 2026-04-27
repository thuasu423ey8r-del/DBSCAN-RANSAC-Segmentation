import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import DBSCAN  # 导入 sklearn 的 DBSCAN


def pipeline_with_normal_weight(pcd_path):
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

    # --- 新增步骤：估计法向量 ---
    print("-> 正在估计剩余物体的法向量...")
    rest.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 4. 使用法向量权重的自定义 DBSCAN (使用 Sklearn)
    print("-> 正在运行带法向量权重的 DBSCAN (Sklearn)...")

    # 提取坐标和法向量
    xyz = np.asarray(rest.points)
    normals = np.asarray(rest.normals)

    # 【法向量权重设置】
    # normal_weight 越大，方向不同的物体越容易被切分。通常设在 0.05-0.2 之间测试
    normal_weight = 0.1

    # 构建 6D 特征向量: [x, y, z, nx*w, ny*w, nz*w]
    features = np.concatenate((xyz, normals * normal_weight), axis=1)

    # 使用 Sklearn 的 DBSCAN
    # 注意：因为增加了维度，eps 可能需要稍微调大一点点
    db = DBSCAN(eps=0.12, min_samples=10, n_jobs=-1).fit(features)
    labels = db.labels_

    max_label = labels.max()
    print(f"检测到 {max_label + 1} 个聚类物体")

    # 为聚类物体染色
    cmap = plt.get_cmap("tab20")
    # 生成颜色数组：噪声点（-1）会分配到黑色
    colors = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        if labels[i] >= 0:
            colors[i] = cmap(labels[i] / (max_label if max_label > 0 else 1))[:3]
        else:
            colors[i] = [0, 0, 0]  # 噪声为黑色

    rest.colors = o3d.utility.Vector3dVector(colors)

    # 5. 可视化准备
    offset = 8.0
    pcd_raw = copy.deepcopy(pcd)
    pcd_all_planes = o3d.geometry.PointCloud()
    for p in planes_list:
        pcd_all_planes += p
    pcd_all_planes.translate((offset, 0, 0))
    pcd_objects = copy.deepcopy(rest)
    pcd_objects.translate((offset * 2, 0, 0))
    o3d.visualization.draw_geometries([rest], point_show_normal=True)
    o3d.visualization.draw_geometries([pcd_raw, pcd_all_planes, pcd_objects],
                                      window_name="法向量权重 DBSCAN 聚类结果",
                                      width=1600, height=900)


# 运行测试
pipeline_with_normal_weight("Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt")