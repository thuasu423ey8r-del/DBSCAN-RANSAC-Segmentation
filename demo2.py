import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

def pipeline_xyz_only(pcd_path):
    # 1. 加载 S3DIS .txt 点云
    print("-> 正在读取 S3DIS 原始数据...")
    data = np.loadtxt(pcd_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    if data.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)

    # 2. 预处理 (下采样对所有聚类都很重要)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 3. RANSAC 去平面 (必须去掉地面，否则所有物体都会因地面连通)
    print("-> RANSAC 去除地面...")
    _, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)
    pcd_objects = pcd.select_by_index(inliers, invert=True)
    pcd_ground = pcd.select_by_index(inliers)
    pcd_ground.paint_uniform_color([0.5, 0.5, 0.5]) # 地面设为灰色

    # 4. Open3D 原生 DBSCAN 聚类 (仅考虑 XYZ 空间距离)
    print("-> 正在运行 Open3D 原生 DBSCAN (仅考虑 XYZ)...")
    # eps: 两个点被认为是邻居的最大空间距离
    # min_points: 形成簇所需的最小点数
    labels = np.array(pcd_objects.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))

    # 5. 可视化
    max_label = labels.max()
    print(f"检测到 {max_label + 1} 个聚类物体")

    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 噪声为黑色
    pcd_objects.colors = o3d.utility.Vector3dVector(colors[:, :3])

    print("-> 正在显示结果...")
    # 为了不破坏原始坐标，我们创建副本进行平移
    pcd_raw = copy.deepcopy(pcd)
    pcd_g = copy.deepcopy(pcd_ground)
    pcd_obj = copy.deepcopy(pcd_objects)

    # 计算点云的宽度，用于确定平移距离
    # 或者直接根据 S3DIS 场景大小手动设定一个位移，比如 10 米
    offset = 4.0

    # 1. 原始点云不动 (留在原点)
    # 2. 地面往右移一个单位
    pcd_g.translate((offset, 0, 0))
    # 3. 聚类物体往右移两个单位
    pcd_obj.translate((offset * 2, 0, 0))

    # 一起显示
    print(f"正在显示并排对比：左(原始) | 中(地面) | 右(聚类物体)")
    o3d.visualization.draw_geometries([pcd_raw, pcd_g, pcd_obj],
                                      window_name="对比：原始 vs 地面 vs 聚类结果",
                                      width=1600, height=900)

# 运行 (路径换成你的测试文件)
pipeline_xyz_only("Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt")