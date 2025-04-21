import numpy as np

# 禁用科学计数法
np.set_printoptions(suppress=True)

def compute_transformation_matrix(points_cam, points_base):

    assert points_cam.shape == points_base.shape, "点集必须具有相同的形状。"

    # 计算两个点集的质心
    centroid_cam = np.mean(points_cam, axis=0)
    centroid_base = np.mean(points_base, axis=0)

    # 使点集居中
    centered_cam = points_cam - centroid_cam
    centered_base = points_base - centroid_base

    # 计算互协方差矩阵
    H = np.dot(centered_base.T, centered_cam)

    # 进行奇异值分解 (SVD)
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    # 计算平移向量
    t = centroid_base - np.dot(R, centroid_cam)

    # 构造 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

# 示例使用
if __name__ == "__main__":
    # 定义相机坐标系和基座坐标系中的 6 个对应点
    points_cam = np.array([
        [-15, -2, 263],
        [9, -3, 264],
        [33, -3, 265],
        [-21, -15, 274],
        [3, -16, 275],
        [27, -16, 276],
        [-17, -29, 286],
        [7, -30, 287],
        [30, -35, 292]
    ])



    #机械臂base下的在确定的特征点的坐标


    points_base = np.array([
        [257, -11, 110],
        [255, 8, 80],
        [255, 27, 80],
        [244, -17, 83],
        [244, 3, 83],
        [243, 24, 82],
        [236, -12, 87],
        [236, 9, 87],
        [232, 29, 88]
    ])

    # 计算变换矩阵
    T = compute_transformation_matrix(points_cam, points_base)

    print("相机到基座的变换矩阵:")
    print(T)

    # 通过变换 points_cam 并与 points_base 进行比较来验证结果
    transformed_points = (T[:3, :3] @ points_cam.T).T + T[:3, 3]
    print("\n变换后的点 (相机到基座):")
    print(transformed_points)

    print("\n基座中的原始点:")
    print(points_base)

    # 计算误差
    error = np.linalg.norm(transformed_points - points_base, axis=1)
    print("\n误差:")
    print(error)
