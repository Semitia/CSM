import numpy as np

def calculate_angular_velocity(v1, v2, delta_t):
    """
    计算从方向向量 v1 到 v2 的角速度。

    参数:
    v1 : array_like
        初始方向向量。
    v2 : array_like
        最终方向向量。
    delta_t : float
        时间差（秒）。

    返回:
    omega : ndarray
        角速度向量 (ωx, ωy, ωz)。
    """
    # 确保v1和v2是单位向量
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 计算旋转轴 (叉积)
    n = np.cross(v1, v2)

    # 计算旋转角 (点积)
    cos_theta = np.dot(v1, v2)
    # 限制cos_theta的范围在[-1, 1]内，以防止计算误差导致的问题
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # 计算角速度的大小
    if delta_t == 0:
        raise ValueError("delta_t cannot be zero, as it would result in a division by zero.")
    omega_magnitude = theta / delta_t

    # 计算角速度向量
    if np.linalg.norm(n) == 0:
        # 如果n是零向量，则方向不变，角速度为0
        return np.array([0, 0, 0])
    else:
        # 单位化旋转轴向量
        n_unit = n / np.linalg.norm(n)
        # 角速度向量
        omega = omega_magnitude * n_unit

    return omega

# 示例使用
v1 = np.array([0, 0, 1])  # 初始方向向量，指向z轴
v2 = np.array([1, 0, 0])  # 最终方向向量，指向x轴
delta_t = 1.0  # 时间差，例如1秒

# omega = calculate_angular_velocity(v1, v2, delta_t)
# print("Angular velocity vector (rad/s):", omega)

x_0 = np.array([0.631,0.549,0.549])
x_1 = np.array([0.64,0.56,0.526])
omega = calculate_angular_velocity(x_0, x_1, delta_t)
print("Angular velocity vector (rad/s):", omega)
