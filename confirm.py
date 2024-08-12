import numpy as np
import sympy as sp

# 定义符号变量
theta_t, delta_t, L_t = sp.symbols('theta_t delta_t L_t')
theta_dot, delta_dot, L_dot = sp.symbols('theta_dot delta_dot L_dot')

# 定义旋转矩阵 R_b_1, R_1_2, R_2_e
R_b_1 = sp.Matrix([
    [0, sp.cos(delta_t), sp.sin(delta_t)],
    [0, -sp.sin(delta_t), sp.cos(delta_t)],
    [1, 0, 0]
])

R_1_2 = sp.Matrix([
    [sp.cos(theta_t), -sp.sin(theta_t), 0],
    [sp.sin(theta_t), sp.cos(theta_t), 0],
    [0, 0, 1]
])

R_2_e = sp.Matrix([
    [0, 0, 1],
    [sp.cos(delta_t), -sp.sin(delta_t), 0],
    [sp.sin(delta_t), sp.cos(delta_t), 0]
])

# 计算旋转矩阵 R_te
R_te = R_b_1 * R_1_2 * R_2_e

# 计算 R_te 的时间导数
R_te_dot = R_te.diff(theta_t) * theta_dot + R_te.diff(delta_t) * delta_dot + R_te.diff(L_t) * L_dot

# 计算角速度反对称矩阵 omega_hat
omega_hat = R_te.T * R_te_dot

# 提取角速度向量 omega
omega = sp.Matrix([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

# 计算角速度雅可比矩阵 J_w
J_w = omega.jacobian([theta_dot, delta_dot, L_dot])


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

def get_trans_mat(theta_t, delta_t,  L_t):
    R_b_1 = np.array([  [0, np.cos(delta_t), np.sin(delta_t)],
                        [0, -np.sin(delta_t), np.cos(delta_t)],
                        [1, 0, 0]])
    R_1_2 = np.array(  [[np.cos(theta_t), -np.sin(theta_t), 0],
                        [np.sin(theta_t), np.cos(theta_t), 0],
                        [0, 0, 1]])
    R_2_e = np.array([  [0, 0, 1],
                        [np.cos(delta_t), -np.sin(delta_t), 0],
                        [np.sin(delta_t), np.cos(delta_t), 0]])
    R_te = R_b_1 @ R_1_2 @ R_2_e
    if theta_t == 0:
        P_te = np.array([0, 0, L_t])
    else:
        P_te = np.array([L_t * np.cos(delta_t) * (1 - np.cos(theta_t)) / theta_t,
                        L_t * np.sin(delta_t) * (np.cos(theta_t) - 1) / theta_t,
                        L_t * np.sin(theta_t) / theta_t])
    T = np.eye(4)
    T[:3, :3] = R_te
    T[:3, 3] = P_te
    return T

theta = np.pi/4
delta = np.pi/3
L = 0.3
values = {
    theta_t: theta,
    delta_t: delta,
    L_t: L
}
J_w_numeric = J_w.subs(values)
J_w_numeric = np.array([    [np.sin(delta),  np.cos(delta)*np.sin(theta), 0],
                            [np.cos(delta), -np.sin(delta)*np.sin(theta), 0],
                            [0, np.cos(theta)-1, 0]])

init_ori = np.array([[0],[0],[1]])
T = get_trans_mat(theta, delta, L)
end_ori = T[:3, :3] @ init_ori
d = [-0.1, 0.1, 0.1]
T = get_trans_mat(theta + d[0], delta + d[1], L + d[2])
new_end_ori = T[:3, :3] @ init_ori
omega = calculate_angular_velocity(end_ori.reshape(-1), new_end_ori.reshape(-1), 1)
expect_omega = J_w_numeric @ np.array(d).reshape(-1)
print("expect_w: ", expect_omega)
print("w       : ", omega)
