"""
Module: utils.py
Description: Utility functions for vector operations, angular velocity calculation, and data loading.
"""
import json
import random
import numpy as np

def skew_symmetric_matrix(p):
    """
    计算向量p的反对称矩阵
    参数:
        p: 输入向量 (3,)
    返回:
        反对称矩阵 (3,3)
    """
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])


def damped_pseudo_inverse(J, damping_factor=0.01):
    """
    计算阻尼最小二乘法的伪逆

    参数:
    J: 输入矩阵
    damping_factor: 阻尼因子

    返回:
    J_damped_pinv: 阻尼最小二乘法伪逆
    """
    m, n = J.shape
    if m >= n:
        return np.linalg.inv(J.T @ J + (damping_factor**2) * np.eye(n)) @ J.T
    else:
        return J.T @ np.linalg.inv(J @ J.T + (damping_factor**2) * np.eye(m))

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
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    n = np.cross(v1, v2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if delta_t == 0:
        raise ValueError("delta_t cannot be zero")
    if np.linalg.norm(n) == 0:
        return np.array([0, 0, 0])
    return (theta / delta_t) * (n / np.linalg.norm(n))


def axis_angle_from_vectors(v1, v2, eps=1e-8):
    """
    输入：
        方向向量 v1, v2,
    返回：
        - axis_hat: 归一化转轴(从 v1 右手旋到 v2 的方向)
        - theta:    旋转角(弧度，范围 [0, pi])

    采用 arctan2(||v1 X v2||, v1·v2) 计算角度，更稳定。
    对于 v1≈-v2(180°0时, 叉积接近0, 选择一条与 v1 正交的任意轴。
    """
    n1 = np.asarray(v1, dtype=float)
    n2 = np.asarray(v2, dtype=float)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    cross = np.cross(n1, n2)
    s = np.linalg.norm(cross)
    theta = np.arctan2(s, np.clip(np.dot(n1, n2), -1.0, 1.0))
    axis_hat = cross if s < eps else cross / s
    return axis_hat, theta


def normalize_vector(v, threshold=1e-6):
    norm = np.linalg.norm(v)
    return v if norm < threshold else v / norm


def load_workspace_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_random_target(data):
    target = random.choice(data)
    return target["mode"], np.array(target["pose"]), target.get("config", None)
