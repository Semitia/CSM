import time
import json
import random
import numpy as np
from csm import CSM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def get_random_target(data):
    target = random.choice(data)
    mode = target["mode"]
    pose = np.array(target["pose"])
    return mode, pose

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm

def load_workspace_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

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
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = v1 / (np.linalg.norm(v1) )
    n2 = v2 / (np.linalg.norm(v2) )

    cross = np.cross(n1, n2)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    s = np.linalg.norm(cross)

    theta = np.arctan2(s, dot)

    # if s < eps:
    #     # 方向几乎一致
    #     if theta < 1e-6:
    #         return np.zeros(3), 0.0
    #     # 方向几乎相反（~pi），叉积数值上也可能很小：选一条与 n1 正交的“任意”轴
    #     # 这里通过与 x 或 y 轴叉乘构造一个正交向量
    #     helper = np.array([1.0, 0.0, 0.0]) if abs(n1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    #     axis = np.cross(n1, helper)
    #     axis /= (np.linalg.norm(axis) + eps)
    #     return axis, np.pi

    if s< eps:
        axis_hat = cross  
    else:
        axis_hat = cross / s
    return axis_hat, theta


# 统计步数和失败目标的列表
step_count = 0
target_cnt = 1
max_steps = 5000  # 设置达到目标的最大步数
delta_t = 0.001   # 时间间隔，单位秒
failures = []
successes = []
v_lim = 0.2   # 线速度限制
w_lim = 2     # 角速度限制
draw_interval = 10
max_targets = 10
last_time = time.time()
frame_times = []
def animate(i, csm, ax):
    global step_count, target_cnt, max_steps, last_time, frame_times
    csm.check_transition()
    csm.update()
    csm.update_jacobians()

    v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * v_lim
    # w = calculate_angular_velocity(csm.pose[3:], csm.target_pose[3:], 0.1)
    axis_hat, _ = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
    w = w_lim * axis_hat

    # print("v:", v, "w:", w)
    csm.get_dot_PHI(v, w)
    csm.step()
    csm.plot_manipulator(ax)
    # # --- 控制绘图频率 ---
    # if i % draw_interval == 0:
    #     ax.cla()
    #     csm.plot_manipulator(ax)
    # # ----------------------

    step_count += 1
    if np.linalg.norm(csm.pose - csm.target_pose) < 1e-3:
        print("Reached target after", step_count, "steps")
        mode, new_target_pose = get_random_target(workspace_data)  # 从数据中随机选择一个新的目标
        csm.target_pose = new_target_pose

        step_count = 0  # 重置步数统计
        target_cnt += 1
        print("mode:", mode, "target:", new_target_pose)
    elif step_count > max_steps:
        print("Failed to reach target after", step_count, "steps")

        mode, new_target_pose = get_random_target(workspace_data)  # 从数据中随机选择一个新的目标
        csm.target_pose = new_target_pose
        step_count = 0  # 重置步数统计
        target_cnt += 1
        print("mode:", mode, "target:", new_target_pose)
    
    # # FPS计算
    # now = time.time()
    # frame_times.append(now - last_time)
    # last_time = now
    # if len(frame_times) >= 30:  # 每30帧更新一次
    #     avg_fps = 1 / (sum(frame_times) / len(frame_times))
    #     print(f"平均FPS: {avg_fps:.2f}")
    #     frame_times.clear()

    if target_cnt > max_targets:
        print("已完成全部目标，停止动画。")
        ani.event_source.stop()
        return


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, delta_t)
    workspace_data = load_workspace_data("./data/workspace_data.json")
    mode, pose = get_random_target(workspace_data)
    csm.target_pose = pose
    print("mode:", mode, "target:", pose)

    try:
        ani = FuncAnimation(fig, animate, fargs=(csm, ax), repeat=False)
        
        plt.show()
        print("Finished")
    except KeyboardInterrupt:
        ani.event_source.stop()
        plt.close(fig)
        print("Interrupted")

