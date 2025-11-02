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
    n1 = v1 / (np.linalg.norm(v1) + eps)
    n2 = v2 / (np.linalg.norm(v2) + eps)

    cross = np.cross(n1, n2)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    s = np.linalg.norm(cross)

    # 角度（更数值稳定的写法）
    theta = np.arctan2(s, dot)

    if s < eps:
        # 方向几乎一致
        if theta < 1e-6:
            return np.zeros(3), 0.0
        # 方向几乎相反（~pi），叉积数值上也可能很小：选一条与 n1 正交的“任意”轴
        # 这里通过与 x 或 y 轴叉乘构造一个正交向量
        helper = np.array([1.0, 0.0, 0.0]) if abs(n1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(n1, helper)
        axis /= (np.linalg.norm(axis) + eps)
        return axis, np.pi

    axis_hat = cross / s
    return axis_hat, theta


# 统计步数和失败目标的列表
step_count = 0
target_cnt = 1
max_steps = 5000  # 设置达到目标的最大步数
delta_t = 0.001   # 时间间隔，单位秒
failures = []
successes = []
v_lim = 1e-3   # 线速度限制
w_lim = 5e-4  # 角速度限制

def animate(i, csm, ax):
    global step_count, target_cnt, max_steps, failures, successes
    csm.check_transition()
    csm.update()
    csm.update_jacobians()

    v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * v_lim
    # w = calculate_angular_velocity(csm.pose[3:], csm.target_pose[3:], 0.5)
    axis_hat, theta = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
    w = w_lim * axis_hat

    # print("v:", v, "w:", w)
    csm.get_dot_PHI(v, w)
    csm.step()
    csm.plot_manipulator(ax)

    step_count += 1
    if np.linalg.norm(csm.pose - csm.target_pose) < 2e-6:
        print("Reached target")
        mode, new_target_pose = get_random_target(workspace_data)  # 从数据中随机选择一个新的目标
        csm.target_pose = new_target_pose
        # 记录成功目标和当前的各个参数
        success_data = {
            "steps_taken": step_count
        }
        successes.append(success_data)
        step_count = 0  # 重置步数统计
        target_cnt += 1
        print("space:", mode, "target:", new_target_pose)
    elif step_count > max_steps:
        print("Failed to reach target after", step_count, "steps")
        # 记录失败目标和当前的各个参数
        failure_data = {
            "target_pose": csm.target_pose.tolist(),
            "mode": csm.mode,
            "phi": csm.phi,
            "theta_1": csm.theta_1,
            "theta_2": csm.theta_2,
            "delta_1": csm.delta_1,
            "delta_2": csm.delta_2,
            "Ls": csm.Ls,
            "Lr": csm.Lr,
            "steps_taken": step_count
        }
        failures.append(failure_data)
        mode, new_target_pose = get_random_target(workspace_data)  # 从数据中随机选择一个新的目标
        csm.target_pose = new_target_pose
        step_count = 0  # 重置步数统计
        target_cnt += 1
        print("space:", mode, "target:", new_target_pose)


if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, delta_t)
    workspace_data = load_workspace_data("./data/workspace_data.json")
    mode, pose = get_random_target(workspace_data)
    csm.target_pose = [0.3, 0.5, 0.8, 0, 1, 0]
    print("space:", mode, "target:", pose)

    try:
        ani = FuncAnimation(fig, animate, fargs=(csm, ax), frames=100, interval=17)
        plt.show()
        print("Finished")
    except KeyboardInterrupt:
        ani.event_source.stop()
        # 关闭图形窗口
        plt.close(fig)
        print("Interrupted")
        # 保存成功和失败数据
        # if successes:
        #     with open("successes_play.json", "w") as f:
        #         json.dump(successes, f, indent=4)
        #     print(f"Recorded {len(successes)} successes to successes.json")
        
        # if failures:
        #     with open("failures_play.json", "w") as f:
        #         json.dump(failures, f, indent=4)
        #     print(f"Recorded {len(failures)} failures to failures.json")
