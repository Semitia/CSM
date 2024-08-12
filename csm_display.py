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
    if norm < 0.5:
        return v
    return v / norm

def load_workspace_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# 统计步数和失败目标的列表
step_count = 0
target_cnt = 1
max_steps = 500  # 设置达到目标的最大步数
failures = []
successes = []
def animate(i, csm, ax):
    global step_count, target_cnt, max_steps, failures, successes
    csm.check_transition()
    csm.update()
    # csm.debug()
    csm.update_jacobians()

    v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * 4
    w = calculate_angular_velocity(csm.pose[3:], csm.target_pose[3:], 0.5)
    # print("v:", v, "w:", w)
    csm.get_dot_PHI(v, w)
    csm.step()
    csm.plot_manipulator(ax)

    step_count += 1
    if np.linalg.norm(csm.pose - csm.target_pose) < 0.08:
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
    csm = CSM(0.5, 0.5, 0.15, 0.15, 0.01)
    workspace_data = load_workspace_data("workspace_data.json")
    mode, pose = get_random_target(workspace_data)
    csm.target_pose = pose
    print("space:", mode, "target:", pose)

    try:
        # 那我能不能单开一个线程来画动画？
        ani = FuncAnimation(fig, animate, fargs=(csm, ax), frames=100, interval=17)
        # ani._args = (csm, ax, ani)  # 在调用后更新 fargs 以传入 ani 自身
        plt.show()
        print("Finished")
    except KeyboardInterrupt:
        ani.event_source.stop()
        # 关闭图形窗口
        plt.close(fig)
        print("Interrupted")
        # 保存成功和失败数据
        if successes:
            with open("successes_play.json", "w") as f:
                json.dump(successes, f, indent=4)
            print(f"Recorded {len(successes)} successes to successes.json")
        
        if failures:
            with open("failures_play.json", "w") as f:
                json.dump(failures, f, indent=4)
            print(f"Recorded {len(failures)} failures to failures.json")
