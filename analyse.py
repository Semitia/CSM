import json
import threading
import numpy as np
from csm import CSM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# filename = "successes_play.json"
# with open(filename, 'r') as f:
#     data = json.load(f)

# steps = [success["steps_taken"] for success in data]
# # 绘制直方图
# plt.figure(figsize=(10, 6))
# plt.hist(steps, bins=50, color='blue', edgecolor='black', alpha=0.7)
# plt.title('Distribution of Steps Taken for Successes')
# plt.xlabel('Number of Steps')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# 分析失败数据
filename = "failures_play.json"
with open(filename, 'r') as f:
    data = json.load(f)
print("failure_len:", len(data))

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 0.5:
        return v
    return v / norm

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

start_play = False
def animate(csm, ax):
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

csm = CSM(0.5, 0.5, 0.15, 0.15, 0.01)
stop_animation = threading.Event()  # 用于控制动画停止
def animate_thread():
    global csm
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    def update_animation(frame):
        if stop_animation.is_set():
            plt.close(fig)  # 关闭当前窗口
            return
        animate(csm, ax)
    ani = FuncAnimation(fig, update_animation, frames=100, interval=17)
    plt.show()
    print("End animation")
animation_thread = threading.Thread(target=animate_thread)

try:
    for i, failure in enumerate(data):
        print("Failure", i)
        stop_animation.set() # 停止上一个动画
        plt.close('all')
        Ls = failure["Ls"]
        Lr = failure["Lr"]
        L1 = failure["L1"]
        L2 = failure["L2"]
        phi = failure["phi"]
        mode = failure["mode"]
        theta_1 = failure["theta_1"]
        delta_1 = failure["delta_1"]
        theta_2 = failure["theta_2"]
        delta_2 = failure["delta_2"]
        csm.set_state(mode, phi, L1, L2, Lr, Ls, theta_1, theta_2, delta_1, delta_2)
        csm.target_pose = failure["target_pose"]
        animation_thread.start()
        user_input = input("Press 'q' to quit, 'n' check next failure: ")
        if user_input == "q":
            break
        elif user_input == "n":
            continue

    print("End")

except KeyboardInterrupt:
    plt.close('all') 
    print("Interrupt")

