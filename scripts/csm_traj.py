import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from csm import CSM
from csm_display import axis_angle_from_vectors, normalize_vector

# ======================
# 参数定义
# ======================
delta_t = 0.001
v_lim = 0.3
w_lim = 3

# 初始化模型
csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi / 2, 2 * np.pi / 3, delta_t)

# ======================
# 1️⃣ 设置起点与终点配置
# ======================
# 起点配置
start_cfg = (4, 2.74, 0.04, 0.06, 0.02, 0.055, 1.13, 0.9, 0.53, 1.83)
end_cfg   = (1, -0.95, 0, 0.026, 0, 0, 0, 0.5, 0, 0.03)
# start_cfg = (1, 1.34, 0, 0.036, 0, 0, 0, 0.95, 0, 1.43)
# end_cfg   = (4, 1.77, 0.04, 0.06, 0.02, 0.127, 1.09, 0.99, -0.8, 2.2)


csm.set_state(*end_cfg)
target_pose = csm.pose.copy()  # 保存目标位姿
print("Target pose:", target_pose)
csm.reset()
csm.set_state(*start_cfg)  # 回到起点
csm.target_pose = target_pose.copy()
# ======================
# 2️⃣ 动画与数据记录变量
# ======================
position_errors = []
orientation_errors = []
theta1_vals, theta2_vals = [], []
L1_vals, L2_vals = [], []

max_iter = 3000
tolerance = 1e-4

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.05, top=0.95)

# ======================
# 3️⃣ 误差计算函数
# ======================
def compute_errors(csm, target_pose):
    # 位置误差 / mm
    pos_err = np.linalg.norm(csm.pose[:3] - target_pose[:3]) * 1000 

    # 姿态误差 / rad
    a = csm.pose[3:] / np.linalg.norm(csm.pose[3:])
    b = target_pose[3:] / np.linalg.norm(target_pose[3:])
    cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
    ang_err = np.arccos(cos_theta)

    return pos_err, ang_err

# ======================
# 4️⃣ 动画帧更新函数
# ======================
def animate(i):
    global csm
    csm.check_transition()
    csm.update()
    csm.update_jacobians()

    v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * v_lim
    axis_hat, _ = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
    w = w_lim * axis_hat

    # 更新雅可比与状态
    csm.get_dot_PHI(v, w)
    csm.step()
    csm.plot_manipulator(ax)

    # 计算误差
    pos_err, ang_err = compute_errors(csm, target_pose)
    position_errors.append(pos_err)
    orientation_errors.append(ang_err)

    theta1_vals.append(csm.theta_1)
    theta2_vals.append(csm.theta_2)
    L1_vals.append(csm.L1)
    L2_vals.append(csm.L2)

    # 绘制机械臂
    ax.clear()
    csm.plot_manipulator(ax)
    ax.set_title(f"Step {i} | PosErr={pos_err:.2f} mm | AngErr={ang_err:.3f} rad")

    # 判断收敛
    if pos_err < 1.0 and ang_err < 0.02:
        print(f"Converged at step {i}")
        ani.event_source.stop()
        plot_results()
    elif i >= max_iter - 1:
        print("Max iteration reached")
        ani.event_source.stop()
        plot_results()

# ======================
# 5️⃣ 绘制误差和变量曲线
# ======================
def plot_results():
    fig2, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    t = np.arange(len(position_errors)) * delta_t

    # 左轴：位置误差(mm)
    ax1.plot(t, position_errors, 'b-', label='Position Error (mm)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position Error (mm)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 右轴：角度误差(rad)
    ax2.plot(t, orientation_errors, 'r--', label='Orientation Error (rad)')
    ax2.set_ylabel('Orientation Error (rad)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Tracking Errors Over Time')
    plt.tight_layout()
    plt.show()

    # 第二张图：变量变化
    fig3, ax3 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax3[0].plot(t, theta1_vals, label='θ1')
    ax3[0].plot(t, theta2_vals, label='θ2')
    ax3[0].legend()
    ax3[0].set_ylabel('Angle (rad)')

    ax3[1].plot(t, L1_vals, label='L1')
    ax3[1].plot(t, L2_vals, label='L2')
    ax3[1].legend()
    ax3[1].set_xlabel('Time (s)')
    ax3[1].set_ylabel('Length (m)')
    plt.tight_layout()
    plt.show()

# ======================
# 6️⃣ 启动动画
# ======================
ani = FuncAnimation(fig, animate, frames=max_iter, interval=20)
print("Rendering video...")
ani.save("./vedios/c4_c1.mp4", fps=24, dpi=150, writer="ffmpeg")
print("Saved video.")
plt.show()
