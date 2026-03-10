import time
import numpy as np
from csm import CSM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from utils import calculate_angular_velocity, axis_angle_from_vectors, normalize_vector, load_workspace_data, get_random_target

# 统计步数和失败目标的列表
step_count = 0
target_cnt = 1
max_steps = 8000  # 设置达到目标的最大步数
delta_t = 0.001   # 时间间隔，单位秒
failures = []
successes = []
v_lim = 0.2   # 线速度限制
w_lim = 2     # 角速度限制
draw_interval = 10
max_targets = 10
last_time = time.time()
frame_times = []
finished = False
def animate(i, csm, ax):
    global finished, step_count, target_cnt, max_steps, last_time, frame_times
    if finished:   
        return
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
        mode, new_target_pose, _ = get_random_target(workspace_data)  # 从数据中随机选择一个新的目标
        csm.target_pose = new_target_pose

        step_count = 0  # 重置步数统计
        target_cnt += 1
        print("mode:", mode, "target:", new_target_pose)
    elif step_count > max_steps:
        print("Failed to reach target after", step_count, "steps")

        mode, new_target_pose, _ = get_random_target(workspace_data)  # 从数据中随机选择一个新的目标
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
        finished = True 
        ani.event_source.stop()
        return


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, delta_t)
    # start_cfg = (4, 2.74, 0.04, 0.06, 0.02, 0.055, 1.13, 0.9, 0.53, 1.83)
    # csm.set_state(*start_cfg)
    workspace_data = load_workspace_data("./data/workspace_data.json")
    mode, pose, _ = get_random_target(workspace_data)
    csm.target_pose = pose
    print("mode:", mode, "target:", pose)

    video_dir = Path("./vedios")
    video_dir.mkdir(parents=True, exist_ok=True) 
    try:
        ani = FuncAnimation(fig, animate, fargs=(csm, ax), frames=10000, interval=20, repeat=False)
        import matplotlib
        matplotlib.rcParams['animation.ffmpeg_path'] = r"D:\TOOLS\ffmpeg-2025-10-27-git-68152978b5-essentials_build\bin\ffmpeg.exe"
        ani.save("./vedios/display.mp4", fps=24, dpi=150, writer="ffmpeg")
        # plt.show()
        print("Finished")
    except KeyboardInterrupt:
        ani.event_source.stop()
        plt.close(fig)
        print("Interrupted")

