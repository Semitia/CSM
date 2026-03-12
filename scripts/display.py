"""
Module: display.py
Description: Script to visualize and animate the CSM model reaching targets.
"""
import time
import yaml
import numpy as np
from csm.model import CSM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from csm.utils import calculate_angular_velocity, axis_angle_from_vectors, normalize_vector, load_workspace_data, get_random_target

step_count = 0
target_cnt = 1
max_steps = 8000
delta_t = 0.001
failures = []
successes = []
v_lim = 0.2
w_lim = 2
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
    axis_hat, _ = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
    w = w_lim * axis_hat

    csm.get_dot_PHI(v, w)
    csm.step()
    csm.plot_manipulator(ax)

    step_count += 1
    if np.linalg.norm(csm.pose - csm.target_pose) < 1e-3:
        print("Reached target after", step_count, "steps")
        mode, new_target_pose, _ = get_random_target(workspace_data)
        csm.target_pose = new_target_pose
        step_count = 0
        target_cnt += 1
        print("mode:", mode, "target:", new_target_pose)
    elif step_count > max_steps:
        print("Failed to reach target after", step_count, "steps")
        mode, new_target_pose, _ = get_random_target(workspace_data)
        csm.target_pose = new_target_pose
        step_count = 0
        target_cnt += 1
        print("mode:", mode, "target:", new_target_pose)

    if target_cnt > max_targets:
        print("已完成全部目标，停止动画。")
        finished = True
        ani.event_source.stop()
        return


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    config_path = Path("./config/csm_config_3.4mm.yaml")
    
    # Load control parameters from config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    if "control" in config_data:
        ctrl = config_data["control"]
        if "v_lim" in ctrl: v_lim = ctrl["v_lim"]
        if "w_lim" in ctrl: w_lim = ctrl["w_lim"]
        if "max_steps" in ctrl: max_steps = ctrl["max_steps"]
        
    csm = CSM.from_config(config_path)
    
    workspace_data = load_workspace_data("./data/workspace_data_3.4mm.json")
    
    mode, pose, _ = get_random_target(workspace_data)
    csm.target_pose = pose
    print("mode:", mode, "target:", pose)

    video_dir = Path("./videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    try:
        ani = FuncAnimation(fig, animate, fargs=(csm, ax), frames=10000, interval=20, repeat=False)
        plt.show()
        print("Finished")
    except KeyboardInterrupt:
        ani.event_source.stop()
        plt.close(fig)
        print("Interrupted")
