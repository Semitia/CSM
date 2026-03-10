import json
import numpy as np
from csm import CSM
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def generate_workspace(csm, mode, num_samples=2000):
    csm.state_transition(csm.mode, mode)
    workspace_points = []

    # DEBUG
    # 在 特定 mode 追踪“离 z 轴最远”的点与配置
    target_mode = 4
    max_r = -np.inf
    best_point = None
    best_cfg = None
    # END DEBUG

    for _ in range(num_samples):
        # 随机生成姿态参数
        csm.phi = np.random.uniform(0, 2 * np.pi)
        if mode == 1:
            csm.L2 = np.random.uniform(0, csm.L_20)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 2:
            csm.Lr = np.random.uniform(0, csm.L_r0)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 3:
            csm.L1 = np.random.uniform(0, csm.L_10)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 4:
            csm.Ls = np.random.uniform(0, csm.L_s0)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        csm.update()
        p = csm.pose[:3].copy()
        workspace_points.append(p)

    # DEBUG
        if mode == target_mode:
            r = float(np.hypot(p[0], p[1]))  # 等价于 sqrt(x^2 + y^2)
            if r > max_r:
                max_r = r
                best_point = p.copy()
                # 保存当前配置（按你在 mode 3 中会用到的量为主，其他也一并记录以便排查）
                best_cfg = {
                    "phi": float(csm.phi),
                    "L1": float(getattr(csm, "L1", np.nan)),
                    "L2": float(getattr(csm, "L2", np.nan)),
                    "Ls": float(getattr(csm, "Ls", np.nan)),
                    "Lr": float(getattr(csm, "Lr", np.nan)),
                    "theta_1": float(getattr(csm, "theta_1", np.nan)),
                    "delta_1": float(getattr(csm, "delta_1", np.nan)),
                    "theta_2": float(getattr(csm, "theta_2", np.nan)),
                    "delta_2": float(getattr(csm, "delta_2", np.nan)),
                }

    if mode == target_mode and best_point is not None:
        print("[mode {:d}] Point with max radius from z-axis:".format(target_mode))
        print("  max radius r = {:.6f}".format(max_r))
        print("  point (x, y, z) =", best_point)
        print("  config =", best_cfg)
    # END DEBUG

    return np.array(workspace_points)

def generate_workspace_data(csm, mode, num_samples=2500, show_progress=True, desc=None):
    """
    生成给定 mode 的工作空间采样数据。
    - show_progress: 是否显示 tqdm 进度条
    - desc: 进度条前缀描述(默认 'Mode {mode}')
    """
    csm.state_transition(csm.mode, mode)
    workspace_data = []

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, total=num_samples, desc=desc or f"Mode {mode}", unit="sample", dynamic_ncols=True)

    for _ in iterator:
        # 随机生成姿态参数
        csm.phi = np.random.uniform(0, 2 * np.pi)
        if mode == 1:
            csm.L2 = np.random.uniform(0, csm.L_20)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 2:
            csm.Lr = np.random.uniform(0, csm.L_r0)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 3:
            csm.L1 = np.random.uniform(0, csm.L_10)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 4:
            csm.Ls = np.random.uniform(0, csm.L_s0)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)

        csm.update()
        pose = csm.pose.copy().tolist()
        config = {
            "phi": csm.phi,
            "theta_1": csm.theta_1,
            "theta_2": csm.theta_2,
            "delta_1": csm.delta_1,
            "delta_2": csm.delta_2,
            "L1": csm.L1,
            "L2": csm.L2,
            "Lr": csm.Lr,
            "Ls": csm.Ls,
        }
        workspace_data.append({
            "mode": mode,
            "pose": pose,
            "config": config
        })

    return workspace_data

def plot_workspace(ax, workspace, label, color):
    ax.scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], s=2, label=label, color=color)

def save_workspace_to_file(filename, data):
    path = Path(filename)
    # parents=True 表示连带父文件夹一起创建，exist_ok=True 表示如果已存在就不报错
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f)
        
if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 20))
    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, 0.001)
    
    # 生成并保存工作空间数据
    all_workspace_data = []
    num_samples_per_mode = [125000, 125000, 125000, 125000]  # 为每个模式指定样本数量
    for mode in range(1, 5):  # 生成模式1到模式4的数据
        workspace_data = generate_workspace_data(csm, mode, num_samples=num_samples_per_mode[mode-1])
        all_workspace_data.extend(workspace_data)
    
    save_workspace_to_file("./data/workspace_data.json", all_workspace_data)
