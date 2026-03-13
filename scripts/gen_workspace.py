"""
Module: gen_workspace.py
Description: Script to generate workspace points for different modes of the CSM.
"""
import json
import yaml
import numpy as np
from csm.model import CSM
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

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
            "rotation_matrix": csm.rotation_matrix.tolist(),
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
    config_name = "csm_config_3.4mm.yaml"
    config_path = Path("./config") / config_name
    csm = CSM.from_config(config_path)
    
    # 生成并保存工作空间数据
    all_workspace_data = []
    # num_samples_per_mode = [125000, 125000, 125000, 125000]  # 为每个模式指定样本数量
    num_samples_per_mode = [200000, 200000, 200000, 0]  # 为每个模式指定样本数量，3.4mm 模型只需要 3 个模式
    for mode in range(1, 5):  # 生成模式1到模式4的数据
        workspace_data = generate_workspace_data(csm, mode, num_samples=num_samples_per_mode[mode-1])
        all_workspace_data.extend(workspace_data)

    save_workspace_to_file("./data/workspace_data_3.4mm.json", all_workspace_data)
