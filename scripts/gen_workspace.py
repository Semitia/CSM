"""
Module: gen_workspace.py
Description: Script to generate workspace points for different modes of the CSM.
"""
import json
import numpy as np
from csm.model import CSM
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def _build_data_dict(csm, mode):
    return {
        "mode": mode,
        "pose": csm.pose.copy().tolist(),
        "rotation_matrix": csm.rotation_matrix.tolist(),
        "config": {
            "phi": float(csm.phi),
            "theta_1": float(csm.theta_1),
            "theta_2": float(csm.theta_2),
            "delta_1": float(csm.delta_1),
            "delta_2": float(csm.delta_2),
            "L1": float(csm.L1),
            "L2": float(csm.L2),
            "Lr": float(csm.Lr),
            "Ls": float(csm.Ls),
        },
    }


def _linspace_from_zero(stop, num):
    if stop <= 0:
        return np.array([0.0])
    return np.linspace(0, stop, num)


def _uniform_mode_resolution(mode, grid_res):
    if mode in (3, 4):
        return max(5, grid_res - 5)
    return grid_res


def _uniform_total_samples(csm, mode, mode_res):
    dependent_theta_total = 1 + (mode_res - 1) * mode_res

    if mode == 1:
        return (mode_res ** 2) * dependent_theta_total
    if mode == 2:
        return mode_res ** 4
    if mode == 3:
        return (mode_res ** 4) * dependent_theta_total
    if mode == 4:
        return mode_res ** 6
    raise ValueError(f"Unsupported mode: {mode}")


def generate_workspace_data(csm, mode, method="random", num_samples=2500, grid_res=15,
                            show_progress=True, desc=None):
    """
    生成给定 mode 的工作空间采样数据。
    - method: "random" 为随机采样, "uniform" 为均匀网格采样
    - grid_res: 当 method="uniform" 时，每个变量的基础分辨率
    - show_progress: 是否显示 tqdm 进度条
    - desc: 进度条前缀描述(默认 'Mode {mode}')
    """
    if method not in {"random", "uniform"}:
        raise ValueError(f"Unsupported sampling method: {method}")

    csm.state_transition(csm.mode, mode)
    workspace_data = []

    if method == "random":
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=num_samples,
                desc=desc or f"Mode {mode} (Random)",
                unit="sample",
                dynamic_ncols=True,
            )

        for _ in iterator:
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
            workspace_data.append(_build_data_dict(csm, mode))

    else:
        mode_res = _uniform_mode_resolution(mode, grid_res)
        phi_vals = np.linspace(0, 2 * np.pi, mode_res)
        progress = None

        if show_progress:
            progress = tqdm(
                total=_uniform_total_samples(csm, mode, mode_res),
                desc=desc or f"Mode {mode} (Uniform)",
                unit="sample",
                dynamic_ncols=True,
            )

        try:
            if mode == 1:
                for phi in phi_vals:
                    csm.phi = phi
                    for L2 in _linspace_from_zero(csm.L_20, mode_res):
                        csm.L2 = L2
                        for theta_2 in _linspace_from_zero(csm.kappa_20 * L2, mode_res):
                            csm.theta_2 = theta_2
                            for delta_2 in np.linspace(0, 2 * np.pi, mode_res):
                                csm.delta_2 = delta_2
                                csm.update()
                                workspace_data.append(_build_data_dict(csm, mode))
                                if progress is not None:
                                    progress.update(1)

            elif mode == 2:
                for phi in phi_vals:
                    csm.phi = phi
                    for Lr in _linspace_from_zero(csm.L_r0, mode_res):
                        csm.Lr = Lr
                        for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                            csm.theta_2 = theta_2
                            for delta_2 in np.linspace(0, 2 * np.pi, mode_res):
                                csm.delta_2 = delta_2
                                csm.update()
                                workspace_data.append(_build_data_dict(csm, mode))
                                if progress is not None:
                                    progress.update(1)

            elif mode == 3:
                for phi in phi_vals:
                    csm.phi = phi
                    for L1 in _linspace_from_zero(csm.L_10, mode_res):
                        csm.L1 = L1
                        for theta_1 in _linspace_from_zero(csm.kappa_10 * L1, mode_res):
                            csm.theta_1 = theta_1
                            for delta_1 in np.linspace(0, 2 * np.pi, mode_res):
                                csm.delta_1 = delta_1
                                for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                                    csm.theta_2 = theta_2
                                    for delta_2 in np.linspace(0, 2 * np.pi, mode_res):
                                        csm.delta_2 = delta_2
                                        csm.update()
                                        workspace_data.append(_build_data_dict(csm, mode))
                                        if progress is not None:
                                            progress.update(1)

            elif mode == 4:
                for phi in phi_vals:
                    csm.phi = phi
                    for Ls in _linspace_from_zero(csm.L_s0, mode_res):
                        csm.Ls = Ls
                        for theta_1 in _linspace_from_zero(csm.kappa_10 * csm.L1, mode_res):
                            csm.theta_1 = theta_1
                            for delta_1 in np.linspace(0, 2 * np.pi, mode_res):
                                csm.delta_1 = delta_1
                                for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                                    csm.theta_2 = theta_2
                                    for delta_2 in np.linspace(0, 2 * np.pi, mode_res):
                                        csm.delta_2 = delta_2
                                        csm.update()
                                        workspace_data.append(_build_data_dict(csm, mode))
                                        if progress is not None:
                                            progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    return workspace_data

def plot_workspace(ax, workspace, label, color):
    ax.scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], s=2, label=label, color=color)

def save_workspace_to_file(filename, data):
    path = Path(filename)
    # parents=True 表示连带父文件夹一起创建，exist_ok=True 表示如果已存在就不报错
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"Saved workspace data file: {path.name}")
    print(f"Saved workspace data path: {path.resolve()}")
        
if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 20))
    config_name = "csm_cfg_3.4mm.yaml"
    config_path = Path("./config") / config_name
    csm = CSM.from_config(config_path)
    
    # 生成并保存工作空间数据
    all_workspace_data = []
    num_samples_per_mode = [200000, 200000, 200000, 0]  # 为每个模式指定样本数量，3.4mm 模型只需要 3 个模式
    sampling_method = "uniform"
    uniform_grid_res = 15
    output_filename = f"workspace_data_{Path(config_name).stem}_{sampling_method}.json"
    output_path = Path("./data") / output_filename

    for mode in range(1, 4):  # 3.4mm 模型只需要模式1到模式3
        workspace_data = generate_workspace_data(
            csm,
            mode,
            method=sampling_method,
            num_samples=num_samples_per_mode[mode - 1],
            grid_res=uniform_grid_res,
        )
        all_workspace_data.extend(workspace_data)

    save_workspace_to_file(output_path, all_workspace_data)
