"""
Module: gen_workspace.py
Description: Script to generate workspace points for different modes of the CSM.
"""
import multiprocessing as mp
import numpy as np
from csm.model import CSM
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat

def _build_data_dict(csm, mode, pose_only=False):
    if pose_only:
        return {
            "mode": mode,
            "position": csm.pose[:3].copy().tolist(),
        }
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


def _uniform_phi_resolution(grid_res, phi_grid_res):
    if phi_grid_res is None:
        return grid_res
    return max(1, int(phi_grid_res))


def _uniform_total_samples(csm, mode, mode_res, phi_res):
    dependent_theta_total = 1 + (mode_res - 1) * mode_res

    if mode == 1:
        return phi_res * mode_res * dependent_theta_total
    if mode == 2:
        return phi_res * (mode_res ** 3)
    if mode == 3:
        return phi_res * (mode_res ** 2) * dependent_theta_total
    if mode == 4:
        return phi_res * (mode_res ** 5)
    raise ValueError(f"Unsupported mode: {mode}")


def _serialize_csm_init_kwargs(csm):
    return {
        "L_10": float(csm.L_10),
        "L_20": float(csm.L_20),
        "L_r0": float(csm.L_r0),
        "L_s0": float(csm.L_s0),
        "L_tool": float(csm.L_tool),
        "theta1_max": float(csm.kappa_10 * csm.L_10),
        "theta2_max": float(csm.kappa_20 * csm.L_20),
        "delta_t": float(csm.delta_t),
    }


def _split_evenly(total, num_parts):
    num_parts = max(1, min(num_parts, total))
    base, remainder = divmod(total, num_parts)
    return [base + (1 if i < remainder else 0) for i in range(num_parts) if base + (1 if i < remainder else 0) > 0]


def _chunk_sequence(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]


def _prepare_csm_for_mode(csm, mode):
    if mode == 1:
        csm.set_state(mode=1, phi=0, L1=0, L2=csm.L_20, Lr=0, Ls=0, theta_1=0, theta_2=0, delta_1=0, delta_2=0)
    elif mode == 2:
        csm.set_state(mode=2, phi=0, L1=0, L2=csm.L_20, Lr=0, Ls=0, theta_1=0, theta_2=0, delta_1=0, delta_2=0)
    elif mode == 3:
        csm.set_state(mode=3, phi=0, L1=0, L2=csm.L_20, Lr=csm.L_r0, Ls=0, theta_1=0, theta_2=0, delta_1=0, delta_2=0)
    elif mode == 4:
        csm.set_state(
            mode=4,
            phi=0,
            L1=csm.L_10,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=0,
            theta_1=0,
            theta_2=0,
            delta_1=0,
            delta_2=0,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _create_worker_csm(csm_init_kwargs, mode):
    csm = CSM(**csm_init_kwargs)
    _prepare_csm_for_mode(csm, mode)
    return csm


def _random_sample_once(csm, mode, rng, pose_only=False):
    csm.phi = rng.uniform(0, 2 * np.pi)
    if mode == 1:
        csm.L2 = rng.uniform(0, csm.L_20)
        csm.theta_2 = rng.uniform(0, csm.kappa_20 * csm.L2)
        csm.delta_2 = rng.uniform(0, 2 * np.pi)
    elif mode == 2:
        csm.Lr = rng.uniform(0, csm.L_r0)
        csm.theta_2 = rng.uniform(0, csm.kappa_20 * csm.L2)
        csm.delta_2 = rng.uniform(0, 2 * np.pi)
    elif mode == 3:
        csm.L1 = rng.uniform(0, csm.L_10)
        csm.theta_1 = rng.uniform(0, csm.kappa_10 * csm.L1)
        csm.delta_1 = rng.uniform(0, 2 * np.pi)
        csm.theta_2 = rng.uniform(0, csm.kappa_20 * csm.L2)
        csm.delta_2 = rng.uniform(0, 2 * np.pi)
    elif mode == 4:
        csm.Ls = rng.uniform(0, csm.L_s0)
        csm.theta_1 = rng.uniform(0, csm.kappa_10 * csm.L1)
        csm.delta_1 = rng.uniform(0, 2 * np.pi)
        csm.theta_2 = rng.uniform(0, csm.kappa_20 * csm.L2)
        csm.delta_2 = rng.uniform(0, 2 * np.pi)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    csm.update()
    return _build_data_dict(csm, mode, pose_only=pose_only)


def _generate_random_workspace_chunk(args):
    csm_init_kwargs, mode, sample_count, seed, pose_only = args
    rng = np.random.default_rng(seed)
    csm = _create_worker_csm(csm_init_kwargs, mode)
    return [_random_sample_once(csm, mode, rng, pose_only=pose_only) for _ in range(sample_count)]


def _generate_uniform_workspace_chunk(args):
    csm_init_kwargs, mode, outer_vals, mode_res, pose_only = args
    csm = _create_worker_csm(csm_init_kwargs, mode)
    workspace_data = []
    delta_vals = np.linspace(0, 2 * np.pi, mode_res)

    if mode == 1:
        for phi in outer_vals:
            csm.phi = phi
            for L2 in _linspace_from_zero(csm.L_20, mode_res):
                csm.L2 = L2
                for theta_2 in _linspace_from_zero(csm.kappa_20 * L2, mode_res):
                    csm.theta_2 = theta_2
                    for delta_2 in delta_vals:
                        csm.delta_2 = delta_2
                        csm.update()
                        workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))

    elif mode == 2:
        for phi in outer_vals:
            csm.phi = phi
            for Lr in _linspace_from_zero(csm.L_r0, mode_res):
                csm.Lr = Lr
                for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                    csm.theta_2 = theta_2
                    for delta_2 in delta_vals:
                        csm.delta_2 = delta_2
                        csm.update()
                        workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))

    elif mode == 3:
        csm.delta_1 = 0.0
        for phi in outer_vals:
            csm.phi = phi
            for L1 in _linspace_from_zero(csm.L_10, mode_res):
                csm.L1 = L1
                for theta_1 in _linspace_from_zero(csm.kappa_10 * L1, mode_res):
                    csm.theta_1 = theta_1
                    for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                        csm.theta_2 = theta_2
                        for delta_2 in delta_vals:
                            csm.delta_2 = delta_2
                            csm.update()
                            workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))

    elif mode == 4:
        for phi in outer_vals:
            csm.phi = phi
            for Ls in _linspace_from_zero(csm.L_s0, mode_res):
                csm.Ls = Ls
                for theta_1 in _linspace_from_zero(csm.kappa_10 * csm.L1, mode_res):
                    csm.theta_1 = theta_1
                    for delta_1 in delta_vals:
                        csm.delta_1 = delta_1
                        for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                            csm.theta_2 = theta_2
                            for delta_2 in delta_vals:
                                csm.delta_2 = delta_2
                                csm.update()
                                workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return workspace_data


def _generate_workspace_data_parallel(csm, mode, method, num_samples, grid_res, phi_grid_res, show_progress, desc, num_workers, pose_only):
    csm_init_kwargs = _serialize_csm_init_kwargs(csm)
    workspace_data = []

    if method == "random":
        num_chunks = max(num_workers * 4, 1)
        chunk_sizes = _split_evenly(num_samples, num_chunks)
        seed_sequence = np.random.SeedSequence()
        child_sequences = seed_sequence.spawn(len(chunk_sizes))
        tasks = [
            (csm_init_kwargs, mode, chunk_size, int(child_seq.generate_state(1, dtype=np.uint64)[0]), pose_only)
            for chunk_size, child_seq in zip(chunk_sizes, child_sequences)
        ]
        total_samples = num_samples
        worker_fn = _generate_random_workspace_chunk
    else:
        mode_res = _uniform_mode_resolution(mode, grid_res)
        phi_res = _uniform_phi_resolution(grid_res, phi_grid_res)
        outer_vals = np.linspace(0, 2 * np.pi, phi_res)
        outer_chunk_size = max(1, len(outer_vals) // (num_workers * 4))
        outer_chunks = _chunk_sequence(outer_vals, outer_chunk_size)
        tasks = [(csm_init_kwargs, mode, outer_chunk, mode_res, pose_only) for outer_chunk in outer_chunks]
        total_samples = _uniform_total_samples(csm, mode, mode_res, phi_res)
        worker_fn = _generate_uniform_workspace_chunk

    progress = None
    if show_progress:
        progress = tqdm(
            total=total_samples,
            desc=desc or f"Mode {mode} ({method.title()} MP)",
            unit="sample",
            dynamic_ncols=True,
        )

    try:
        with mp.Pool(processes=num_workers) as pool:
            for chunk_data in pool.imap_unordered(worker_fn, tasks):
                workspace_data.extend(chunk_data)
                if progress is not None:
                    progress.update(len(chunk_data))
    finally:
        if progress is not None:
            progress.close()

    return workspace_data


def generate_workspace_data(csm, mode, method="random", num_samples=2500, grid_res=15, phi_grid_res=None,
                            show_progress=True, desc=None, num_workers=1, pose_only=False):
    """
    生成给定 mode 的工作空间采样数据。
    - method: "random" 为随机采样, "uniform" 为均匀网格采样
    - grid_res: 当 method="uniform" 时，每个变量的基础分辨率
    - phi_grid_res: 当 method="uniform" 时，phi 的离散数量；默认与 grid_res 一致
    - show_progress: 是否显示 tqdm 进度条
    - desc: 进度条前缀描述(默认 'Mode {mode}')
    """
    if method not in {"random", "uniform"}:
        raise ValueError(f"Unsupported sampling method: {method}")

    if num_workers is None:
        num_workers = 1
    num_workers = max(1, int(num_workers))

    if num_workers > 1:
        return _generate_workspace_data_parallel(
            csm,
            mode,
            method=method,
            num_samples=num_samples,
            grid_res=grid_res,
            phi_grid_res=phi_grid_res,
            show_progress=show_progress,
            desc=desc,
            num_workers=num_workers,
            pose_only=pose_only,
        )

    _prepare_csm_for_mode(csm, mode)
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
            workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))

    else:
        mode_res = _uniform_mode_resolution(mode, grid_res)
        phi_res = _uniform_phi_resolution(grid_res, phi_grid_res)
        phi_vals = np.linspace(0, 2 * np.pi, phi_res)
        progress = None

        if show_progress:
            progress = tqdm(
                total=_uniform_total_samples(csm, mode, mode_res, phi_res),
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
                                workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))
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
                                workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))
                                if progress is not None:
                                    progress.update(1)

            elif mode == 3:
                csm.delta_1 = 0.0
                for phi in phi_vals:
                    csm.phi = phi
                    for L1 in _linspace_from_zero(csm.L_10, mode_res):
                        csm.L1 = L1
                        for theta_1 in _linspace_from_zero(csm.kappa_10 * L1, mode_res):
                            csm.theta_1 = theta_1
                            for theta_2 in _linspace_from_zero(csm.kappa_20 * csm.L2, mode_res):
                                csm.theta_2 = theta_2
                                for delta_2 in np.linspace(0, 2 * np.pi, mode_res):
                                    csm.delta_2 = delta_2
                                    csm.update()
                                    workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))
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
                                        workspace_data.append(_build_data_dict(csm, mode, pose_only=pose_only))
                                        if progress is not None:
                                            progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    return workspace_data

def _extract_position(item):
    if "position" in item:
        return item["position"]
    return item["pose"][:3]


def save_workspace_to_npz(filename, data, config_name, sampling_method, grid_res, num_samples_per_mode,
                          phi_grid_res=None, pose_only=False):
    """Save workspace data to NPZ file (compact binary format)."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode_points = {}
    mode_rotations = {} if not pose_only else None
    mode_configs = {} if not pose_only else None

    for mode in (1, 2, 3, 4):
        entries = [item for item in data if int(item["mode"]) == mode]
        if entries:
            mode_points[mode] = np.asarray([_extract_position(item) for item in entries], dtype=np.float64)
            if not pose_only:
                mode_rotations[mode] = np.asarray([item["rotation_matrix"] for item in entries], dtype=np.float64)
                mode_configs[mode] = np.asarray([
                    [
                        item["config"]["phi"],
                        item["config"]["theta_1"],
                        item["config"]["theta_2"],
                        item["config"]["delta_1"],
                        item["config"]["delta_2"],
                        item["config"]["L1"],
                        item["config"]["L2"],
                        item["config"]["Lr"],
                        item["config"]["Ls"],
                    ]
                    for item in entries
                ], dtype=np.float64)
        else:
            mode_points[mode] = np.empty((0, 3), dtype=np.float64)
            if not pose_only:
                mode_rotations[mode] = np.empty((0, 3, 3), dtype=np.float64)
                mode_configs[mode] = np.empty((0, 9), dtype=np.float64)

    all_points = np.vstack([mode_points[m] for m in (1, 2, 3, 4) if mode_points[m].size]) if any(
        mode_points[m].size for m in (1, 2, 3, 4)
    ) else np.empty((0, 3), dtype=np.float64)

    npz_payload = {
        "config_name": config_name,
        "sampling_method": sampling_method,
        "uniform_grid_res": grid_res,
        "uniform_phi_grid_res": -1 if phi_grid_res is None else int(phi_grid_res),
        "num_samples_per_mode": np.asarray(num_samples_per_mode, dtype=np.int32),
        "pose_only": np.asarray(pose_only),
        "all_points": all_points,
    }
    if not pose_only:
        npz_payload["config_fields"] = ["phi", "theta_1", "theta_2", "delta_1", "delta_2", "L1", "L2", "Lr", "Ls"]

    for mode in (1, 2, 3, 4):
        npz_payload[f"mode{mode}_points"] = mode_points[mode]
        if not pose_only:
            npz_payload[f"mode{mode}_rotation_matrices"] = mode_rotations[mode]
            npz_payload[f"mode{mode}_configs"] = mode_configs[mode]

    np.savez_compressed(path, **npz_payload)
    print(f"Saved NPZ workspace file: {path.name}")
    print(f"Saved NPZ workspace path: {path.resolve()}")


def save_workspace_to_mat_file(filename, data, config_name, sampling_method, grid_res, num_samples_per_mode,
                               phi_grid_res=None, pose_only=False):
    """Save workspace data to MAT file (for MATLAB compatibility)."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode_points = {}
    mode_rotations = {} if not pose_only else None
    mode_configs = {} if not pose_only else None

    for mode in (1, 2, 3, 4):
        entries = [item for item in data if int(item["mode"]) == mode]
        if entries:
            mode_points[mode] = np.asarray([_extract_position(item) for item in entries], dtype=float)
            if not pose_only:
                mode_rotations[mode] = np.asarray([item["rotation_matrix"] for item in entries], dtype=float)
                mode_configs[mode] = np.asarray([
                    [
                        item["config"]["phi"],
                        item["config"]["theta_1"],
                        item["config"]["theta_2"],
                        item["config"]["delta_1"],
                        item["config"]["delta_2"],
                        item["config"]["L1"],
                        item["config"]["L2"],
                        item["config"]["Lr"],
                        item["config"]["Ls"],
                    ]
                    for item in entries
                ], dtype=float)
        else:
            mode_points[mode] = np.empty((0, 3), dtype=float)
            if not pose_only:
                mode_rotations[mode] = np.empty((0, 3, 3), dtype=float)
                mode_configs[mode] = np.empty((0, 9), dtype=float)

    all_points = np.vstack([mode_points[m] for m in (1, 2, 3, 4) if mode_points[m].size]) if any(
        mode_points[m].size for m in (1, 2, 3, 4)
    ) else np.empty((0, 3), dtype=float)

    mat_payload = {
        "config_name": np.asarray([config_name], dtype=object),
        "sampling_method": np.asarray([sampling_method], dtype=object),
        "uniform_grid_res": np.asarray([[grid_res]], dtype=np.int32),
        "uniform_phi_grid_res": np.asarray([[-1 if phi_grid_res is None else int(phi_grid_res)]], dtype=np.int32),
        "num_samples_per_mode": np.asarray([num_samples_per_mode], dtype=np.int32),
        "pose_only": np.asarray([[pose_only]], dtype=np.uint8),
        "all_points": all_points,
    }
    if not pose_only:
        mat_payload["config_fields"] = np.asarray(
            [["phi", "theta_1", "theta_2", "delta_1", "delta_2", "L1", "L2", "Lr", "Ls"]],
            dtype=object,
        )

    for mode in (1, 2, 3, 4):
        mat_payload[f"mode{mode}_points"] = mode_points[mode]
        if not pose_only:
            mat_payload[f"mode{mode}_rotation_matrices"] = mode_rotations[mode]
            mat_payload[f"mode{mode}_configs"] = mode_configs[mode]

    savemat(path, mat_payload, do_compression=True)
    print(f"Saved MATLAB workspace file: {path.name}")
    print(f"Saved MATLAB workspace path: {path.resolve()}")
        
if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 20))
    config_name = "csm_cfg_3.4mm.yaml"
    config_path = Path("./config") / config_name
    csm = CSM.from_config(config_path)
    num_workers = max(1, mp.cpu_count() - 2)
    
    # 生成并保存工作空间数据
    all_workspace_data = []
    num_samples_per_mode = [200000, 200000, 200000, 0]  # 为每个模式指定样本数量，3.4mm 模型只需要 3 个模式
    sampling_method = "uniform"
    uniform_grid_res = 12
    uniform_phi_grid_res = 60
    pose_only = True
    output_stem = f"workspace_data_{Path(config_name).stem}_{sampling_method}"
    output_npz_path = Path("./data") / f"{output_stem}.npz"
    output_mat_path = Path("./data") / f"{output_stem}.mat"

    for mode in range(1, 4):  # 3.4mm 模型只需要模式1到模式3
        workspace_data = generate_workspace_data(
            csm,
            mode,
            method=sampling_method,
            num_samples=num_samples_per_mode[mode - 1],
            grid_res=uniform_grid_res,
            phi_grid_res=uniform_phi_grid_res,
            num_workers=num_workers,
            pose_only=pose_only,
        )
        all_workspace_data.extend(workspace_data)

    save_workspace_to_npz(
        output_npz_path,
        all_workspace_data,
        config_name=config_name,
        sampling_method=sampling_method,
        grid_res=uniform_grid_res,
        num_samples_per_mode=num_samples_per_mode,
        phi_grid_res=uniform_phi_grid_res,
        pose_only=pose_only,
    )
    save_workspace_to_mat_file(
        output_mat_path,
        all_workspace_data,
        config_name=config_name,
        sampling_method=sampling_method,
        grid_res=uniform_grid_res,
        num_samples_per_mode=num_samples_per_mode,
        phi_grid_res=uniform_phi_grid_res,
        pose_only=pose_only,
    )
