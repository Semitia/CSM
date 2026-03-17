"""
统一 FK cmap 构建入口
支持所有注册的机器人类型，只需修改 ROBOT_NAME 变量
"""
import numpy as np
import time
import os
import json
import multiprocessing as mp
import ctypes
import signal
from tqdm import tqdm
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# 将 dexterous_analyse 目录加入路径，确保 ws_discretizer 和 robots 可以被导入
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ws_discretizer import WsDiscretizer
from robots import ROBOT_REGISTRY, DISCR_CONFIG_REGISTRY, SAVE_PATH_REGISTRY, robot_from_config_dict


# ==============================================================
# Halton 序列生成器（独立函数，可在子进程中使用）
# ==============================================================
def get_halton_sequence(index: int, bases: list) -> np.ndarray:
    """生成 Halton 低差异序列"""
    res = np.zeros(len(bases))
    for j, b in enumerate(bases):
        f = 1.0 / b
        i = index
        val = 0.0
        while i > 0:
            val += f * (i % b)
            i = i // b
            f = f / b
        res[j] = val
    return res


# ==============================================================
# Worker 进程任务（完全通用，不含任何机器人特定逻辑）
# ==============================================================
def worker_task_fk(
    worker_id, start_step, max_fk, num_processes,
    robot_config_dict, discr_config_dict,
    shared_array_base, cmap_shape,
    shared_steps_counter, stop_event,
):
    """
    通用 FK worker 进程
    通过 robot_config_dict 重建机器人实例，不含任何机器人特定逻辑
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # 1. 恢复离散化器和共享内存视图
    discretizer = WsDiscretizer.from_config(discr_config_dict)
    cmap_shared = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)

    # 2. 在子进程内重建并加载机器人
    robot = robot_from_config_dict(robot_config_dict)
    robot.load()

    # 3. 准备 Halton 基数和旋转查找表
    halton_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][: robot.n_halton_dims]

    R_table = np.zeros((discretizer.n_p, discretizer.m_o, 3, 3))
    for f in discretizer.discrete_frames:
        R_table[f["point_idx"], f["rot_idx"]] = f["T"][:3, :3]

    # 4. 判断是否为 CSM 机器人（需要特殊的 mode 循环）
    is_csm = robot_config_dict.get("type") == "csm"
    if is_csm:
        valid_modes = robot.valid_modes
        num_modes = len(valid_modes)

    # 5. 交织循环
    for step in range(start_step + worker_id, max_fk + 1, num_processes):
        if stop_event.is_set():
            break

        if is_csm:
            # CSM：循环分配 mode，并对 step 做归一化
            mode = valid_modes[step % num_modes]
            halton_vals = get_halton_sequence(step // num_modes, halton_bases)
            robot.map_halton_to_csm(mode, halton_vals)
            tcp_pose, tcp_rot = robot.fk(None)
        else:
            # Pinocchio 机器人：直接映射 Halton -> q -> FK
            halton_vals = get_halton_sequence(step, halton_bases)
            q = robot.sample_q(halton_vals)
            tcp_pose, tcp_rot = robot.fk(q)

        # 6. 更新 cmap
        g_idx = discretizer.get_voxel_index(tcp_pose)
        if g_idx is not None and np.all(g_idx >= 0) and np.all(g_idx < discretizer.n_c):
            Z_tcp = tcp_rot[:, 2]
            p_idx = np.argmax(np.dot(discretizer.sphere_points, -Z_tcp))
            r_idx = np.argmax(np.einsum("nij,ij->n", R_table[p_idx], tcp_rot))

            if not cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx]:
                if is_csm or not robot.has_collision(q):
                    cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx] = True

        shared_steps_counter[worker_id] += 1


# ==============================================================
# 主程序
# ==============================================================
if __name__ == "__main__":
    # ==================== 只需修改这里 ====================
    ROBOT_NAME = "csm"   # 可选: 'ur5', 'panda', 'csm', 'csm_0', 'csm_tool'
    max_fk = 100_000_000
    USE_RICH = True
    # ======================================================

    # 自动从注册表获取配置
    if ROBOT_NAME not in ROBOT_REGISTRY:
        raise ValueError(f"未知机器人: '{ROBOT_NAME}'，可用: {list(ROBOT_REGISTRY.keys())}")

    robot_instance = ROBOT_REGISTRY[ROBOT_NAME]()
    robot_config_dict = robot_instance.get_config_dict()

    discr_config_path = DISCR_CONFIG_REGISTRY[ROBOT_NAME]
    save_path = SAVE_PATH_REGISTRY[ROBOT_NAME].replace("_fk_cmap", "_fk_cmap_mp")

    with open(discr_config_path, "r") as f:
        discr_config_dict = json.load(f)
    discretizer = WsDiscretizer.from_config(discr_config_dict)

    # 对 CSM，需要提前检测 valid_modes 并写入 robot_config_dict
    if robot_config_dict.get("type") == "csm":
        robot_instance.load()
        robot_config_dict["valid_modes"] = robot_instance.valid_modes
        print(f"CSM 有效模式: {robot_instance.valid_modes}")

    cmap_shape = (discretizer.n_c, discretizer.n_c, discretizer.n_c, discretizer.n_p, discretizer.m_o)
    total_blocks = int(np.prod(cmap_shape))

    shared_array_base = mp.RawArray(ctypes.c_bool, total_blocks)
    cmap_view = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)

    start_step = 1
    total_filled = 0

    # ==================== 断点续传 ====================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"发现已有保存文件 {save_path}，正在验证配置...")
        saved = np.load(save_path, allow_pickle=True)
        if "config" in saved and saved["config"].item() == discr_config_dict:
            np.copyto(cmap_view, saved["cmap"])
            start_step = int(saved.get("step", 1))
            total_filled = int(np.sum(cmap_view))
            print(f"配置校验通过！恢复进度: Step={start_step}, 已填充: {total_filled / total_blocks * 100:.4f}%")
        else:
            print("[错误] 配置文件不匹配或缺失，程序已停止。")
            exit()
    else:
        print(f"启动全新构建任务 [{ROBOT_NAME}]...")
        cmap_view.fill(False)

    if start_step >= max_fk:
        print("设定步数已全部完成，无需继续。")
        exit()

    # ==================== 进程管理 ====================
    num_processes = max(1, mp.cpu_count() - 2)
    print(f"\n开始并行 FK 采样 [{ROBOT_NAME}] (进程数: {num_processes})，按 Ctrl+C 可安全中断并保存...\n")

    shared_steps_counter = mp.RawArray("i", num_processes)
    stop_event = mp.Event()
    processes = []

    for i in range(num_processes):
        p = mp.Process(
            target=worker_task_fk,
            args=(
                i, start_step, max_fk, num_processes,
                robot_config_dict, discr_config_dict,
                shared_array_base, cmap_shape,
                shared_steps_counter, stop_event,
            ),
        )
        processes.append(p)
        p.start()

    # ==================== 主进程监控 ====================
    try:
        last_total_steps = 0
        last_filled = total_filled

        if USE_RICH:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                main_task = progress.add_task("[bold green]总进度", total=max_fk, completed=start_step)
                worker_tasks = []
                for i in range(num_processes):
                    worker_total = len(range(start_step + i, max_fk + 1, num_processes))
                    worker_tasks.append(progress.add_task(f"[cyan]Worker {i:02d}", total=worker_total))

                while any(p.is_alive() for p in processes):
                    time.sleep(1.0)
                    current_total_steps = sum(shared_steps_counter)
                    step_delta = current_total_steps - last_total_steps
                    current_filled = int(np.sum(cmap_view))
                    filled_delta = current_filled - last_filled
                    filled_percent = current_filled / total_blocks * 100.0
                    cost = (step_delta / filled_delta) if filled_delta > 0 else float("inf")

                    progress.update(
                        main_task,
                        completed=start_step + current_total_steps,
                        description=(
                            f"[bold green]总进度[/] "
                            f"[yellow](Filled: {filled_percent:.4f}% | Cost: {cost:.2f} FK/区块)[/]"
                        ),
                    )
                    for i in range(num_processes):
                        progress.update(worker_tasks[i], completed=shared_steps_counter[i])

                    last_total_steps = current_total_steps
                    last_filled = current_filled
        else:
            pbar = tqdm(total=max_fk, initial=start_step, desc=f"FK [{ROBOT_NAME}]")
            while any(p.is_alive() for p in processes):
                time.sleep(1.0)
                current_total_steps = sum(shared_steps_counter)
                step_delta = current_total_steps - last_total_steps
                current_filled = int(np.sum(cmap_view))
                filled_delta = current_filled - last_filled
                pbar.update(step_delta)
                filled_percent = current_filled / total_blocks * 100.0
                cost = (step_delta / filled_delta) if filled_delta > 0 else float("inf")
                pbar.set_postfix({"Filled": f"{filled_percent:.4f}%", "Cost": f"{cost:.2f} FK/区块"})
                last_total_steps = current_total_steps
                last_filled = current_filled

    except KeyboardInterrupt:
        print("\n\n收到中断指令，正在通知所有工作进程优雅退出 (请等待 1-2 秒)...")
        stop_event.set()
        for p in processes:
            p.join()

    finally:
        if not USE_RICH and "pbar" in locals():
            pbar.close()

        min_steps_done = min(shared_steps_counter)
        safe_step_to_save = start_step + min_steps_done * num_processes

        print(f"\n正在将数据落盘 (安全步数锚点: Step={safe_step_to_save})...")
        np.savez_compressed(save_path, cmap=cmap_view, config=discr_config_dict, step=safe_step_to_save)
        print("保存完毕！下次启动将从该锚点无缝恢复。")
