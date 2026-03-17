"""
统一 IK cmap 构建入口
支持所有支持 IK 的机器人类型（Pinocchio 系列），只需修改 ROBOT_NAME 变量
"""
import numpy as np
import pinocchio as pin
import multiprocessing as mp
import ctypes
import os
import json
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from ws_discretizer import WsDiscretizer
from robots import ROBOT_REGISTRY, DISCR_CONFIG_REGISTRY, SAVE_PATH_REGISTRY, robot_from_config_dict


# ==================
# Worker 进程任务（完全通用）
# ==================
def worker_task_ik(
    worker_id,
    num_samples,
    robot_config_dict,
    discr_config_dict,
    shared_array_base,
    cmap_shape,
):
    """
    通用 IK worker 进程
    """
    # 1. 恢复离散化器和共享内存视图
    discretizer = WsDiscretizer.from_config(discr_config_dict)
    cmap_shared = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)

    # 2. 在子进程内重建并加载机器人
    robot = robot_from_config_dict(robot_config_dict)
    robot.load()

    # 检查是否支持 IK
    if not robot.supports_ik:
        raise RuntimeError(f"机器人类型 '{robot_config_dict.get('type')}' 不支持 IK")

    # 3. 获取关节限位（用于随机采样）
    # 注意：这里假设 Pinocchio 机器人有 model 属性
    q_min = robot.model.lowerPositionLimit
    q_max = robot.model.upperPositionLimit

    # 4. 进度条（仅 worker 0 显示）
    iterator = range(num_samples)
    if worker_id == 0:
        iterator = tqdm(iterator, desc="Worker 0 [IK + Collision]", position=0)

    for _ in iterator:
        # 1. 随机采样初始关节配置
        q_r = np.random.uniform(q_min, q_max)

        # 2. FK 获取采样点位置
        tcp_pose, _ = robot.fk(q_r)

        # 3. 获取体素索引
        g_idx = discretizer.get_voxel_index(tcp_pose)
        if g_idx is None or np.any(g_idx < 0) or np.any(g_idx >= discretizer.n_c):
            continue

        voxel_center = discretizer.get_voxel_center(g_idx)

        # 4. 遍历该体素的所有离散姿态
        for frame_dict in discretizer.discrete_frames:
            p_idx, r_idx = frame_dict["point_idx"], frame_dict["rot_idx"]

            # 快速检查（无锁）
            if cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx]:
                continue

            # 组装目标位姿
            T_target = frame_dict["T"].copy()
            T_target[:3, 3] = voxel_center
            oMdes = pin.SE3(T_target)

            # 5. 求解 IK
            is_reachable, q_sol = robot.ik(oMdes, q_init=q_r)

            if is_reachable:
                # 6. 碰撞检测
                if not robot.has_collision(q_sol):
                    cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx] = True


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # ======= 只需修改这里 ==================
    ROBOT_NAME = "ur5"  # 可选: 'ur5', 'panda' (CSM 不支持 IK)
    total_samples = 100_000
    # ===========================

    # 自动从注册表获取配置
    if ROBOT_NAME not in ROBOT_REGISTRY:
        raise ValueError(f"未知机器人: '{ROBOT_NAME}'，可用: {list(ROBOT_REGISTRY.keys())}")

    robot_instance = ROBOT_REGISTRY[ROBOT_NAME]()
    robot_config_dict = robot_instance.get_config_dict()

    # 检查是否支持 IK
    if not robot_instance.supports_ik:
        raise RuntimeError(f"机器人 '{ROBOT_NAME}' 不支持 IK，请使用 build_fk_cmap_mp.py")

    discr_config_path = DISCR_CONFIG_REGISTRY[ROBOT_NAME]
    save_path = SAVE_PATH_REGISTRY[ROBOT_NAME].replace("_fk_cmap", "_ik_cmap_mp")

    with open(discr_config_path, "r") as f:
        discr_config_dict = json.load(f)
    discretizer = WsDiscretizer.from_config(discr_config_dict)

    cmap_shape = (discretizer.n_c, discretizer.n_c, discretizer.n_c, discretizer.n_p, discretizer.m_o)
    total_elements = int(np.prod(cmap_shape))

    # 使用无锁共享内存
    shared_array_base = mp.RawArray(ctypes.c_bool, total_elements)

    # ==================== 分配多进程任务 =============
    num_processes = max(1, mp.cpu_count() - 2)
    samples_per_worker = total_samples // num_processes

    print(f"启动 {num_processes} 个进程，总采样: {total_samples} [{ROBOT_NAME}]")

    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=worker_task_ik,
            args=(
                i,
                samples_per_worker,
                robot_config_dict,
                discr_config_dict,
                shared_array_base,
                cmap_shape,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # ========= 保存结果 ===============
    final_cmap = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, cmap=final_cmap, config=discr_config_dict)
    print(f"多进程 IK 能力图构建完成并保存至: {save_path}")
