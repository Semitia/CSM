import numpy as np
import pinocchio as pin
import example_robot_data as erd
import hppfcl
import time
import os
import json
import multiprocessing as mp
import ctypes
import signal  
from tqdm import tqdm
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from ws_discretizer import WsDiscretizer

def get_halton_sequence(index, bases):
    """独立的 Halton 序列生成器"""
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

def worker_task_fk(worker_id, start_step, max_fk, num_processes, config_dict, 
                   robot_name, tcp_frame_name, shared_array_base, cmap_shape, 
                   shared_steps_counter, stop_event):
    """
    工作进程任务：利用交织步长并行计算
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 1. 恢复离散化器
    discretizer = WsDiscretizer.from_config(config_dict)
    cmap_shared = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)
    
    # 2. 重新加载机器人与碰撞模型
    robot = erd.load(robot_name)
    model = robot.model
    data = robot.data
    collision_model = robot.collision_model
    
    pedestal_height = 0.2
    pedestal_radius = 0.15
    pedestal_geom = pin.GeometryObject(
        "pedestal", 0, 
        pin.SE3(np.eye(3), np.array([0, 0, -pedestal_height / 2])),
        hppfcl.Cylinder(pedestal_radius, pedestal_height)
    )
    pedestal_id = collision_model.addGeometryObject(pedestal_geom)
    
    for i in range(len(collision_model.geometryObjects)):
        if i == pedestal_id:
            continue
        geom_obj = collision_model.geometryObjects[i]
        if geom_obj.parentJoint > 1: 
            collision_model.addCollisionPair(pin.CollisionPair(i, pedestal_id))
            
    collision_data = collision_model.createData()
    
    # 【核心修改 1】：Panda 的默认末端名字通常是 panda_link8，比 nframes - 1 更准确
    tcp_id = model.getFrameId(tcp_frame_name) if tcp_frame_name else model.nframes - 1
    q_min, q_max = model.lowerPositionLimit, model.upperPositionLimit
    
    # 【核心修改 2】：Panda 是 7 轴 (nq=7)，必须补齐至少 7 个质数，这里多给几个防爆
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:model.nq] 
    
    R_table = np.zeros((discretizer.n_p, discretizer.m_o, 3, 3))
    for f in discretizer.discrete_frames:
        R_table[f['point_idx'], f['rot_idx']] = f['T'][:3, :3]
        
    # 3. 开始执行交织循环
    for step in range(start_step + worker_id, max_fk + 1, num_processes):
        if stop_event.is_set():
            break
            
        halton_vals = get_halton_sequence(step, bases)
        q = q_min + halton_vals * (q_max - q_min)
        
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, tcp_id)
        tcp_pose = data.oMf[tcp_id].translation
        tcp_rot = data.oMf[tcp_id].rotation
        
        g_idx = discretizer.get_voxel_index(tcp_pose)
        if g_idx is not None and np.all(g_idx >= 0) and np.all(g_idx < discretizer.n_c):
            Z_tcp = tcp_rot[:, 2]
            p_idx = np.argmax(np.dot(discretizer.sphere_points, -Z_tcp))
            r_idx = np.argmax(np.einsum('nij,ij->n', R_table[p_idx], tcp_rot))
            
            if not cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx]:
                pin.computeCollisions(model, data, collision_model, collision_data, q, False)
                if not any(result.isCollision() for result in collision_data.collisionResults):
                    cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx] = True

        shared_steps_counter[worker_id] += 1

if __name__ == '__main__':
    # ========================== 参数配置 ==========================
    # 【核心修改 3】：更改机器人型号为 Panda，并修改保存路径
    robot_name = 'panda'
    save_path = "./data/panda_fk_cmap_multi.npz"
    max_fk = 200_000_000
    USE_RICH = True 

    # 【核心修改 4】：加载离散化器配置（你可以直接用 ur5 的 config，或者复制一份改名）
    # 这里我默认你复制了一份命名为 discr_cfg_panda.json，如果没改请把名字换回 ur5
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/discr_cfg_panda.json"))
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    discretizer = WsDiscretizer.from_config(config_dict)
    
    cmap_shape = (discretizer.n_c, discretizer.n_c, discretizer.n_c, discretizer.n_p, discretizer.m_o)
    total_blocks = int(np.prod(cmap_shape))
    
    shared_array_base = mp.RawArray(ctypes.c_bool, total_blocks)
    cmap_view = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)
    
    start_step = 1
    total_filled = 0
    
    # ========================== 断点续传逻辑 ==========================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"发现已有保存文件 {save_path}，正在验证配置...")
        data = np.load(save_path, allow_pickle=True)
        if 'config' in data and data['config'].item() == config_dict:
            np.copyto(cmap_view, data['cmap'])
            start_step = int(data.get('step', 1))
            total_filled = np.sum(cmap_view)
            print(f"配置校验通过！恢复进度: Step={start_step}, 已填充: {(total_filled/total_blocks)*100:.4f}%")
        else:
            print("[错误] 配置文件不匹配或缺失，程序已停止。")
            exit()
    else:
        print("启动全新构建任务...")
        cmap_view.fill(False)
        
    if start_step >= max_fk:
        print("设定步数已全部完成，无需继续。")
        exit()

    # ========================== 进程管理与调度 ==========================
    num_processes = mp.cpu_count() - 2
    num_processes = max(1, num_processes)
    print(f"\n开始并行正解采样 (进程数: {num_processes})，按 Ctrl+C 可安全中断并保存...\n")

    shared_steps_counter = mp.RawArray('i', num_processes)
    stop_event = mp.Event()
    processes = []
    
    for i in range(num_processes):
        # 【核心修改 5】：显式传入 Panda 的末端法兰名字 'panda_link8'
        p = mp.Process(target=worker_task_fk, args=(
            i, start_step, max_fk, num_processes, config_dict,
            robot_name, 'panda_link8', shared_array_base, cmap_shape,
            shared_steps_counter, stop_event
        ))
        processes.append(p)
        p.start()
        
    # ========================== 主进程监听与统计 ==========================
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
                    current_filled = np.sum(cmap_view)
                    filled_delta = current_filled - last_filled
                    
                    filled_percent = (current_filled / total_blocks) * 100.0
                    cost = (step_delta / filled_delta) if filled_delta > 0 else float('inf')
                    
                    progress.update(
                        main_task, 
                        completed=start_step + current_total_steps,
                        description=f"[bold green]总进度[/] [yellow](Filled: {filled_percent:.4f}% | Cost: {cost:.2f} FK/区块)[/]"
                    )
                    
                    for i in range(num_processes):
                        progress.update(worker_tasks[i], completed=shared_steps_counter[i])
                    
                    last_total_steps = current_total_steps
                    last_filled = current_filled

        else:
            pbar = tqdm(total=max_fk, initial=start_step, desc="Multiprocess FK")
            while any(p.is_alive() for p in processes):
                time.sleep(1.0)
                
                current_total_steps = sum(shared_steps_counter)
                step_delta = current_total_steps - last_total_steps
                
                current_filled = np.sum(cmap_view)
                filled_delta = current_filled - last_filled
                
                pbar.update(step_delta)
                
                filled_percent = (current_filled / total_blocks) * 100.0
                cost = (step_delta / filled_delta) if filled_delta > 0 else float('inf')
                
                pbar.set_postfix({
                    "Filled": f"{filled_percent:.4f}%", 
                    "Recent Cost": f"{cost:.2f} FK/区块"
                })
                
                last_total_steps = current_total_steps
                last_filled = current_filled

    except KeyboardInterrupt:
        print("\n\n收到中断指令，正在通知所有工作进程优雅退出 (请等待 1-2 秒)...")
        stop_event.set()
        for p in processes:
            p.join() 
            
    finally:
        if not USE_RICH and 'pbar' in locals():
            pbar.close()
            
        min_steps_done = min(shared_steps_counter)
        safe_step_to_save = start_step + min_steps_done * num_processes
        
        print(f"\n正在将数据落盘 (安全步数锚点: Step={safe_step_to_save})...")
        np.savez_compressed(save_path, cmap=cmap_view, config=config_dict, step=safe_step_to_save)
        print("保存完毕！下次启动将从该锚点无缝恢复。")