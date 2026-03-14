import os
import json
import time
import ctypes
import signal
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path

# Adjust imports based on your exact directory structure
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from ws_discretizer import WsDiscretizer
from csm.model import CSM

def get_halton_sequence(index, bases):
    """Independent Halton sequence generator"""
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

def map_halton_to_csm(csm, mode, halton_vals):
    """
    Maps 0-1 Halton sequence values to the appropriate CSM configuration bounds
    based on the current operating mode.
    """
    csm.mode = mode
    csm.phi = halton_vals[0] * 2 * np.pi
    
    if mode == 1:
        csm.L2 = halton_vals[1] * csm.L_20
        csm.theta_2 = halton_vals[2] * (csm.kappa_20 * csm.L2)
        csm.delta_2 = halton_vals[3] * 2 * np.pi
    elif mode == 2:
        csm.Lr = halton_vals[1] * csm.L_r0
        csm.L2 = csm.L_20
        csm.theta_2 = halton_vals[2] * (csm.kappa_20 * csm.L2)
        csm.delta_2 = halton_vals[3] * 2 * np.pi
    elif mode == 3:
        csm.L1 = halton_vals[1] * csm.L_10
        csm.theta_1 = halton_vals[2] * (csm.kappa_10 * csm.L1)
        csm.delta_1 = halton_vals[3] * 2 * np.pi
        csm.L2 = csm.L_20
        csm.Lr = csm.L_r0
        csm.theta_2 = halton_vals[4] * (csm.kappa_20 * csm.L2)
        csm.delta_2 = halton_vals[5] * 2 * np.pi
    elif mode == 4:
        csm.Ls = halton_vals[1] * csm.L_s0
        csm.L1 = csm.L_10
        csm.theta_1 = halton_vals[2] * (csm.kappa_10 * csm.L1)
        csm.delta_1 = halton_vals[3] * 2 * np.pi
        csm.L2 = csm.L_20
        csm.Lr = csm.L_r0
        csm.theta_2 = halton_vals[4] * (csm.kappa_20 * csm.L2)
        csm.delta_2 = halton_vals[5] * 2 * np.pi

def worker_task_fk(worker_id, start_step, max_fk, num_processes, valid_modes, 
                   discr_config_dict, csm_config_path, shared_array_base, cmap_shape, 
                   shared_steps_counter, stop_event):
    """
    Worker process: Parallel interleaved FK sampling for the CSM robot
    """
    # Let the main process handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 1. Restore the discretizer and shared memory array
    discretizer = WsDiscretizer.from_config(discr_config_dict)
    cmap_shared = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)
    
    # 2. Initialize the CSM model independently for this worker
    csm = CSM.from_config(csm_config_path)
    
    # We need up to 6 parameters for mode 3 & 4
    bases = [2, 3, 5, 7, 11, 13]
    num_modes = len(valid_modes)
    
    # Pre-build Discretizer rotation table for fast alignment
    R_table = np.zeros((discretizer.n_p, discretizer.m_o, 3, 3))
    for f in discretizer.discrete_frames:
        R_table[f['point_idx'], f['rot_idx']] = f['T'][:3, :3]
        
    # 3. Interleaved execution loop
    for step in range(start_step + worker_id, max_fk + 1, num_processes):
        if stop_event.is_set():
            break
            
        # Determine the mode cyclically and generate Halton sequence 
        mode = valid_modes[step % num_modes]
        halton_vals = get_halton_sequence(step // num_modes, bases)
        
        # Apply configurations to CSM and calculate Forward Kinematics
        map_halton_to_csm(csm, mode, halton_vals)
        csm.update()
        
        # Extract End-effector Pose
        tcp_pose = csm.pose[:3]
        tcp_rot = csm.rotation_matrix
        
        # Update Cmap
        g_idx = discretizer.get_voxel_index(tcp_pose)
        if g_idx is not None and np.all(g_idx >= 0) and np.all(g_idx < discretizer.n_c):
            Z_tcp = tcp_rot[:, 2]
            p_idx = np.argmax(np.dot(discretizer.sphere_points, -Z_tcp))
            r_idx = np.argmax(np.einsum('nij,ij->n', R_table[p_idx], tcp_rot))
            
            # Since we are just mapping workspace reachability, we set it to True directly 
            # (No collision models required for standard CSM reachability)
            if not cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx]:
                cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx] = True

        # Lock-free progress increment
        shared_steps_counter[worker_id] += 1

if __name__ == '__main__':
    # ========================== Parameters ==========================
    save_path = "./data/csm_fk_cmap_multi.npz"
    max_fk = 10_000_000  # Adjust as needed
    USE_RICH = True

    # Load Discretizer Config
    discr_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/discr_cfg_csm.json"))
    with open(discr_config_path, "r") as f:
        discr_config_dict = json.load(f)
    discretizer = WsDiscretizer.from_config(discr_config_dict)
    
    # Check CSM Config to determine available modes
    csm_config_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/csm_cfg_3.4mm.yaml")))
    csm_temp = CSM.from_config(csm_config_path)
    
    valid_modes = [1, 2, 3]
    if csm_temp.L_s0 > 0:
        valid_modes.append(4)
        
    print(f"Loaded CSM Model. Detected valid structural modes: {valid_modes}")
    
    cmap_shape = (discretizer.n_c, discretizer.n_c, discretizer.n_c, discretizer.n_p, discretizer.m_o)
    total_blocks = int(np.prod(cmap_shape))
    
    shared_array_base = mp.RawArray(ctypes.c_bool, total_blocks)
    cmap_view = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)
    
    start_step = 1
    total_filled = 0
    
    # ========================== Resume logic ==========================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"Found existing file {save_path}, verifying configurations...")
        data = np.load(save_path, allow_pickle=True)
        if 'config' in data and data['config'].item() == discr_config_dict:
            np.copyto(cmap_view, data['cmap'])
            start_step = int(data.get('step', 1))
            total_filled = np.sum(cmap_view)
            print(f"Validation passed! Resuming: Step={start_step}, Filled: {(total_filled/total_blocks)*100:.4f}%")
        else:
            print("[Error] Config mismatch. Aborting.")
            exit()
    else:
        print("Starting fresh CSM Cmap build...")
        cmap_view.fill(False)
        
    if start_step >= max_fk:
        print("Required steps already completed. Exiting.")
        exit()

    # ========================== Process Management ==========================
    num_processes = max(1, mp.cpu_count() - 2)
    print(f"\nStarting parallel CSM FK sampling (Workers: {num_processes}). Press Ctrl+C to safely interrupt & save...\n")

    shared_steps_counter = mp.RawArray('i', num_processes)
    stop_event = mp.Event()
    processes = []
    
    for i in range(num_processes):
        p = mp.Process(target=worker_task_fk, args=(
            i, start_step, max_fk, num_processes, valid_modes, discr_config_dict,
            csm_config_path, shared_array_base, cmap_shape, shared_steps_counter, stop_event
        ))
        processes.append(p)
        p.start()
        
    # ========================== Main Thread Monitoring ==========================
    try:
        last_total_steps = 0
        last_filled = total_filled
        
        if USE_RICH:
            # ---------------- 模式 A: Rich 多行炫酷模式 ----------------
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                
                main_task = progress.add_task("[bold green]Total Progress", total=max_fk, completed=start_step)
                
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
                        description=f"[bold green]Total Progress[/] [yellow](Filled: {filled_percent:.4f}% | Cost: {cost:.2f} FK/Voxel)[/]"
                    )
                    
                    for i in range(num_processes):
                        progress.update(worker_tasks[i], completed=shared_steps_counter[i])
                    
                    last_total_steps = current_total_steps
                    last_filled = current_filled

        else:
            # ---------------- 模式 B: Tqdm 经典单行模式 ----------------
            pbar = tqdm(total=max_fk, initial=start_step, desc="Multiprocess CSM FK")
            
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
                    "Cost": f"{cost:.2f} FK/Voxel"
                })
                
                last_total_steps = current_total_steps
                last_filled = current_filled

    except KeyboardInterrupt:
        print("\n\nInterrupt received. Signaling workers to terminate gracefully (Please wait 1-2s)...")
        stop_event.set()
        for p in processes:
            p.join()
            
    finally:
        # 如果使用的是 tqdm，需要手动关闭
        if not USE_RICH and 'pbar' in locals():
            pbar.close()
            
        min_steps_done = min(shared_steps_counter)
        safe_step_to_save = start_step + min_steps_done * num_processes
        
        print(f"\nSaving data securely (Checkpoint anchor: Step={safe_step_to_save})...")
        np.savez_compressed(save_path, cmap=cmap_view, config=discr_config_dict, step=safe_step_to_save)
        print("Save complete. Run script again to seamlessly resume.")