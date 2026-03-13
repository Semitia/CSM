import numpy as np
import pinocchio as pin
import example_robot_data as erd
import multiprocessing as mp
import ctypes
from tqdm import tqdm
from ws_discretizer import WsDiscretizer

def worker_task(worker_id, num_samples, discretizer, robot_name, tcp_frame_name, shared_array_base, cmap_shape):
    import numpy as np
    import pinocchio as pin
    import example_robot_data as erd
    import hppfcl
    from tqdm import tqdm

    # 1. 初始化视图与模型
    cmap_shared = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)
    robot = erd.load(robot_name)
    model = robot.model
    collision_model = robot.collision_model
    data = robot.data

    # --- 关键：手动添加基座碰撞体（避免机器人“穿模”基座） ---
    # 定义一个圆柱体作为基座 (半径15cm, 高80cm)，位置在机器人 Base 下方
    pedestal_geom = pin.GeometryObject(
        "pedestal", 0, 
        pin.SE3(np.eye(3), np.array([0, 0, -0.4])), # 向下偏移一半高度
        hppfcl.Cylinder(0.15, 0.2)
    )
    pedestal_geom.meshColor = np.array([0.5, 0.5, 0.5, 1.0]) # 灰色
    
    # 正确的做法：获取基座ID，并仅将基座与现有的其他连杆建立碰撞对
    pedestal_id = collision_model.addGeometryObject(pedestal_geom)
    for i in range(len(collision_model.geometryObjects)):
        if i != pedestal_id:
            collision_model.addCollisionPair(pin.CollisionPair(i, pedestal_id))
            
    # 在所有修改完成后，再创建碰撞数据对象
    collision_data = collision_model.createData()

    # 获取 TCP ID
    tcp_id = model.getFrameId(tcp_frame_name) if tcp_frame_name else model.nframes - 1
    
    q_min = model.lowerPositionLimit
    q_max = model.upperPositionLimit

    iterator = range(num_samples)
    if worker_id == 0:
        iterator = tqdm(iterator, desc="Worker 0 [IK + Collision]", position=0)

    for _ in iterator:
        # 1. 采样初始位姿 (确保在关节限位内)
        q_r = np.random.uniform(q_min, q_max)
        
        # 2. 得到采样点的体素索引
        pin.forwardKinematics(model, data, q_r)
        pin.updateFramePlacement(model, data, tcp_id)
        tcp_pose = data.oMf[tcp_id].translation
        
        g_idx = discretizer.get_voxel_index(tcp_pose)
        if g_idx is None or np.any(g_idx < 0) or np.any(g_idx >= discretizer.n_c):
            continue
            
        voxel_center = discretizer.get_voxel_center(g_idx)
        
        for frame_dict in discretizer.discrete_frames:
            p_idx, r_idx = frame_dict['point_idx'], frame_dict['rot_idx']
            
            # 无锁快速检查
            if cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx]:
                continue
            
            # 组装目标位姿
            T_target = frame_dict['T'].copy()
            T_target[:3, 3] = voxel_center
            oMdes = pin.SE3(T_target)
            
            # 3. 求解 IK (已包含关节限位校验)
            is_reachable, q_sol = solve_ik_robust(model, data, tcp_id, oMdes, q_init=q_r)
            
            if is_reachable:
                # 4. 【核心】物理碰撞检测 (自碰撞 + 基座碰撞)
                pin.computeCollisions(model, data, collision_model, collision_data, q_sol, False)
                
                # 检查是否有任何碰撞对发生接触
                has_collision = False
                for result in collision_data.collisionResults:
                    if result.isCollision():
                        has_collision = True
                        break
                
                if not has_collision:
                    cmap_shared[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx] = True


def solve_ik_robust(model, data, tcp_id, oMdes, q_init, eps=1e-4, IT_MAX=1000, DT=1e-1, damp=1e-12):
    """
    独立版 Pinocchio 逆向运动学求解器（带关节限位校验）
    参数中显式传入了当前进程独占的 model, data 和 tcp_id
    """
    import pinocchio as pin
    import numpy as np
    
    q = q_init.copy()
    success = False
    
    q_min = model.lowerPositionLimit
    q_max = model.upperPositionLimit
    
    for i in range(IT_MAX):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, tcp_id)
        
        dMi = oMdes.actInv(data.oMf[tcp_id])
        err = pin.log(dMi).vector
        
        if np.linalg.norm(err) < eps:
            q_normalized = pin.normalize(model, q)
            if np.all(q_normalized >= q_min) and np.all(q_normalized <= q_max):
                success = True
            break
            
        J = pin.computeFrameJacobian(model, data, q, tcp_id, pin.ReferenceFrame.LOCAL)
        v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model, q, v * DT)
        
    return success, q

# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    import json
    import os
    
    # 1. 初始化你的离散化器
    # 读取配置文件
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/discr_cfg_ur5.json"))
    with open(config_path, "r") as f:
        config = json.load(f)
        
    discretizer = WsDiscretizer.from_config(config)
    
    # 2. 计算需要开辟的共享内存大小
    # 假设 cmap 的 shape 是 (n_c, n_c, n_c, n_p, m_o)
    cmap_shape = (discretizer.n_c, discretizer.n_c, discretizer.n_c, discretizer.n_p, discretizer.m_o)
    total_elements = int(np.prod(cmap_shape))
    
    # 使用 ctypes.c_bool 开辟一块无锁的共享连续内存 (RawArray)
    shared_array_base = mp.RawArray(ctypes.c_bool, total_elements)
    
    # 3. 分配多进程任务
    total_samples = 1_00
    num_processes = mp.cpu_count() - 2  # 留2个核心给操作系统
    samples_per_worker = total_samples // num_processes
    
    print(f"启动 {num_processes} 个进程，总采样: {total_samples}")
    
    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=worker_task, 
            args=(i, samples_per_worker, discretizer, 'ur5', None, shared_array_base, cmap_shape)
        )
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    # 4. 所有进程结束后，主进程中获取最终结果并保存
    final_cmap = np.ctypeslib.as_array(shared_array_base).reshape(cmap_shape)
    
    # 保存 cmap 数据和 discretizer 配置
    config = discretizer.to_config()
    np.savez_compressed("./data/ur5_capability_map.npz", cmap=final_cmap, config=config)
    print("多进程能力图构建完成并落盘！")