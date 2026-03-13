import numpy as np
import pinocchio as pin
import example_robot_data as erd
import hppfcl
import time
import os
import json
from tqdm import tqdm
from ws_discretizer import WsDiscretizer

class CmapBuilderFK:
    def __init__(self, discretizer, robot_name='ur5', tcp_frame_name=None):
        self.discretizer = discretizer
        self.robot_name = robot_name
        
        # 1. 加载机器人与碰撞模型
        print(f"正在加载 {robot_name} 模型...")
        self.robot = erd.load(robot_name)
        self.model = self.robot.model
        self.data = self.robot.data
        self.collision_model = self.robot.collision_model
        
        # 添加基座碰撞体，避免机械臂穿模基座
        pedestal_geom = pin.GeometryObject(
            "pedestal", 0, 
            pin.SE3(np.eye(3), np.array([0, 0, -0.4])), # 向下偏移
            hppfcl.Cylinder(0.15, 0.2)
        )
        pedestal_id = self.collision_model.addGeometryObject(pedestal_geom)
        for i in range(len(self.collision_model.geometryObjects)):
            if i != pedestal_id:
                self.collision_model.addCollisionPair(pin.CollisionPair(i, pedestal_id))
        self.collision_data = self.collision_model.createData()

        # 获取 TCP Frame ID
        self.tcp_id = self.model.getFrameId(tcp_frame_name) if tcp_frame_name else self.model.nframes - 1
        
        # 预计算姿态匹配查找表，极大加速 FK 姿态到网格的映射
        print("正在预计算姿态匹配查找表...")
        self.R_table = np.zeros((self.discretizer.n_p, self.discretizer.m_o, 3, 3))
        for f in self.discretizer.discrete_frames:
            self.R_table[f['point_idx'], f['rot_idx']] = f['T'][:3, :3] #
            
        # 关节限位
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        
        # 设定 Halton 序列使用的质数底数 (UR5有6个关节，取前6个质数)
        self.bases = [2, 3, 5, 7, 11, 13][:self.model.nq]

    def _halton_sequence(self, index):
        """生成多维 Halton 序列，实现空间平滑二分遍历
        """
        res = np.zeros(self.model.nq)
        for j, b in enumerate(self.bases):
            f = 1.0 / b
            i = index
            val = 0.0
            while i > 0:
                val += f * (i % b)
                i = i // b
                f = f / b
            res[j] = val
        return res

    def get_nearest_pose_index(self, tcp_rot):
        """通过矩阵运算快速找到最接近的 p_idx 和 r_idx"""
        # 1. 匹配球面点 (Z轴对齐)
        Z_tcp = tcp_rot[:, 2]
        # discretizer中 z_axis = -p_i，所以求点积最大化
        dots = np.dot(self.discretizer.sphere_points, -Z_tcp)
        p_idx = np.argmax(dots)
        
        # 2. 匹配旋转 (寻找 trace 最大)
        R_p = self.R_table[p_idx] # shape: (m_o, 3, 3)
        # 批量计算 trace(R_p^T @ tcp_rot)
        traces = np.einsum('nij,ij->n', R_p, tcp_rot)
        r_idx = np.argmax(traces)
        
        return p_idx, r_idx

    def build(self, max_fk=10_000_000, save_path="./data/ur5_fk_cmap.npz"):
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 初始化或恢复数据
        cmap_shape = (self.discretizer.n_c, self.discretizer.n_c, self.discretizer.n_c, 
                      self.discretizer.n_p, self.discretizer.m_o)
        
        start_step = 1
        total_filled = 0
        
        if os.path.exists(save_path):
            print(f"发现已有保存文件 {save_path}，正在验证配置...")
            data = np.load(save_path, allow_pickle=True)
            
            # 1. 尝试提取文件中的配置信息
            if 'config' in data:
                # np.savez 保存字典时会将其包装成 0 维对象数组，需要用 .item() 提取
                saved_config = data['config'].item()
                current_config = self.discretizer.to_config()
                
                # 2. 校验配置是否严格匹配
                if saved_config == current_config:
                    self.cmap = data['cmap']
                    start_step = int(data.get('step', 1))
                    total_filled = np.sum(self.cmap)
                    print(f"配置校验通过，成功恢复！当前进度: Step={start_step}, 已填充区块={total_filled}")
                else:
                    print("\n[错误] 配置不匹配，拒绝恢复！")
                    print(f"存档中的配置: {saved_config}")
                    print(f"当前的配置:   {current_config}")
                    print("由于空间划分参数不同，无法基于此文件继续构建。请修改 save_path 路径，或删除旧文件后开启新任务。")
                    return  # 直接安全退出
            else:
                print("\n[错误] 该存档版本过旧，未包含 config 信息，为防止维度错乱，拒绝加载。")
                return
        else:
            print("启动全新构建任务...")
            self.cmap = np.zeros(cmap_shape, dtype=bool)

        print("\n开始正解采样 (按 Ctrl+C 可随时安全中断并保存)...\n")
        
        recent_fks = 0
        recent_filled = 0
        stat_window = 10000  # 每 10000 次 FK 更新一次统计信息
        
        pbar = tqdm(total=max_fk, initial=start_step, desc="FK Sampling")
        
        try:
            for step in range(start_step, max_fk + 1):
                # 1. 生成采样配置 (0~1之间) 并映射到关节空间
                halton_vals = self._halton_sequence(step)
                q = self.q_min + halton_vals * (self.q_max - self.q_min)
                
                # 2. 核心: 正向运动学求解
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacement(self.model, self.data, self.tcp_id)
                tcp_pose = self.data.oMf[self.tcp_id].translation
                tcp_rot = self.data.oMf[self.tcp_id].rotation
                
                # 3. 映射到体素
                g_idx = self.discretizer.get_voxel_index(tcp_pose)
                if g_idx is not None and np.all(g_idx >= 0) and np.all(g_idx < self.discretizer.n_c):
                    p_idx, r_idx = self.get_nearest_pose_index(tcp_rot)
                    
                    # 4. 查表拦截：如果该配置已经被填充过，直接跳过碰撞检测
                    if not self.cmap[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx]:
                        
                        # 5. 延迟碰撞检测：只有发现新区块才计算碰撞
                        pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, False)
                        has_collision = any(result.isCollision() for result in self.collision_data.collisionResults)
                        
                        if not has_collision:
                            self.cmap[g_idx[0], g_idx[1], g_idx[2], p_idx, r_idx] = True
                            total_filled += 1
                            recent_filled += 1

                recent_fks += 1
                pbar.update(1)
                
                # 周期性更新统计信息
                if step % stat_window == 0:
                    cost = (recent_fks / recent_filled) if recent_filled > 0 else float('inf')
                    filled_percent = (total_filled / self.discretizer.total_discrete_frames) * 100.0
                    
                    pbar.set_postfix({
                        # 建议保留 4 位小数，因为区块总数通常很大，这样能清楚看到进度在涨
                        "Filled": f"{filled_percent:.4f}%", 
                        "Recent Cost": f"{cost:.2f} FK/区块"
                    })
                    recent_fks = 0
                    recent_filled = 0

        except KeyboardInterrupt:
            print("\n\n收到中断指令，正在保存当前进度...")
        finally:
            pbar.close()
            # 退出时保存文件
            config = self.discretizer.to_config() #
            np.savez_compressed(save_path, cmap=self.cmap, config=config, step=step)
            print(f"数据已安全保存至: {save_path} (进度: Step={step})")

if __name__ == '__main__':
    # 使用你提供的配置结构加载离散化器
    import json
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), "discretizer_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    discretizer = WsDiscretizer.from_config(config) #
    
    builder = CmapBuilderFK(discretizer, robot_name='ur5')
    
    # max_fk 探测的最大正解次数。
    builder.build(max_fk=100_000_000, save_path="./data/ur5_fk_cmap.npz")