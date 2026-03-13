import numpy as np
import pinocchio as pin
import example_robot_data as erd
from tqdm import tqdm  # 用于显示进度条
from pathlib import Path
from ws_discretizer import WsDiscretizer
import os

class CmapBuilderIK:
    def __init__(self, discretizer, robot_name='ur5', tcp_frame_name=None):
        """
        初始化能力图构建器。
        :param discretizer: WsDiscretizer 实例
        :param robot_name: example-robot-data 中的机器人名称
        :param tcp_frame_name: 末端执行器 Frame 名称。若为 None，默认取最后一个 Frame
        """
        self.discretizer = discretizer
        
        # 1. 加载 Pinocchio 模型
        print(f"正在加载 {robot_name} 模型...")
        self.robot = erd.load(robot_name)
        self.model = self.robot.model
        self.data = self.robot.data
        
        # 获取 TCP Frame ID
        if tcp_frame_name is None:
            self.tcp_id = self.model.nframes - 1
        else:
            self.tcp_id = self.model.getFrameId(tcp_frame_name)
            
        print(f"TCP Frame 设置为: {self.model.frames[self.tcp_id].name}")
        
        # 2. 初始化 5D 布尔数组用于存储能力图 (n_c, n_c, n_c, n_p, m_o)
        n_c = self.discretizer.n_c
        n_p = self.discretizer.n_p
        m_o = self.discretizer.m_o
        self.cmap = np.zeros((n_c, n_c, n_c, n_p, m_o), dtype=bool)

    def _solve_ik(self, oMdes, q_init, eps=1e-4, IT_MAX=1000, DT=1e-1, damp=1e-12):
        """
        基于 Pinocchio 雅可比矩阵的 Gauss-Newton 数值逆运动学求解器。
        :param oMdes: pin.SE3, 目标齐次变换矩阵
        :param q_init: np.array, 迭代的初始关节种子
        """
        q = q_init.copy()
        success = False
        
        for i in range(IT_MAX):
            # 前向运动学并更新 Frame
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.tcp_id)
            dMi = oMdes.actInv(self.data.oMf[self.tcp_id])
            err = pin.log(dMi).vector
            
            # 如果误差在容差范围内，则判定为收敛
            if np.linalg.norm(err) < eps:
                success = True
                break
                
            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp_id, pin.ReferenceFrame.LOCAL)
            
            # 阻尼最小二乘法 (Damped Pseudo-Inverse) 更新 q
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            
        return success, q

    def build(self, num_samples=100_000):
        """
        执行蒙特卡洛采样并构建能力图 (对应论文 Algorithm 1)。
        """
        print(f"开始构建能力图，计划采样次数: {num_samples}...")
        
        # 获取关节限制
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit
        
        for _ in tqdm(range(num_samples), desc="Sampling Workspace"):
            # 1. 在关节限制内随机采样 q_r
            q_r = np.random.uniform(q_min, q_max)
            
            # 2. 计算当前随机配置下的 TCP 位姿
            pin.forwardKinematics(self.model, self.data, q_r)
            pin.updateFramePlacement(self.model, self.data, self.tcp_id)
            tcp_pose = self.data.oMf[self.tcp_id].translation
            
            # 3. 将 TCP 笛卡尔坐标映射到体素索引
            g_idx = self.discretizer.get_voxel_index(tcp_pose)
            
            # 如果超出了我们定义的边界盒，直接跳过
            if g_idx is None or np.any(g_idx < 0) or np.any(g_idx >= self.discretizer.n_c):
                continue
                
            # 获取该体素的中心坐标
            voxel_center = self.discretizer.get_voxel_center(g_idx)
            
            # 4. 遍历该体素内的所有离散目标姿态
            for frame_dict in self.discretizer.discrete_frames:
                point_idx = frame_dict['point_idx']
                rot_idx = frame_dict['rot_idx']
                
                # 如果该姿态已经被标记为可达，跳过以节省计算资源
                if self.cmap[g_idx[0], g_idx[1], g_idx[2], point_idx, rot_idx]:
                    continue
                
                # 构建目标齐次矩阵 T_{Sphere}^{Base}(g) * F_{i, \alpha_k}
                T_target_mat = frame_dict['T'].copy()
                T_target_mat[:3, 3] = voxel_center
                
                # 转换为 Pinocchio SE3 对象
                oMdes = pin.SE3(T_target_mat)
                
                # 5. 求解 IK，将 q_r 作为初始种子探索局部空间
                is_reachable, _ = self._solve_ik(oMdes, q_init=q_r)
                
                # 6. 如果可达，更新布尔数组
                if is_reachable:
                    self.cmap[g_idx[0], g_idx[1], g_idx[2], point_idx, rot_idx] = True

        print("能力图构建完成！")

    def save(self, filepath):
        """
        将构建好的能力图保存到本地硬盘。
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 保存 cmap 数据和 discretizer 配置
        config = self.discretizer.to_config()
        np.savez_compressed(filepath, cmap=self.cmap, config=config)
        print(f"数据已成功保存至: {filepath}")

# --- 测试与运行逻辑 ---
if __name__ == "__main__":
    import json
    
    # 读取配置文件
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/discr_cfg_ur5.json"))
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 使用配置参数初始化离散化器
    discretizer = WsDiscretizer.from_config(config)
    
    # 初始化构建器 (使用默认的 UR5)
    builder = CmapBuilderIK(discretizer, robot_name='ur5')
    
    # 运行算法 (为了快速测试，这里设置 100 次。论文标准为 1,000,000 次)
    builder.build(num_samples=100)
    
    # 保存结果到个人文件盘符，方便后续读取分析
    save_path = "./data/ur5_capability_map.npz"
    builder.save(save_path)