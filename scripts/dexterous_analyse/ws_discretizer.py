import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class WsDiscretizer:
    """
    负责机器人工作空间位置(R^3)和姿态(SO(3))的离散化。
    """
    def __init__(self, arm_length, l_c=0.05, n_p=200, delta_o=30.0):
        # 1. 初始化基本参数
        # 边界盒边长 l_ws 为机械臂长度的两倍 [cite: 193]
        self.l_ws = 2.0 * arm_length 
        # 体素边长 l_c，通常设为 0.05m (50mm) [cite: 198]
        self.l_c = float(l_c)
        # 沿每个维度的体素数量 n_c [cite: 209]
        self.n_c = math.ceil(self.l_ws / self.l_c)
        
        # 姿态离散化参数
        self.n_p = n_p  # 球面均匀分布的点数 [cite: 236]
        self.delta_o = delta_o  # 绕 Z 轴的旋转步长 (度) [cite: 248]
        # 绕 Z 轴的离散方向数量 m_o [cite: 253]
        self.m_o = math.floor(360.0 / self.delta_o)
        
        # 2. 预计算离散化姿态集 O_s [cite: 281]
        self.sphere_points = self._generate_spiral_points()
        self.discrete_frames = self._generate_discrete_frames()
        
        # 3. 计算总离散化体素数 N_s 
        self.total_discrete_frames = self.n_c**3 * self.n_p * self.m_o

    def to_config(self):
        """
        导出当前实例的配置参数
        """
        return {
            "arm_length": self.l_ws / 2.0,
            "l_c": self.l_c,
            "n_p": self.n_p,
            "delta_o": self.delta_o
        }

    @classmethod
    def from_config(cls, config):
        """
        从配置字典创建实例
        """
        return cls(
            arm_length=config["arm_length"],
            l_c=config["l_c"],
            n_p=config["n_p"],
            delta_o=config["delta_o"]
        )

    def get_voxel_index(self, t):
        """
        映射函数 v(t)：将笛卡尔坐标 t (x, y, z) 映射到体素网格坐标 g [cite: 225]。
        """
        t = np.asarray(t)
        # 检查是否在包围盒外
        if np.any(np.abs(t) > self.l_ws / 2.0):
            return None 
            
        # 根据公式 13 计算网格坐标 [cite: 225]
        # 注意：论文中使用向上取整 ceiling operator [cite: 211]
        g = np.ceil(t / self.l_c) + (self.n_c / 2.0 - 1.0)
        return g.astype(int)

    def get_voxel_center(self, g):
        """
        映射函数 w(g)：将体素网格坐标 g 映射回其在笛卡尔空间中的中心坐标 t [cite: 227]。
        """
        g = np.asarray(g)
        # 根据公式 14 计算体素中心 [cite: 228]
        t = (g + (1.0 - self.n_c / 2.0)) * self.l_c - (self.l_c / 2.0)
        return t

    def _generate_spiral_points(self):
        """
        使用 Saff 等人的螺旋点算法生成球面上均匀分布的点 [cite: 236]。
        """
        points = np.zeros((self.n_p, 3))
        phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
        
        for i in range(self.n_p):
            y = 1.0 - (i / float(self.n_p - 1)) * 2.0 # y 从 1 到 -1
            radius = np.sqrt(1.0 - y * y) # 在当前 y 高度的圆半径
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
            
        return points

    def _generate_discrete_frames(self):
        """
        生成体素内所有的离散姿态帧 (n_p * m_o 个矩阵) [cite: 280]。
        """
        frames = []
        
        for i, p_i in enumerate(self.sphere_points):
            # 1. 构造基础旋转矩阵 R_{i,0} 
            # -p_i 作为 z 轴 
            z_axis = -p_i
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # 选择一个任意向量来计算 x 轴，确保它们正交
            up = np.array([0, 0, 1])
            if np.abs(np.dot(z_axis, up)) > 0.99:
                up = np.array([1, 0, 0])
                
            x_axis = np.cross(up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis) # 构成右手坐标系 
            
            R_i0 = np.column_stack((x_axis, y_axis, z_axis))
            
            # 2. 绕 z 轴旋转 [cite: 248]
            for k in range(self.m_o):
                alpha_k = np.deg2rad(k * self.delta_o)
                # 构建绕 z 轴旋转的矩阵
                rot_z = R.from_euler('z', alpha_k).as_matrix()
                
                # 得到最终的旋转矩阵 Rot(i, k) [cite: 274]
                R_ik = R_i0 @ rot_z
                
                # 构建 4x4 齐次矩阵 (仅包含旋转，平移后续在具体体素中结合)
                T_ik = np.eye(4)
                T_ik[:3, :3] = R_ik
                
                frames.append({
                    'point_idx': i,
                    'rot_idx': k,
                    'T': T_ik
                })
                
        return frames

# --- 测试用例 ---
if __name__ == "__main__":
    TEST_CONFIG = {
        "arm_length": 0.85,                 # 机械臂最大伸展长度 (m)，例如 UR5 约为 0.85m
        "voxel_size": 0.05,                 # 体素边长 (m)，通常设为 0.05m (50mm)
        "sphere_points": 200,               # 球面离散点数，用于生成均匀分布的方向向量
        "rotation_step": 30.0,              # 绕 Z 轴旋转的步长 (度)，决定了绕每个方向轴的旋转分辨率
        "test_point": [0.21, -0.34, 0.56]   # 用于测试空间映射功能的测试点坐标 [x, y, z] (m)
    }
    
    # 使用配置参数初始化离散化器
    discretizer = WsDiscretizer(
        arm_length=TEST_CONFIG["arm_length"], 
        l_c=TEST_CONFIG["voxel_size"], 
        n_p=TEST_CONFIG["sphere_points"], 
        delta_o=TEST_CONFIG["rotation_step"]
    )
    
    print(f"总计算体素维度: {discretizer.n_c} x {discretizer.n_c} x {discretizer.n_c}")
    print(f"每个体素内的姿态总数: {len(discretizer.discrete_frames)} ({discretizer.n_p} 个点 * {discretizer.m_o} 个旋转方向)")
    
    # 测试空间映射
    test_pt = TEST_CONFIG["test_point"]
    g_idx = discretizer.get_voxel_index(test_pt)
    center_pt = discretizer.get_voxel_center(g_idx)
    
    print(f"\n测试点位置: {test_pt}")
    print(f"映射到体素索引: {g_idx}")
    print(f"体素中心点位置: {center_pt}")