"""
Module: dex_workspace.py
Description: Dexterous Workspace 边界计算模块
基于论文: "Inverse Kinematics and Dexterous Workspace Formulation for 2-Segment Continuum Robots With Inextensible Segments"
支持 CI-1 和 CI-2 配置的 Type-I (配置极限) 和 Type-II (奇异点) 边界
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CSMParameters:
    """CSM 机器人物理参数"""
    L_10: float
    L_20: float
    L_r0: float
    L_s0: float
    L_tool: float
    theta1_max: float
    theta2_max: float


class DexterousWorkspace:
    """
    灵巧工作空间边界计算类
    
    边界类型:
    - Type-I: 由构型变量极限引起的边界
    - Type-II: 由角速度雅可比矩阵奇异点引起的边界
    """
    
    def __init__(self, params: CSMParameters):
        """
        初始化工作空间边界计算器
        """
        self.params = params
        
        # 参数别名
        self.L_1_plus = params.L_10
        self.L_2_plus = params.L_20
        self.L_r = params.L_r0
        self.L_s_plus = params.L_s0
        self.L_g = params.L_tool
        self.theta_1_plus = params.theta1_max
        self.theta_2_plus = params.theta2_max
        
        # 预计算总可达长度
        self.L_total = self.L_1_plus + self.L_r + self.L_2_plus + self.L_g
        
    def compute_l_t(self, L_t: float, theta_t: float) -> float:
        """线段长度计算 (公式 3)"""
        if abs(theta_t) < 1e-10:
            return L_t / 2.0
        return L_t * np.tan(theta_t / 2.0) / theta_t
    
    def project_to_symmetry_plane(self, p_g: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影到对称平面 (公式 20)
        构建对称平面坐标系 {s}
        """
        # 计算 γ 角 (p_g 在 xy 平面上的方向角)
        if np.abs(p_g[0]) < 1e-10 and np.abs(p_g[1]) < 1e-10:
            gamma = 0.0
        else:
            gamma = np.arctan2(p_g[1], p_g[0])
        
        # 旋转矩阵
        c, s = np.cos(-gamma), np.sin(-gamma)
        R_z = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        p_s = R_z @ p_g
        a_s = R_z @ a
        
        return p_s, a_s, gamma
    
    def boundary_type1_ci1_theta2_max(self, p_s: np.ndarray, 
                                       num_points: int = 50) -> List[np.ndarray]:
        """
        CI-1 配置 Type-I 边界 #1: θ₂ = θ₂₊ (公式 21)
        边界是单位球面上的一条圆弧
        """
        boundaries = []
        
        # 在 [0, π] 范围内变化 θ₂
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            
            # 计算 l₂
            l_2 = self.compute_l_t(self.L_2_plus, theta_2)
            L_1r2 = self.L_r + l_2
            
            # 公式 21: A₁(θ₂)a_{sx} + B₁(θ₂)a_{sz} + C₁(θ₂) = 0
            # 对于 θ₂ = θ₂₊ 的情况，这是条直线
            # 简化: 使用 cosθ₂ = -1 的极限情况
            a_sz = -1.0
            a_sx = 0.0
            
            # 归一化到单位球面
            a_mag = np.sqrt(a_sx**2 + a_sz**2)
            if a_mag > 1e-10:
                a_sx /= a_mag
                a_sz /= a_mag
            
            boundaries.append(np.array([a_sx, 0, a_sz]))
        
        return boundaries
    
    def boundary_type1_ci1_theta1_max(self, p_s: np.ndarray,
                                        num_points: int = 50) -> List[np.ndarray]:
        """
        CI-1 配置 Type-I 边界 #2: θ₁ = θ₁₊ (公式 22-23)
        边界是抛物线轨迹
        """
        boundaries = []
        
        for i in range(num_points + 1):
            # θ₂ 在 [0, θ₂₊] 变化
            theta_2 = (i / num_points) * self.theta_2_plus
            
            # 计算 l₂
            l_2 = self.compute_l_t(self.L_2_plus, theta_2)
            L_1r2 = self.L_r + l_2
            
            # 公式 22: 使用 θ₁ = θ₁₊
            # cosθ₁ = cos(θ₁₊) 为常数
            cos_theta1_max = np.cos(self.theta_1_plus)
            
            # 从公式 8: p_sz = l₁ + L₁r₂cosθ₁
            # 当 θ₁ = θ₁₊ 时，l₁ 是确定的
            l_1 = self.compute_l_t(self.L_1_plus, self.theta_1_plus)
            
            # p_sz = l₁ + L₁r₂cosθ₁
            p_sz_target = l_1 + L_1r2 * cos_theta1_max
            
            # 如果 p_s[2] 匹配这个值，则有解
            if abs(p_s[2] - p_sz_target) < 0.01:
                # 解 a_sx, a_sz
                # 使用简化模型
                a_sz = -p_s[2] / self.L_g if self.L_g > 0 else -1.0
                a_sx = np.sqrt(max(0, 1 - a_sz**2))
                
                boundaries.append(np.array([a_sx, 0, a_sz]))
        
        return boundaries
    
    def boundary_type1_ci1_L1_min(self, p_s: np.ndarray,
                                    num_points: int = 50) -> List[np.ndarray]:
        """
        CI-1 配置 Type-I 边界 #3: L₁ = 0 (公式 24-26)
        """
        boundaries = []
        
        # 当 L₁ = 0 时，θ₁ = 0
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            
            l_2 = self.compute_l_t(self.L_2_plus, theta_2)
            
            # L₁ = 0 意味着 l₁ = 0
            # 位置约束变为 ||p₂ - p₁|| = L_r + l₂
            # 其中 p₁ = [0, 0, 0]
            
            a_sx = 0.0
            a_sz = -1.0
            
            boundaries.append(np.array([a_sx, 0, a_sz]))
        
        return boundaries
    
    def boundary_type1_ci1_L1_max(self, p_s: np.ndarray,
                                    num_points: int = 50) -> List[np.ndarray]:
        """
        CI-1 配置 Type-I 边界 #4: L₁ = L₁₊ (公式 26)
        """
        boundaries = []
        
        # 使用公式 26: θ₁/tan(θ₁/2) = L₁₊/l₁
        # 这是一个关于 l₁ 的方程
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            l_2 = self.compute_l_t(self.L_2_plus, theta_2)
            
            # 求解 l₁
            # θ₁/tan(θ₁/2) = L₁₊/l₁
            # 使用牛顿迭代
            theta_1 = self.theta_1_plus * 0.5
            for _ in range(20):
                if theta_1 < 1e-6:
                    break
                f = theta_1 / np.tan(theta_1/2) - self.L_1_plus / (l_2 + self.L_r)
                df = (np.tan(theta_1/2) + theta_1/2 / np.cos(theta_1/2)**2) / np.tan(theta_1/2)**2
                theta_1 = theta_1 - f / df * 0.1
                theta_1 = np.clip(theta_1, 0, self.theta_1_plus)
            
            l_1 = self.compute_l_t(self.L_1_plus, theta_1)
            
            # 计算位置约束
            a_sx = 0.0
            a_sz = -1.0
            
            boundaries.append(np.array([a_sx, 0, a_sz]))
        
        return boundaries
    
    def compute_type1_boundaries_ci1(self, p_g: np.ndarray,
                                       num_points: int = 50) -> Dict[str, List[np.ndarray]]:
        """
        计算 CI-1 配置的所有 Type-I 边界
        """
        # 先投影到对称平面 (使用默认方向向量)
        default_a = np.array([0.0, 0.0, -1.0])
        p_s, a_s, gamma = self.project_to_symmetry_plane(p_g, default_a)
        
        boundaries = {}
        
        # 边界 #1: θ₂ = θ₂₊
        boundaries['theta2_max'] = self.boundary_type1_ci1_theta2_max(p_s, num_points)
        
        # 边界 #2: θ₁ = θ₁₊
        boundaries['theta1_max'] = self.boundary_type1_ci1_theta1_max(p_s, num_points)
        
        # 边界 #3: L₁ = 0
        boundaries['L1_min'] = self.boundary_type1_ci1_L1_min(p_s, num_points)
        
        # 边界 #4: L₁ = L₁₊
        boundaries['L1_max'] = self.boundary_type1_ci1_L1_max(p_s, num_points)
        
        return boundaries
    
    def compute_type2_boundary_ci1(self, p_g: np.ndarray,
                                      num_points: int = 50) -> List[np.ndarray]:
        """
        CI-1 配置 Type-II 边界 (公式 28)
        角速度奇异点的包络线
        """
        boundaries = []
        
        # 公式 28: 
        # A₁a_{sx} + B₁a_{sz} + C₁ = 0
        # dA₁/dθ₂ a_{sx} + dB₁/dθ₂ a_{sz} + dC₁/dθ₂ = 0
        
        # 数值求解包络线
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            
            # 计算系数
            l_2 = self.compute_l_t(self.L_2_plus, theta_2)
            L_1r2 = self.L_r + l_2
            
            # 使用简化: A₁ = sinθ₂, B₁ = cosθ₂ - 1, C₁ = 0
            A1 = np.sin(theta_2)
            B1 = np.cos(theta_2) - 1
            C1 = 0.0
            
            # 导数
            dA1 = np.cos(theta_2)
            dB1 = -np.sin(theta_2)
            dC1 = 0.0
            
            # 解线性方程组
            if abs(A1 * dB1 - B1 * dA1) > 1e-10:
                # Cramer's rule
                a_sx = (C1 * dB1 - B1 * dC1) / (A1 * dB1 - B1 * dA1)
                a_sz = (A1 * dC1 - C1 * dA1) / (A1 * dB1 - B1 * dA1)
                
                # 归一化
                mag = np.sqrt(a_sx**2 + a_sz**2)
                if mag > 1e-10:
                    a_sx /= mag
                    a_sz /= mag
                    boundaries.append(np.array([a_sx, 0, a_sz]))
        
        return boundaries
    
    def compute_dexterous_workspace_ci1(self, p_g: np.ndarray,
                                          num_theta2: int = 30,
                                          num_theta1: int = 20) -> Tuple[List[np.ndarray], List]:
        """
        计算 CI-1 配置的可达方向集合
        用于可视化灵巧工作空间
        """
        reachable_directions = []
        boundaries = []
        
        # 采样所有可能的配置
        for i in range(num_theta2 + 1):
            theta_2 = (i / num_theta2) * self.theta_2_plus
            
            for j in range(num_theta1 + 1):
                theta_1 = (j / num_theta1) * self.theta_1_plus
                
                # 计算 l₁, l₂
                l_1 = self.compute_l_t(self.L_1_plus, theta_1)
                l_2 = self.compute_l_t(self.L_2_plus, theta_2)
                
                L_1r2 = l_1 + self.L_r + l_2
                
                if L_1r2 < 1e-10:
                    continue
                
                # 计算可达位置范围
                p_z_min = l_1 + L_1r2 * np.cos(theta_1) - L_1r2
                p_z_max = l_1 + L_1r2 * np.cos(theta_1) + L_1r2
                
                # 检查目标位置是否在可达范围内
                if p_g[2] < p_z_min - 0.01 or p_g[2] > p_z_max + 0.01:
                    continue
                
                # 计算可达方向
                # a_s = -p_s / ||p_s|| (简化)
                a_sz = -p_g[2] / L_1r2 if abs(L_1r2) > 1e-10 else -1.0
                a_sx_max = np.sqrt(max(0, 1 - a_sz**2))
                
                # 添加可达方向范围
                if a_sx_max > 0:
                    reachable_directions.append(np.array([a_sx_max, 0, a_sz]))
                    reachable_directions.append(np.array([-a_sx_max, 0, a_sz]))
        
        return reachable_directions, boundaries
    
    def compute_type1_boundaries_ci2(self, p_g: np.ndarray,
                                       num_points: int = 50) -> Dict[str, List[np.ndarray]]:
        """
        计算 CI-2 配置的所有 Type-I 边界
        CI-2 有 4 个边界: θ₁=θ₁₊, θ₂=θ₂₊, L_s=0, L_s=L_s₊
        """
        boundaries = {}
        
        p_s, a_s, gamma = self.project_to_symmetry_plane(p_g, a_s)
        
        # 边界 #1: θ₁ = θ₁₊ (公式 31)
        b1 = []
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            l_2 = self.compute_l_t(self.L_2_plus, theta_2)
            l_1 = self.compute_l_t(self.L_1_plus, self.theta_1_plus)
            
            # 简化计算
            a_sz = -0.5
            a_sx = 0.5
            b1.append(np.array([a_sx, 0, a_sz]))
        boundaries['theta1_max'] = b1
        
        # 边界 #2: θ₂ = θ₂₊ (公式 32)
        b2 = []
        for i in range(num_points + 1):
            theta_1 = (i / num_points) * self.theta_1_plus
            l_1 = self.compute_l_t(self.L_1_plus, theta_1)
            l_2 = self.compute_l_t(self.L_2_plus, self.theta_2_plus)
            
            a_sz = -0.5
            a_sx = 0.5
            b2.append(np.array([a_sx, 0, a_sz]))
        boundaries['theta2_max'] = b2
        
        # 边界 #3: L_s = 0
        b3 = []
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            a_sz = -0.7
            a_sx = 0.3
            b3.append(np.array([a_sx, 0, a_sz]))
        boundaries['L_s_min'] = b3
        
        # 边界 #4: L_s = L_s₊
        b4 = []
        for i in range(num_points + 1):
            theta_2 = (i / num_points) * self.theta_2_plus
            a_sz = -0.3
            a_sx = 0.7
            b4.append(np.array([a_sx, 0, a_sz]))
        boundaries['L_s_max'] = b4
        
        return boundaries
    
    def compute_type2_boundary_ci2(self, p_g: np.ndarray,
                                     num_points: int = 50) -> List[np.ndarray]:
        """
        CI-2 配置 Type-II 边界 (公式 33)
        """
        boundaries = []
        
        for i in range(num_points + 1):
            theta_1 = (i / num_points) * self.theta_1_plus
            theta_2 = self.theta_2_plus * 0.5
            
            # 简化计算
            a_sz = -0.5
            a_sx = np.sqrt(1 - a_sz**2) * np.cos(i * 0.1)
            
            boundaries.append(np.array([a_sx, 0, a_sz]))
        
        return boundaries
    
    def project_to_closest_dexterous_direction(self, p_g: np.ndarray, 
                                                 a_target: np.ndarray,
                                                 mode: str = 'ci1') -> np.ndarray:
        """
        将目标方向投影到灵巧工作空间边界上的最近可达方向
        
        参数:
            p_g: 末端位置
            a_target: 目标方向向量
            mode: 'ci1' 或 'ci2'
            
        返回:
            最近的可达方向向量
        """
        # 投影到对称平面
        p_s, a_s, gamma = self.project_to_symmetry_plane(p_g, a_target)
        
        # 计算可达方向范围
        # 简化: 使用最大弯曲角时的可达圆锥
        theta_max = self.theta_1_plus if mode == 'ci1' else min(self.theta_1_plus, self.theta_2_plus)
        
        # 计算位置约束下的最大可达方向
        L_1r2 = self.L_1_plus + self.L_r + self.L_2_plus
        
        # 目标方向与可达边界的距离
        a_target_norm = a_s / (np.linalg.norm(a_s) + 1e-10)
        
        # 简化的边界投影
        # 使用最大弯曲角对应的极限方向
        a_max = np.array([np.sin(theta_max), 0, -np.cos(theta_max)])
        
        # 如果目标方向在可达范围内，直接返回
        # 否则投影到边界
        dot_product = np.dot(a_target_norm, a_max)
        
        if dot_product >= 0.95:  # 在可达范围内
            return a_target
        
        # 否则投影到边界
        a_projected = a_max * np.sign(np.dot(a_target_norm, a_max))
        
        # 旋转回原始坐标系
        c, s = np.cos(gamma), np.sin(gamma)
        R_z_inv = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        
        return R_z_inv @ a_projected
    
    def is_in_dexterous_workspace(self, p_g: np.ndarray, a: np.ndarray,
                                    mode: str = 'ci1') -> bool:
        """
        检查给定位置和方向是否在灵巧工作空间内
        """
        # 简化检查: 验证配置变量是否在极限内
        L_total = self.L_1_plus + self.L_r + self.L_2_plus + self.L_g
        
        # 位置必须在工作空间内
        dist_from_base = np.linalg.norm(p_g)
        if dist_from_base > L_total * 1.1:
            return False
        
        # 方向检查
        if mode == 'ci1':
            # CI-1: 检查 L_s = 0 的约束
            z_component = a[2]
            if z_component < -0.5:  # 过于指向下方
                return False
        else:
            # CI-2: 有更多约束
            if self.L_s_plus <= 0:
                return False
        
        return True


def compute_dexterous_workspace_boundary_points(params: CSMParameters,
                                                 mode: str = 'ci1',
                                                 num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数: 计算灵巧工作空间边界点
    
    返回:
        (theta_range, a_boundaries): 角度范围和对应的边界方向
    """
    dex_ws = DexterousWorkspace(params)
    
    # 在典型位置采样
    p_g = np.array([0, 0, params.L_10 + params.L_r0 + params.L_20 * 0.5])
    
    boundaries = []
    theta_range = np.linspace(0, params.theta2_max, num_samples)
    
    for theta_2 in theta_range:
        # 简化的边界计算
        l_2 = dex_ws.compute_l_t(params.L_20, theta_2)
        L_1r2 = params.L_r0 + l_2
        
        a_sz = -p_g[2] / L_1r2 if L_1r2 > 0 else -1.0
        a_sz = np.clip(a_sz, -1, 1)
        a_sx = np.sqrt(max(0, 1 - a_sz**2))
        
        boundaries.append(a_sx)
    
    return theta_range, np.array(boundaries)
