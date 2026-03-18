"""
Module: vsik.py
Description: Variable Separation Inverse Kinematics (VS-IK) for 2-Segment Continuum Robots.
基于论文: "Inverse Kinematics and Dexterous Workspace Formulation for 2-Segment Continuum Robots With Inextensible Segments"
支持 Configuration CI-1 (单不可伸长节段) 和 CI-2 (双不可伸长节段)
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CSMParameters:
    """CSM 机器人物理参数，对应论文中的符号"""
    L_10: float  # 第一段最大长度 (m), 对应 L_1+
    L_20: float  # 第二段最大长度 (m), 对应 L_2+
    L_r0: float  # 刚性段长度 (m), 对应 L_r
    L_s0: float  # 基底滑动段最大长度 (m), 对应 L_{s+}
    L_tool: float  # 末端工具长度 (m), 对应 L_g 或 L_{2g}
    theta1_max: float  # 第一段最大弯曲角 (rad), 对应 θ_{1+}
    theta2_max: float  # 第二段最大弯曲角 (rad), 对应 θ_{2+}
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CSMParameters':
        """从配置字典创建参数对象"""
        robot = config.get('robot', config)
        return cls(
            L_10=robot.get('L_10', 0.04),
            L_20=robot.get('L_20', 0.06),
            L_r0=robot.get('L_r0', 0.02),
            L_s0=robot.get('L_s0', 0.15),
            L_tool=robot.get('L_tool', 0.0),
            theta1_max=robot.get('theta1_max', np.pi/2),
            theta2_max=robot.get('theta2_max', 2*np.pi/3)
        )


class VSIKSolver:
    """
    VS-IK 求解器类
    使用变量分离方法将多变量逆运动学问题转化为单变量非线性方程求解
    """
    
    def __init__(self, params: CSMParameters):
        """
        初始化 VS-IK 求解器
        参数:
            params: CSM 机器人参数
        """
        self.params = params
        
        # 预计算常数
        self.L_1_plus = params.L_10
        self.L_2_plus = params.L_20
        self.L_r = params.L_r0
        self.L_s_plus = params.L_s0
        self.L_g = params.L_tool
        self.theta_1_plus = params.theta1_max
        self.theta_2_plus = params.theta2_max
        
        # 计算最小弯曲半径下限 (防止 θ 接近 π 时 l 趋于无穷)
        # 论文中使用 r_{1-} 表示，这里设为合理的小值
        self.r_1_minus = 1e-6
        self.r_2_minus = 1e-6
    
    def compute_l_t(self, L_t: float, theta_t: float) -> float:
        """
        计算线段长度 l_t (公式 3)
        l_t = L_t * tan(θ_t/2) / θ_t
        当 θ_t = 0 时, l_t = L_t / 2
        """
        if abs(theta_t) < 1e-10:
            return L_t / 2.0
        return L_t * np.tan(theta_t / 2.0) / theta_t
    
    def compute_theta_from_l(self, L_t: float, l_t: float, 
                             theta_plus: float, r_minus: float = 1e-6) -> Optional[float]:
        """
        从线段长度 l_t 反求弯曲角 θ_t
        使用牛顿迭代求解: l_t = L_t * tan(θ_t/2) / θ_t
        """
        # 检查可行性
        L_t = min(L_t, L_t * np.tan(theta_plus/2) / theta_plus * 1.5)  # 宽松上界
        
        # 使用初始猜测
        theta = theta_plus / 2.0
        
        for _ in range(50):
            if abs(theta) < 1e-10:
                break
            f = self.compute_l_t(L_t, theta) - l_t
            # 导数: d(l_t)/d(θ_t)
            if abs(theta) > 1e-10:
                df = L_t * (0.5 * (1/np.cos(theta/2))**2 * 0.5 * theta - np.tan(theta/2)) / theta**2
                # 简化导数计算
                sin_half = np.sin(theta/2)
                cos_half = np.cos(theta/2)
                if abs(cos_half) > 1e-10:
                    tan_half = sin_half / cos_half
                    df = L_t * (0.25 * (1/(cos_half**2)) - tan_half/theta) / theta
                else:
                    df = L_t / (4 * cos_half**2 * theta)
            else:
                df = L_t / 4.0
            
            if abs(df) < 1e-15:
                break
                
            delta = f / df
            theta = theta - delta
            
            if theta < 0:
                theta = -theta
            if theta > theta_plus:
                theta = theta_plus
            
            if abs(delta) < 1e-8:
                break
        
        return theta if 0 < theta < theta_plus else None
    
    def compute_cos_theta1(self, p1: np.ndarray, p2: np.ndarray, L_1r2: float) -> float:
        """
        计算 cosθ₁ (公式 8)
        cosθ₁ = (p₂_z - p₁_z) / L_1r2
        """
        return (p2[2] - p1[2]) / L_1r2
    
    def compute_cos_theta2(self, p1: np.ndarray, p2: np.ndarray, 
                           a: np.ndarray, L_1r2: float) -> float:
        """
        计算 cosθ₂ (公式 9)
        cosθ₂ = (p₂ - p₁) · a / L_1r2
        """
        diff = p2 - p1
        return np.dot(diff, a) / L_1r2
    
    def equation_ci1(self, theta_2: float, p_g: np.ndarray, a: np.ndarray,
                     L_2: float, L_r: float) -> float:
        """
        CI-1 配置的单变量方程 (公式 19)
        仅包含 θ₂ 作为变量
        c₃(cosθ₂ + 1)l₂² + d₁l₂cosθ₂ + d₂l₂ + d₃cosθ₂ + d₄ = 0
        
        参数:
            theta_2: 第二段弯曲角 (rad)
            p_g: 末端执行器位置 (3,)
            a: 末端执行器方向向量 (3,), 即 ^w a = ^w R_g @ [0,0,1]^T
            L_2: 第二段长度
            L_r: 刚性段长度
            
        返回:
            方程残差值
        """
        # 计算 l₂
        l_2 = self.compute_l_t(L_2, theta_2)
        
        # 位置分量
        p_x, p_y, p_z = p_g
        a_x, a_y, a_z = a
        
        # 预计算
        p_sz = p_z - self.L_g * a_z  # 对应 p_{sz} (对称平面投影后)
        p_sx = p_x - self.L_g * a_x  # 对应 p_{sx}
        
        # 中间连接长度 (不含 L_s，因为 CI-1 中 L_s=0)
        L_1r2 = L_r + l_2  # 对应 L_{1r2} 但不含 l₁
        
        # 计算 l₁ 从几何关系
        # 从 (p₂ - p₁) 距离约束
        # ||p₂ - p₁||² = (l₁ + L_r)² + l₂² + 2(l₁+L_r)l₂cosθ₂
        # 其中 p₂ = p_g - L_g*a, p₁ = [0, 0, l₁]
        
        # 解 l₁ 的二次方程
        c1 = 1.0
        c2 = 2 * L_r * np.cos(theta_2)
        c3 = L_r**2 + l_2**2 + 2 * L_r * l_2 * np.cos(theta_2) - p_sx**2 - p_sz**2
        
        # l₁ = (-c2 ± sqrt(c2² - 4*c1*c3)) / (2*c1)
        discriminant = c2**2 - 4 * c1 * c3
        
        if discriminant < 0:
            return 1e10  # 无解
        
        sqrt_disc = np.sqrt(discriminant)
        l_1_1 = (-c2 + sqrt_disc) / 2.0
        l_1_2 = (-c2 - sqrt_disc) / 2.0
        
        # 选择正根
        l_1 = l_1_1 if l_1_1 > 0 else l_1_2
        if l_1 <= 0:
            return 1e10
        
        # 验证 θ₁ 的可行性
        cos_theta_1 = (p_sz - l_1) / (l_1 + L_r) if (l_1 + L_r) > 0 else 0
        cos_theta_1 = np.clip(cos_theta_1, -1, 1)
        theta_1 = np.arccos(cos_theta_1)
        
        # 检查 θ₁ 是否在限值内
        if theta_1 > self.theta_1_plus or theta_1 < 0:
            # 尝试另一个 l₁ 解
            l_1 = l_1_2 if l_1_1 > 0 else l_1_1
            if l_1 > 0:
                cos_theta_1 = (p_sz - l_1) / (l_1 + L_r) if (l_1 + L_r) > 0 else 0
                cos_theta_1 = np.clip(cos_theta_1, -1, 1)
                theta_1 = np.arccos(cos_theta_1)
        
        # 计算 L₁ (实际插入长度)
        L_1 = self.compute_l_t(l_1 * 2, theta_1) * 2 if theta_1 > 1e-6 else 0
        
        # 残差: L₁ 应在 [0, L_1_plus] 范围内
        residual = 0.0
        if L_1 < 0:
            residual += L_1**2
        if L_1 > self.L_1_plus:
            residual += (L_1 - self.L_1_plus)**2
        if theta_1 > self.theta_1_plus:
            residual += (theta_1 - self.theta_1_plus)**2
            
        return residual
    
    def equation_ci2(self, theta_1: float, p_g: np.ndarray, a: np.ndarray,
                     L_1: float, L_2: float, L_r: float) -> float:
        """
        CI-2 配置的单变量方程 (公式 13)
        仅包含 θ₁ 作为变量
        
        参数:
            theta_1: 第一段弯曲角 (rad)
            p_g: 末端执行器位置 (3,)
            a: 末端执行器方向向量 (3,)
            L_1: 第一段长度
            L_2: 第二段长度  
            L_r: 刚性段长度
            
        返回:
            方程残差值
        """
        # 计算 l₁
        l_1 = self.compute_l_t(L_1, theta_1)
        
        # 位置分量
        p_x, p_y, p_z = p_g
        a_x, a_y, a_z = a
        
        # 计算 p₂ (公式 6)
        p_2 = p_g - self.L_g * a
        
        # 计算 L_{1r2} = l₁ + L_r + l₂
        # 需要先通过迭代求解 l₂
        
        # 残差函数: F(l₂) = 0
        # ||p₂ - p₁||² = (l₁ + L_r + l₂)²
        # 其中 p₁ = [0, 0, l₁ + L_s]
        
        # 先假设 L_s = 0 进行初步计算
        p_1z = l_1  # p₁_z = l₁ + L_s (L_s=0 for initial)
        
        # 从距离约束解 l₂ (二次方程)
        # ||p₂ - p₁||² = (l₁ + L_r + l₂)²
        dx = p_2[0]
        dy = p_2[1]
        dz = p_2[2] - p_1z
        
        d_sq = dx**2 + dy**2 + dz**2
        L_1r2_sq = (l_1 + L_r)**2
        
        # 解二次方程: l₂² + 2(l₁+L_r)l₂ + (l₁+L_r)² - d² = 0
        a_coef = 1.0
        b_coef = 2 * (l_1 + L_r)
        c_coef = L_1r2_sq - d_sq
        
        discriminant = b_coef**2 - 4 * a_coef * c_coef
        
        if discriminant < 0:
            return 1e10
            
        sqrt_disc = np.sqrt(discriminant)
        l_2_1 = (-b_coef + sqrt_disc) / (2 * a_coef)
        l_2_2 = (-b_coef - sqrt_disc) / (2 * a_coef)
        
        # 选择合适的解
        l_2 = l_2_1 if l_2_1 > 0 else l_2_2
        if l_2 <= 0:
            return 1e10
            
        # 计算 L_s (公式 10)
        L_s = p_2[2] - self.L_g * a_z - (l_1 + L_r + l_2) * np.cos(theta_1) - l_1
        
        # 检查 L_s 可行性
        if L_s < 0 or L_s > self.L_s_plus:
            # 尝试另一个 l₂ 解
            l_2 = l_2_2 if l_2_1 > 0 else l_2_1
            if l_2 > 0:
                L_s = p_2[2] - self.L_g * a_z - (l_1 + L_r + l_2) * np.cos(theta_1) - l_1
        
        # 计算 cosθ₂ (公式 12)
        L_1r2 = l_1 + L_r + l_2
        if L_1r2 < 1e-10:
            return 1e10
            
        cos_theta_2 = (np.dot(p_2, a) + self.L_g * (a_z**2 - 1) - a_z * (p_2[2] - L_1r2 * np.cos(theta_1))) / L_1r2
        cos_theta_2 = np.clip(cos_theta_2, -1, 1)
        
        # 计算 θ₂
        theta_2 = np.arccos(cos_theta_2)
        
        # 验证 L₂ 与 θ₂ 的关系
        L_2_computed = l_2 * theta_2 / np.tan(theta_2/2) if theta_2 > 1e-6 else l_2 * 2
        
        # 残差
        residual = 0.0
        if L_s < 0:
            residual += L_s**2
        if L_s > self.L_s_plus:
            residual += (L_s - self.L_s_plus)**2
        if theta_2 > self.theta_2_plus:
            residual += (theta_2 - self.theta_2_plus)**2
        if L_2_computed > self.L_2_plus:
            residual += (L_2_computed - self.L_2_plus)**2
            
        return residual
    
    def solve_ci1(self, target_pose: Tuple[np.ndarray, np.ndarray], 
                   max_iter: int = 50, tol: float = 1e-6) -> Optional[Dict[str, float]]:
        """
        求解 CI-1 配置的逆运动学
        
        参数:
            target_pose: (位置, 方向向量) 元组
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            配置字典 {phi, theta_1, L_1, delta_1, theta_2, delta_2, L_s} 或 None
        """
        p_g, a = target_pose
        
        # 使用实际长度
        L_2 = self.L_2_plus  # CI-1 中第二段不可伸长
        L_r = self.L_r
        
        # 在 [0, theta_2_plus] 范围内搜索 θ₂
        best_result = None
        best_residual = 1e10
        
        # 多初始点搜索
        theta_2_initials = np.linspace(0.1, self.theta_2_plus * 0.9, 10)
        
        for theta_2_init in theta_2_initials:
            theta_2 = theta_2_init
            
            for _ in range(max_iter):
                # 计算残差
                f = self.equation_ci1(theta_2, p_g, a, L_2, L_r)
                
                if f < tol:
                    break
                    
                # 数值导数
                eps = 1e-6
                f_plus = self.equation_ci1(theta_2 + eps, p_g, a, L_2, L_r)
                f_minus = self.equation_ci1(theta_2 - eps, p_g, a, L_2, L_r)
                df = (f_plus - f_minus) / (2 * eps)
                
                if abs(df) < 1e-15:
                    break
                    
                delta = f / df
                theta_2 = theta_2 - delta
                
                # 约束
                if theta_2 < 0:
                    theta_2 = 0.01
                if theta_2 > self.theta_2_plus:
                    theta_2 = self.theta_2_plus
                    
                if abs(delta) < tol:
                    break
            
            # 检查解的有效性
            f_final = self.equation_ci1(theta_2, p_g, a, L_2, L_r)
            
            if f_final < best_residual:
                best_residual = f_final
                
                # 计算完整配置
                l_2 = self.compute_l_t(L_2, theta_2)
                
                # 重新计算 l₁
                p_2 = p_g - self.L_g * a
                dx, dy, dz = p_2[0], p_2[1], p_2[2]
                d_sq = dx**2 + dy**2 + dz**2
                L_1r2 = L_r + l_2
                L_1r2_sq = L_1r2**2
                
                a_coef = 1.0
                b_coef = 2 * L_1r2
                c_coef = L_1r2_sq - d_sq
                
                discriminant = b_coef**2 - 4 * a_coef * c_coef
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    l_1 = (-b_coef + sqrt_disc) / (2 * a_coef)
                    if l_1 <= 0:
                        l_1 = (-b_coef - sqrt_disc) / (2 * a_coef)
                    
                    if l_1 > 0:
                        # 计算 θ₁
                        cos_theta_1 = (p_2[2] - l_1) / L_1r2
                        cos_theta_1 = np.clip(cos_theta_1, -1, 1)
                        theta_1 = np.arccos(cos_theta_1)
                        
                        # 计算 L₁
                        L_1 = l_1 * 2 if theta_1 < 1e-6 else l_1 * theta_1 / np.tan(theta_1/2)
                        
                        # 计算 φ, δ₁, δ₂
                        # p₁ = [0, 0, l₁]
                        p_1 = np.array([0, 0, l_1])
                        
                        # φ - δ₁ = atan2(p₂_y, p₂_x) (公式 14)
                        phi_minus_delta1 = np.arctan2(p_2[1], p_2[0])
                        
                        # δ₂ 从姿态方程求解 (简化版)
                        # 这里使用几何近似
                        a_x, a_y, a_z = a[0], a[1], a[2]
                        delta_2 = np.arctan2(a_y, a_x)
                        delta_1 = phi_minus_delta1 - delta_2 if phi_minus_delta1 > delta_2 else phi_minus_delta1 + 2*np.pi - delta_2
                        
                        # φ
                        phi = phi_minus_delta1 + delta_1
                        
                        # 检查配置可行性
                        if (0 <= L_1 <= self.L_1_plus and 
                            0 <= theta_1 <= self.theta_1_plus and
                            0 <= theta_2 <= self.theta_2_plus):
                            best_result = {
                                'phi': phi % (2 * np.pi),
                                'theta_1': theta_1,
                                'L_1': L_1,
                                'delta_1': delta_1 % (2 * np.pi),
                                'theta_2': theta_2,
                                'delta_2': delta_2 % (2 * np.pi),
                                'L_s': 0.0,  # CI-1: L_s = 0
                                'L_r': L_r,
                                'L_2': L_2,
                                'residual': f_final
                            }
        
        return best_result
    
    def solve_ci2(self, target_pose: Tuple[np.ndarray, np.ndarray],
                   max_iter: int = 50, tol: float = 1e-6) -> Optional[Dict[str, float]]:
        """
        求解 CI-2 配置的逆运动学
        
        参数:
            target_pose: (位置, 方向向量) 元组
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            配置字典 {phi, theta_1, L_1, delta_1, theta_2, delta_2, L_s} 或 None
        """
        p_g, a = target_pose
        
        # 使用实际长度
        L_1 = self.L_1_plus  # CI-2 中第一段不可伸长
        L_2 = self.L_2_plus
        L_r = self.L_r
        
        # 在 [0, theta_1_plus] 范围内搜索 θ₁
        best_result = None
        best_residual = 1e10
        
        theta_1_initials = np.linspace(0.1, self.theta_1_plus * 0.9, 10)
        
        for theta_1_init in theta_1_initials:
            theta_1 = theta_1_init
            
            for _ in range(max_iter):
                f = self.equation_ci2(theta_1, p_g, a, L_1, L_2, L_r)
                
                if f < tol:
                    break
                    
                eps = 1e-6
                f_plus = self.equation_ci2(theta_1 + eps, p_g, a, L_1, L_2, L_r)
                f_minus = self.equation_ci2(theta_1 - eps, p_g, a, L_1, L_2, L_r)
                df = (f_plus - f_minus) / (2 * eps)
                
                if abs(df) < 1e-15:
                    break
                    
                delta = f / df
                theta_1 = theta_1 - delta
                
                if theta_1 < 0:
                    theta_1 = 0.01
                if theta_1 > self.theta_1_plus:
                    theta_1 = self.theta_1_plus
                    
                if abs(delta) < tol:
                    break
            
            f_final = self.equation_ci2(theta_1, p_g, a, L_1, L_2, L_r)
            
            if f_final < best_residual:
                best_residual = f_final
                
                # 计算完整配置
                l_1 = self.compute_l_t(L_1, theta_1)
                
                p_2 = p_g - self.L_g * a
                
                # 解 l₂
                dx, dy, dz = p_2[0], p_2[1], p_2[2]
                L_1r = l_1 + L_r
                d_sq = dx**2 + dy**2 + dz**2
                L_1r_sq = L_1r**2
                
                a_coef = 1.0
                b_coef = 2 * L_1r
                c_coef = L_1r_sq - d_sq
                
                discriminant = b_coef**2 - 4 * a_coef * c_coef
                
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    l_2 = (-b_coef + sqrt_disc) / (2 * a_coef)
                    if l_2 <= 0:
                        l_2 = (-b_coef - sqrt_disc) / (2 * a_coef)
                    
                    if l_2 > 0:
                        # 计算 L_s (公式 10)
                        L_1r2 = l_1 + L_r + l_2
                        cos_theta_1 = (p_2[2] - l_1) / L_1r2 if L_1r2 > 0 else 0
                        L_s = p_2[2] - self.L_g * a[2] - L_1r2 * cos_theta_1 - l_1
                        
                        # 计算 θ₂
                        cos_theta_2 = (np.dot(p_2, a) + self.L_g * (a[2]**2 - 1) - 
                                       a[2] * (p_2[2] - L_1r2 * cos_theta_1)) / L_1r2
                        cos_theta_2 = np.clip(cos_theta_2, -1, 1)
                        theta_2 = np.arccos(cos_theta_2)
                        
                        # 计算 φ, δ₁, δ₂
                        p_1 = np.array([0, 0, l_1 + L_s])
                        phi_minus_delta1 = np.arctan2(p_2[1], p_2[0])
                        a_x, a_y, a_z = a[0], a[1], a[2]
                        delta_2 = np.arctan2(a_y, a_x)
                        delta_1 = phi_minus_delta1 - delta_2 if phi_minus_delta1 > delta_2 else phi_minus_delta1 + 2*np.pi - delta_2
                        phi = phi_minus_delta1 + delta_1
                        
                        # 检查配置可行性
                        if (0 <= L_s <= self.L_s_plus and 
                            0 <= theta_1 <= self.theta_1_plus and
                            0 <= theta_2 <= self.theta_2_plus):
                            best_result = {
                                'phi': phi % (2 * np.pi),
                                'theta_1': theta_1,
                                'L_1': L_1,
                                'delta_1': delta_1 % (2 * np.pi),
                                'theta_2': theta_2,
                                'delta_2': delta_2 % (2 * np.pi),
                                'L_s': L_s,
                                'L_r': L_r,
                                'L_2': L_2,
                                'residual': f_final
                            }
        
        return best_result
    
    def solve(self, target_pose: Tuple[np.ndarray, np.ndarray],
              mode: str = 'auto') -> Optional[Dict[str, float]]:
        """
        统一求解入口
        
        参数:
            target_pose: (位置, 方向向量) 元组
            mode: 'ci1', 'ci2', 或 'auto' (自动选择)
            
        返回:
            配置字典或 None
        """
        if mode == 'auto':
            # 尝试 CI-1，再尝试 CI-2
            result = self.solve_ci1(target_pose)
            if result is None:
                result = self.solve_ci2(target_pose)
        elif mode.lower() == 'ci1':
            result = self.solve_ci1(target_pose)
        elif mode.lower() == 'ci2':
            result = self.solve_ci2(target_pose)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return result


def compute_fk(params: CSMParameters, config: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    正运动学验证 - 给定配置计算末端位姿
    
    参数:
        params: 机器人参数
        config: 配置字典
        
    返回:
        (位置, 方向向量)
    """
    from .model import CSM
    
    csm = CSM(
        L_10=params.L_10,
        L_20=params.L_20,
        L_r0=params.L_r0,
        L_s0=params.L_s0,
        L_tool=params.L_tool,
        theta1_max=params.theta1_max,
        theta2_max=params.theta2_max
    )
    
    # 根据配置确定模式
    if config.get('L_s', 0) > 0:
        mode = 4  # CI-2 模式
    else:
        mode = 3  # CI-1 模式
    
    csm.set_state(
        mode=mode,
        phi=config.get('phi', 0),
        L1=config.get('L_1', params.L_10),
        L2=config.get('L_2', params.L_20),
        Lr=config.get('L_r', params.L_r0),
        Ls=config.get('L_s', 0),
        theta_1=config.get('theta_1', 0),
        theta_2=config.get('theta_2', 0),
        delta_1=config.get('delta_1', 0),
        delta_2=config.get('delta_2', 0)
    )
    
    return csm.pose[:3], csm.pose[3:]
