import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from itertools import product, combinations

class WsDiscretizer:
    """
    负责机器人工作空间位置(R^3)和姿态(SO(3))的离散化。
    """
    def __init__(self, arm_length, l_c=0.05, n_p=200, delta_o=30.0):
        # 1. 初始化基本参数
        self.l_ws = 2.0 * arm_length 
        self.l_c = float(l_c)
        self.n_c = math.ceil(self.l_ws / self.l_c)
        
        self.n_p = n_p  
        self.delta_o = delta_o  
        self.m_o = math.floor(360.0 / self.delta_o)
        
        # 2. 预计算离散化姿态集
        self.sphere_points = self._generate_spiral_points()
        self.discrete_frames = self._generate_discrete_frames()

        # 3. 计算总离散化体素数
        self.total_discrete_frames = self.n_c**3 * self.n_p * self.m_o

    # ---------------------------------------------------------
    # 修正后的平移取整映射法
    # ---------------------------------------------------------
    def get_voxel_index(self, t):
        t = np.asarray(t)
        if np.any(np.abs(t) > self.l_ws / 2.0):
            return None 
            
        t_min = -self.l_ws / 2.0
        g = np.floor((t - t_min) / self.l_c)
        g = np.clip(g, 0, self.n_c - 1)
        return g.astype(int)

    def get_voxel_center(self, g):
        g = np.asarray(g)
        t_min = -self.l_ws / 2.0
        t = t_min + (g + 0.5) * self.l_c
        return t

    # ---------------------------------------------------------
    # 内部生成方法 (保持原样)
    # ---------------------------------------------------------
    def _generate_spiral_points(self):
        points = np.zeros((self.n_p, 3))
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(self.n_p):
            y = 1.0 - (i / float(self.n_p - 1)) * 2.0
            radius = np.sqrt(1.0 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
        return points

    def _generate_discrete_frames(self):
        frames = []
        # 为加快可视化测试速度，这里做个精简。
        # 原逻辑是对的，直接保留即可。
        return frames

# --- 可视化测试逻辑 ---
def visualize_mapping(discretizer):
    print("开始生成可视化图像...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 生成连续的测试点 (3D 螺旋线，贯穿工作空间)
    t_vals = np.linspace(-np.pi * 4, np.pi * 4, 300)
    bound = discretizer.l_ws / 2.0
    r = bound * 0.8  # 保持在边界内
    
    x_orig = r * np.cos(t_vals)
    y_orig = r * np.sin(t_vals)
    z_orig = np.linspace(-r, r, 300)
    pts_orig = np.column_stack((x_orig, y_orig, z_orig))

    pts_recon = []
    valid_orig = []

    # 2. 执行 连续 -> 离散 -> 连续 的映射
    for pt in pts_orig:
        idx = discretizer.get_voxel_index(pt)
        if idx is not None:
            recon_pt = discretizer.get_voxel_center(idx)
            pts_recon.append(recon_pt)
            valid_orig.append(pt)

    pts_recon = np.array(pts_recon)
    valid_orig = np.array(valid_orig)

    # 3. 绘制原始的连续曲线
    ax.plot(valid_orig[:,0], valid_orig[:,1], valid_orig[:,2], 
            label='Original Continuous Curve', color='dodgerblue', alpha=0.6, linewidth=3)

    # 4. 绘制离散化后的体素中心点 (阶梯状)
    # 设定一个视觉转换系数（将"米"映射到"屏幕磅值"）
    # 你可以根据你的屏幕分辨率和窗口大小微调这个值，通常在 300-800 之间
    display_scale = 500  
    
    # 目标物理尺寸: 0.8 * l_c
    # scatter 的 s 是面积，所以要对计算出的磅值求平方
    voxel_marker_size = (0.8 * discretizer.l_c * display_scale) ** 2

    # 4. 绘制离散化后的体素中心点 (阶梯状)
    ax.scatter(pts_recon[:,0], pts_recon[:,1], pts_recon[:,2], 
               label=f'Voxel Centers (0.8 * {discretizer.l_c}m)', 
               color='crimson', marker='s', s=voxel_marker_size, alpha=0.8)

    # 5. 绘制映射误差线 (连接原始点和它被映射到的体素中心)
    for i in range(0, len(valid_orig), 5):  # 每隔5个点画一条线防止太密集
        ax.plot([valid_orig[i,0], pts_recon[i,0]],
                [valid_orig[i,1], pts_recon[i,1]],
                [valid_orig[i,2], pts_recon[i,2]], 
                color='gray', linestyle='--', alpha=0.5)

    # 6. 绘制工作空间边界框 (Bounding Box)
    r_bounds = [-bound, bound]
    for s, e in combinations(np.array(list(product(r_bounds, r_bounds, r_bounds))), 2):
        if np.isclose(np.linalg.norm(s - e), 2 * bound):
            ax.plot3D(*zip(s, e), color="black", linestyle=':', alpha=0.3)

    # 7. 设置视图和标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Workspace Voxelization Mapping\n(Arm Length: {bound}m, Voxel Size: {discretizer.l_c}m)')
    ax.legend()
    
    # 强制比例一致，使得包围盒看起来是个正方体
    ax.set_box_aspect([1,1,1]) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用较大的 voxel_size (0.1m) 让阶梯效应在图中更明显
    discretizer = WsDiscretizer(arm_length=0.85, l_c=0.1, n_p=50, delta_o=30.0)
    visualize_mapping(discretizer)