import json
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from scipy.spatial import KDTree
from tqdm.auto import tqdm
from pathlib import Path

# ==========================================
# Phase 1: 集中化配置参数 (Configuration)
# ==========================================
@dataclass
class Config:
    VOXEL_SIZE: float = 0.005          # 体素边长 (米)，建议根据实际工作空间大小调整，如 5mm
    SPHERE_N_DIRS: int = 100           # 球面均匀采样的朝向数量 (斐波那契网格)
    ROLL_BINS: int = 12                # 绕轴旋转 360° 划分的区间数 (12 bins = 30°/bin)
    MIN_POINTS_PER_VOXEL: int = 10     # 统计学阈值，过滤噪声体素
    DATA_PATH: str = "./data/workspace_data.json"
    OUTPUT_PATH: str = "./data/dexterous_workspace_analysis.csv"

# ==========================================
# 工具函数: 斐波那契球面与局部基底
# ==========================================
def generate_fibonacci_sphere(n_points):
    """生成单位球面上的均匀分布点 (斐波那契网格)"""
    phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
    indices = np.arange(n_points)
    y = 1 - (indices / float(n_points - 1)) * 2  # y 从 1 到 -1
    radius = np.sqrt(1 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack((x, y, z))

def generate_local_bases(directions):
    """为每个参考方向生成正交的局部切平面基底 (X, Y)"""
    X_bases, Y_bases = [], []
    for z_dir in directions:
        # 选择一个不与 z_dir 平行的任意向量
        v = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(z_dir, v)) > 0.99:
            v = np.array([0.0, 1.0, 0.0])
        
        # 叉乘求正交基
        x = np.cross(v, z_dir)
        x /= np.linalg.norm(x)
        y = np.cross(z_dir, x)
        y /= np.linalg.norm(y)
        
        X_bases.append(x)
        Y_bases.append(y)
    return np.array(X_bases), np.array(Y_bases)

def main():
    cfg = Config()
    
    # ==========================================
    # Phase 2: 数据加载与预处理 (Data Loading)
    # ==========================================
    print(f"Loading data from {cfg.DATA_PATH}...")
    with open(cfg.DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # # 提取位置和旋转矩阵并向量化
    # if len(data) == 0:
    #     raise ValueError("No data samples found in the JSON file.")
    # if len(data) != len(data[0]['pose']) or len(data) != len(data[0]['rotation_matrix']):
    #     raise ValueError("Mismatch between number of pose samples and rotation matrices.")
    
    positions = np.array([item['pose'][:3] for item in data])
    rotations = np.array([item['rotation_matrix'] for item in data])
    n_samples = len(positions)
    print(f"Loaded {n_samples} samples.")

    # ==========================================
    # Phase 3: 空间体素化分组 (Spatial Voxelization)
    # ==========================================
    print("Voxelizing workspace...")
    P_min = positions.min(axis=0)
    V_indices = np.floor((positions - P_min) / cfg.VOXEL_SIZE).astype(int)
    
    # 使用 defaultdict 稀疏存储相同体素内的点索引
    voxel_dict = defaultdict(list)
    for idx, v in enumerate(V_indices):
        voxel_dict[tuple(v)].append(idx)
        
    print(f"Generated {len(voxel_dict)} unique voxels.")

    # ==========================================
    # Phase 4: 姿态的二维离散化 (Discretization Init)
    # ==========================================
    print("Initializing direction and roll discretization...")
    ref_dirs = generate_fibonacci_sphere(cfg.SPHERE_N_DIRS)
    kdtree = KDTree(ref_dirs)
    X_bases, Y_bases = generate_local_bases(ref_dirs)

    # 准备存储结果的列表
    analysis_results = []

    # ==========================================
    # Phase 5 & 6: 计算 RDI 与 拟合形状原语
    # ==========================================
    print("Analyzing voxels and calculating RDI...")
    for voxel_coords, indices in tqdm(voxel_dict.items(), desc="Analyzing Voxels", dynamic_ncols=True):
        if len(indices) < cfg.MIN_POINTS_PER_VOXEL:
            continue
            
        voxel_rots = rotations[indices]
        
        # 提取当前体素内所有的接近矢量 (z轴) 和 工具方向矢量 (x轴)
        approach_vectors = voxel_rots[:, :, 2]  # N x 3
        x_vectors = voxel_rots[:, :, 0]         # N x 3
        
        # 查找最近的球面朝向索引
        _, D_idx_array = kdtree.query(approach_vectors)
        
        # 计算绕轴旋转 (Roll) 并映射
        R_idx_list = []
        for i, d_idx in enumerate(D_idx_array):
            x_tool = x_vectors[i]
            x_ref = X_bases[d_idx]
            y_ref = Y_bases[d_idx]
            
            # 投影到局部切平面
            proj_x = np.dot(x_tool, x_ref)
            proj_y = np.dot(x_tool, y_ref)
            
            # 映射到 [0, 2π]
            theta_roll = np.arctan2(proj_y, proj_x)
            if theta_roll < 0:
                theta_roll += 2 * np.pi
                
            # 离散化 Roll
            r_idx = int(np.floor((theta_roll / (2 * np.pi)) * cfg.ROLL_BINS)) % cfg.ROLL_BINS
            R_idx_list.append(r_idx)
            
        # 计算 RDI
        # 按朝向 D_idx 分组，统计唯一 R_idx
        dir_to_rolls = defaultdict(set)
        for d_idx, r_idx in zip(D_idx_array, R_idx_list):
            dir_to_rolls[d_idx].add(r_idx)
            
        # 计算“被激活”的朝向的平均 RDI_dir
        rdi_dirs = [len(rolls) / cfg.ROLL_BINS for rolls in dir_to_rolls.values()]
        mean_rdi = np.mean(rdi_dirs) if rdi_dirs else 0.0

        # 圆锥体拟合 (Cone Fitting)
        # 1. 计算主轴 (归一化的平均矢量)
        cone_axis = np.mean(approach_vectors, axis=0)
        norm = np.linalg.norm(cone_axis)
        if norm > 1e-6:
            cone_axis /= norm
        else:
            cone_axis = np.array([0.0, 0.0, 1.0]) # 兜底策略
            
        # 2. 计算 95% 分位数开角 (防止极端离群点导致开角异常)
        dot_products = np.clip(np.sum(approach_vectors * cone_axis, axis=1), -1.0, 1.0)
        angles = np.arccos(dot_products)
        cone_half_angle_rad = np.percentile(angles, 95)
        
        # 计算体素的真实物理中心坐标
        voxel_center_x = P_min[0] + (voxel_coords[0] + 0.5) * cfg.VOXEL_SIZE
        voxel_center_y = P_min[1] + (voxel_coords[1] + 0.5) * cfg.VOXEL_SIZE
        voxel_center_z = P_min[2] + (voxel_coords[2] + 0.5) * cfg.VOXEL_SIZE
        
        # 收集结果
        analysis_results.append({
            "voxel_ix": voxel_coords[0],
            "voxel_iy": voxel_coords[1],
            "voxel_iz": voxel_coords[2],
            "center_x": voxel_center_x,
            "center_y": voxel_center_y,
            "center_z": voxel_center_z,
            "sample_count": len(indices),
            "mean_RDI": mean_rdi,
            "cone_axis_x": cone_axis[0],
            "cone_axis_y": cone_axis[1],
            "cone_axis_z": cone_axis[2],
            "cone_half_angle_rad": cone_half_angle_rad,
            "cone_half_angle_deg": np.degrees(cone_half_angle_rad)
        })

    # ==========================================
    # Phase 7: 结果输出 (Output Generation)
    # ==========================================
    df = pd.DataFrame(analysis_results)
    
    out_path = Path(cfg.OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"\nAnalysis complete! Processed {len(df)} valid voxels.")
    print(f"Results saved to: {out_path}")
    print("\nPreview of metrics:")
    print(df[['sample_count', 'mean_RDI', 'cone_half_angle_deg']].describe())

import matplotlib.pyplot as plt

# ==========================================
# Phase 8: 三维可视化 (Visualization)
# ==========================================
def visualize_workspace(csv_path, arrow_length=0.005, step=1):
    """
    使用 Matplotlib 可视化能力图
    - csv_path: 分析结果的 CSV 路径
    - arrow_length: 圆锥主轴箭头的长度
    - step: 降采样步长（若体素太多，设为 2 或 3 可以稀疏化箭头，避免画面太乱）
    """
    print(f"Loading data for visualization from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取中心坐标与 RDI (灵巧度指数)
    x = df['center_x'].values
    y = df['center_y'].values
    z = df['center_z'].values
    rdi = df['mean_RDI'].values
    
    # 1. 绘制体素中心，使用 jet 或 viridis 颜色映射表示 RDI 大小
    # cmap='jet' 与论文中冷暖色调图类似：红色表示灵巧度低，蓝色表示灵巧度高 (或者相反，可自行调整)
    scatter = ax.scatter(x, y, z, c=rdi, cmap='jet', s=15, alpha=0.8, edgecolors='none')
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15)
    cbar.set_label('Mean RDI (Rotational Dexterity Index)')
    
    # 2. 绘制圆锥的主轴方向 (使用 Quiver 矢量图)
    u = df['cone_axis_x'].values
    v = df['cone_axis_y'].values
    w = df['cone_axis_z'].values
    
    ax.quiver(
        x[::step], y[::step], z[::step], 
        u[::step], v[::step], w[::step], 
        length=arrow_length, normalize=True, color='black', alpha=0.6, linewidth=0.5
    )
    
    # 3. 坐标轴设置
    ax.set_xlabel('X Axis (m)')
    ax.set_ylabel('Y Axis (m)')
    ax.set_zlabel('Z Axis (m)')
    ax.set_title('Robot Dexterous Workspace Capability Map')
    
    # 调整视角和比例
    ax.view_init(elev=30, azim=45)
    
    # 强制各坐标轴比例一致，避免工作空间被拉伸变形
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 重新定义的入口函数
# ==========================================
if __name__ == "__main__":
    # 1. 运行核心分析逻辑（如果 CSV 已经生成且不想重复计算，可以把这行注释掉）
    main()
    
    # 2. 调用可视化
    cfg = Config()
    # step=2 表示每隔一个体素画一个箭头，避免箭头过于密集变成黑乎乎的一团
    visualize_workspace(cfg.OUTPUT_PATH, arrow_length=cfg.VOXEL_SIZE * 1.5, step=2)