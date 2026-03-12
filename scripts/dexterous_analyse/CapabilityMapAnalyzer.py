import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
from WorkspaceDiscretizer import WorkspaceDiscretizer
import pinocchio as pin
import example_robot_data as erd
import meshcat.geometry as mg
import time

class CapabilityMapAnalyzer:
    def __init__(self, filepath, discretizer=None):
        """
        读取本地的 .npz 文件并计算能力图指标
        :param filepath: .npz 文件路径
        :param discretizer: 可选，WorkspaceDiscretizer 实例。如果不提供，尝试从 .npz 中读取配置创建。
        """
        print(f"正在从 {filepath} 加载能力图数据...")
        data = np.load(filepath, allow_pickle=True)
        
        # 兼容旧版 .npy (只存了 array) 和新版 .npz (存了 cmap 和 config)
        if isinstance(data, np.ndarray):
             self.cmap = data
        elif 'cmap' in data:
            self.cmap = data['cmap']
        else:
            # 尝试直接读取第一个数组
            self.cmap = data[data.files[0]]

        # 处理 Discretizer
        if discretizer is not None:
            self.discretizer = discretizer
        elif 'config' in data:
            # 从 npz 中读取配置 (注意：np.load 读取的字典可能包含 0-d array，需要 .item())
            config = data['config'].item()
            print("发现内嵌配置，正在自动创建 WorkspaceDiscretizer...")
            self.discretizer = WorkspaceDiscretizer.from_config(config)
        else:
            raise ValueError("未提供 discretizer 实例，且数据文件中未包含配置信息！")

        self.n_p = self.discretizer.n_p
        self.m_o = self.discretizer.m_o

        print("正在计算 Reachability Indices...")
        self._compute_indices()

    def _compute_indices(self):
        """
        利用 NumPy 的高阶张量操作，极速计算 D 和 Do
        self.cmap 的 shape 为 (x, y, z, point_idx, rot_idx)
        """
        # 计算 D_o (带 Z 轴旋转的到达率)
        # 统计在 axis=3 (点) 和 axis=4 (旋转) 上 True 的总数
        self.R_o = np.sum(self.cmap, axis=(3, 4))
        # 根据公式计算百分比
        self.D_o = (self.R_o / (self.n_p * self.m_o)) * 100.0

        # 计算 D (仅看球面点是否可达)
        # 只要该点在任意 rot_idx 下为 True，该点即视为可达
        self.point_reachable = np.any(self.cmap, axis=4)
        self.R = np.sum(self.point_reachable, axis=3)
        self.D = (self.R / self.n_p) * 100.0

    def visualize(self, metric='D_o', threshold=1.0, cut_half=True, alpha=0.8):
            """
            使用 Open3D 渲染体素球。
            :param metric: 可视化指标，'D_o' 或 'D'
            :param threshold: 可视化阈值，仅显示大于等于该值的体素
            :param cut_half: 是否仅渲染下半部分（针对对称场景）
            :param alpha: 透明度 (0.0 完全透明, 1.0 不透明)
            """
            print(f"正在准备 Open3D 渲染，使用指标: {metric}...")
            data_matrix = self.D_o if metric == 'D_o' else self.D
            
            # 1. 核心修改：使用 'jet_r' 使得 100% (大值) 映射为蓝色
            cmap_color = plt.get_cmap('jet_r') 
            
            max_val = np.max(data_matrix)
            if max_val == 0:
                print("警告：该图没有任何可达体素！")
                return

            master_mesh = o3d.geometry.TriangleMesh()
            # 增加球体的精细度 (resolution=10) 避免合并后顶点过多过卡
            base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.discretizer.l_c * 0.4, resolution=10)
            base_sphere.compute_vertex_normals()

            n_x, n_y, n_z = data_matrix.shape
            count = 0
            
            for i in range(n_x):
                for j in range(n_y):
                    if cut_half and j > n_y // 2:
                        continue
                    for k in range(n_z):
                        val = data_matrix[i, j, k]
                        if val >= threshold:
                            center = self.discretizer.get_voxel_center((i, j, k))
                            
                            # 映射颜色
                            norm_val = val / 100.0
                            color = cmap_color(norm_val)[:3] 
                            
                            sphere = copy.deepcopy(base_sphere)
                            sphere.translate(center)
                            sphere.paint_uniform_color(color)
                            
                            master_mesh += sphere
                            count += 1

            print(f"构建完毕，共渲染 {count} 个可达体素球。")

            # 2. 设置透明材质
            # 创建一个支持透明的材质记录
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"  # 使用支持透明的着色器
            mat.base_color = [1.0, 1.0, 1.0, alpha] # 设置基础颜色的 Alpha 通道
            
            # 坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

            # 3. 使用现代渲染器 o3d.visualization.draw
            # 这比 draw_geometries 渲染效果更好，且原生支持材质属性
            o3d.visualization.draw(
                        [
                            {'name': 'capability_map', 'geometry': master_mesh, 'material': mat},
                            {'name': 'origin', 'geometry': coord_frame}
                        ],
                        title=f"Capability Map ({metric}) - Blue is High", # 修复此处参数名
                        width=1024, 
                        height=768,
                        show_skybox=False  # 可选：关闭背景天空盒，让渲染看起来更干净
                    )


# --- 运行逻辑 ---
if __name__ == "__main__":
    # 指向你刚刚跑出来的 npz 文件
    file_path = "./data/ur5_capability_map.npz"
    
    # 直接加载，无需手动构造 discretizer
    analyzer = CapabilityMapAnalyzer(filepath=file_path)
    
    # 打印一些统计数据
    print(f"最大 D_o: {np.max(analyzer.D_o):.2f}%")
    reachable = analyzer.D_o[analyzer.D_o > 0]
    avg_Do = np.mean(reachable) if len(reachable) > 0 else 0.0
    print(f"平均 D_o (仅计算可达区域): {avg_Do:.2f}%")
    
    # 呼出三维图像 (默认切开一半)
    analyzer.visualize(metric='D', threshold=1.0, cut_half=False)