import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data as erd
import meshcat.geometry as mg
import time
from WorkspaceDiscretizer import WorkspaceDiscretizer

class CapabilityMapAnalyzer:
    def __init__(self, filepath, discretizer=None):
        print(f"正在从 {filepath} 加载能力图数据...")
        data = np.load(filepath, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            self.cmap = data
        elif 'cmap' in data:
            self.cmap = data['cmap']
        else:
            self.cmap = data[data.files[0]]

        if discretizer is not None:
            self.discretizer = discretizer
        elif 'config' in data:
            config = data['config'].item()
            print("发现内嵌配置，正在自动创建 WorkspaceDiscretizer...")
            self.discretizer = WorkspaceDiscretizer.from_config(config)
        else:
            raise ValueError("未提供 discretizer 实例，且数据文件中未包含配置信息！")

        self.n_p = self.discretizer.n_p
        self.m_o = self.discretizer.m_o
        self._compute_indices()

    def _compute_indices(self):
        """计算到达率指标"""
        self.R_o = np.sum(self.cmap, axis=(3, 4))
        self.D_o = (self.R_o / (self.n_p * self.m_o)) * 100.0
        self.point_reachable = np.any(self.cmap, axis=4)
        self.R = np.sum(self.point_reachable, axis=3)
        self.D = (self.R / self.n_p) * 100.0
        # 打印关键指标
        print("=== 能力图关键指标 ===")
        print(f"总姿态数 n_p = {self.n_p}")
        print(f"总朝向数 m_o = {self.m_o}")
        print(f"体素空间尺寸: {self.cmap.shape[:3]}")
        print(f"整体可达率 D  (任意姿态可达): 均值={np.mean(self.D):.2f}%, 最大={np.max(self.D):.2f}%, 最小={np.min(self.D):.2f}%")
        print(f"方向可达率 D_o(任意方向可达): 均值={np.mean(self.D_o):.2f}%, 最大={np.max(self.D_o):.2f}%, 最小={np.min(self.D_o):.2f}%")
        print("====================")

    def visualize_meshcat(self, robot_name='ur5', metric='D_o', threshold=1.0, 
                            slice_axis='y', cut_half=True, alpha=0.3):
            """
            使用 Meshcat 渲染，包含环境美化、原生的底座加载和垂直剖视图
            """
            import hppfcl
            print(f"正在启动 Meshcat 渲染器...")
            
            # 1. 加载机器人
            robot = erd.load(robot_name)
            
            # --- 核心修复：使用你验证过的原生方法添加底座 ---
            # UR5 的基座我们设置高 0.8m，半径 0.15m
            pedestal_h = 0.2
            pedestal_geom = pin.GeometryObject(
                "pedestal", 0, 
                pin.SE3(np.eye(3), np.array([0, 0, -pedestal_h / 2])), # 向下偏移一半高度
                hppfcl.Cylinder(0.15, pedestal_h)
            )
            pedestal_geom.meshColor = np.array([0.3, 0.3, 0.3, 1.0]) # 深灰色
            # 必须在 loadViewerModel 之前添加到 visual_model
            robot.visual_model.addGeometryObject(pedestal_geom)

            # 2. 启动可视化器
            robot.setVisualizer(MeshcatVisualizer())
            robot.initViewer(open=True)
            robot.loadViewerModel()
            
            # --- 核心改进：去除蓝色背景和网格 ---
            viewer = robot.viz.viewer
            # viewer["/Grid"].set_property("visible", False)          # 关掉地面的格子
            viewer["/Background"].set_property("visible", False)    # 关掉默认的渐变蓝背景
            # 设置一个干净的纯白/浅灰背景
            viewer["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
            viewer["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])
            viewer["/Background"].set_property("visible", True)

            # 3. 设置机械臂姿态
            # UR5 自然弯曲姿态
            q_ready = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
            robot.display(q_ready)

            # 4. 渲染体素球 (能力图)
            data_matrix = self.D_o if metric == 'D_o' else self.D
            cmap_color = plt.get_cmap('jet_r')
            radius = self.discretizer.l_c * 0.4
            n_x, n_y, n_z = data_matrix.shape

            def rgba_to_hex(rgba):
                r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                return (r << 16) + (g << 8) + b

            print(f"渲染剖视图中 (剖切轴: {slice_axis})...")
            count = 0
            
            for i in range(n_x):
                if cut_half and slice_axis == 'x' and i > n_x // 2: continue
                for j in range(n_y):
                    if cut_half and slice_axis == 'y' and j > n_y // 2: continue
                    for k in range(n_z):
                        if cut_half and slice_axis == 'z' and k > n_z // 2: continue
                        
                        val = data_matrix[i, j, k]
                        if val >= threshold:
                            center = self.discretizer.get_voxel_center((i, j, k))
                            color_hex = rgba_to_hex(cmap_color(val / 100.0))
                            
                            material = mg.MeshPhongMaterial(color=color_hex, transparent=True, opacity=alpha)
                            name = f"cmap/voxel_{i}_{j}_{k}"
                            viewer[name].set_object(mg.Sphere(radius), material)
                            
                            T = np.eye(4)
                            T[:3, 3] = center
                            viewer[name].set_transform(T)
                            count += 1

            print(f"渲染完成！共显示 {count} 个体素。")
            try:
                while True: time.sleep(1)
            except KeyboardInterrupt:
                print("退出。")

if __name__ == "__main__":
    file_path = "./data/ur5_fk_cmap.npz"
    analyzer = CapabilityMapAnalyzer(filepath=file_path)
    
    # 可视化
    analyzer.visualize_meshcat(robot_name='ur5', metric='D', 
                              threshold=1.0, slice_axis='y', 
                              cut_half=False, alpha=0.3)