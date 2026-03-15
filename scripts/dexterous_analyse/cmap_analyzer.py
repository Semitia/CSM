import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data as erd
import meshcat.geometry as mg
import time
import os
import polyscope as ps
from ws_discretizer import WsDiscretizer
# 强制 Open3D 忽略 Wayland，使用 Xwayland (X11) 模式
os.environ["WAYLAND_DISPLAY"] = ""
import open3d as o3d
import matplotlib as mpl


class CmapAnalyzer:
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
            print("发现内嵌配置，正在自动创建 WsDiscretizer...")
            self.discretizer = WsDiscretizer.from_config(config)

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
            q_ready = robot.q0 
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

    def visualize_polyscope(self, robot_name='ur5', metric='D_o', threshold=1.0, **kwargs):
        """
        高性能渲染：使用 Polyscope 展示能力图
        kwargs 仅为了兼容原有 Meshcat 调用参数，Polyscope 建议在 GUI 中手动剖切
        """
        # 1. 初始化
        if not ps.is_initialized():
            ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none") # 隐藏地面，方便全方位观察

        # 2. 提取并过滤数据 (矢量化处理)
        data_matrix = self.D_o if metric == 'D_o' else self.D
        indices = np.argwhere(data_matrix >= threshold)
        
        if len(indices) == 0:
            print("警告：没有超过阈值的体素，请降低 threshold。")
            return

        values = data_matrix[indices[:, 0], indices[:, 1], indices[:, 2]]
        
        # 批量转换索引到世界坐标 (假设 discretizer.get_voxel_center 支持批量更佳)
        centers = np.array([self.discretizer.get_voxel_center(tuple(idx)) for idx in indices])

        # 3. 注册点云
        # 在 Polyscope 中，点云默认支持透明度和半径调整
        ps_cloud = ps.register_point_cloud("capability_map", centers)

        # color_values = 100 - values
        # # 增加 vminmax=(0, 100) 强制固定颜色映射范围
        # ps_cloud.add_scalar_quantity(
        #     "reachability", 
        #     color_values, 
        #     enabled=True, 
        #     cmap='jet',
        #     vminmax=(0, 100),  # 固定 0~100 的映射范围
        #     onscreen_colorbar_enabled=True    # <-- 开启屏幕独立图例
        # )

        # ====== 修复开始 ======
        # 1. 修复 Warning: 使用 Matplotlib 最新 API 获取反转色谱的 RGB 数组
        reversed_jet_colors = mpl.colormaps['jet_r'](np.linspace(0, 1, 256))[:, :3]
        
        # 2. 修复 Error: 将数组保存为 1x256 像素的临时图片，供 Polyscope 读取
        cmap_img_path = "temp_jet_r.png"
        plt.imsave(cmap_img_path, reversed_jet_colors.reshape(1, 256, 3))
        
        # 让 Polyscope 从该图片加载自定义色谱
        ps.load_color_map("my_jet_reversed", cmap_img_path)
        # ====== 修复结束 ======

        # 4. 添加标量场，直接传入原始 values，图例数值彻底正确！
        ps_cloud.add_scalar_quantity(
            "reachability", 
            values,                           
            enabled=True, 
            cmap='my_jet_reversed',           # <--- 使用刚才加载的反转色谱
            vminmax=(0, 100),                 # <--- 绝对数值范围
            onscreen_colorbar_enabled=True    # <--- 显示屏幕图例
        )

        # 强制 Polyscope 将传入的值作为真实的绝对世界坐标尺寸（米）
        ps_cloud.set_radius(self.discretizer.l_c * 0.5, relative=False)
        ps.add_scene_slice_plane()

        print("--- Polyscope 已启动 ---")
        print("提示：在菜单选择 'View' -> 'Slicing Planes' 开启实时剖切")
        ps.show()

    def visualize_open3d(self, metric='D_o', threshold=1.0, 
                            slice_axis=None, slice_index=None):
            """
            使用 Open3D 展示能力图（体素网格）：支持剖面显示。
            
            Args:
                slice_axis (str): 剖切轴。可以是 'x', 'y', 或 'z'。默认为 None (显示全部)。
                slice_index (int): 剖切面在对应轴上的体素索引（即像素坐标）。
            """
            # 1. 提取数据
            data_matrix = self.D_o if metric == 'D_o' else self.D
            indices = np.argwhere(data_matrix >= threshold)
            
            if len(indices) == 0:
                print("Error: No voxels found above threshold.")
                return

            # --- 新增：剖面处理逻辑 ---
            if slice_axis and slice_index is not None:
                # 确定要过滤的轴的索引 (x=0, y=1, z=2)
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                if slice_axis not in axis_map:
                    print(f"[警告] 无效的剖切轴 '{slice_axis}'，请使用 'x', 'y' 或 'z'。将显示全部。")
                else:
                    axis_idx = axis_map[slice_axis]
                    
                    # 检查索引是否有效
                    max_index = data_matrix.shape[axis_idx]
                    if slice_index < 0 or slice_index >= max_index:
                        print(f"[警告] 剖切索引 {slice_index} 超出轴 '{slice_axis}' 的范围 (0-{max_index-1})。将显示全部。")
                    else:
                        # **核心：创建一个布尔掩码，只保留特定轴上具有特定索引的点**
                        mask = indices[:, axis_idx] == slice_index
                        indices = indices[mask] # 应用掩码，过滤点
                        
                        if len(indices) == 0:
                            print(f"在 '{slice_axis}' 轴上，体素坐标为 {slice_index} 的剖面没有任何点超过阈值。")
                            return
                        print(f"--- 剖切中: 已沿 '{slice_axis}' 轴在索引 {slice_index} 处生成剖面 ---")

            # --- 2-3. 坐标转换和颜色映射保持不变 ---
            # ... (保持原有的代码不变，它们会自动处理过滤后的 indices)
            # 使用你定义的转换逻辑
            offset = (1.0 - self.discretizer.n_c / 2.0)
            centers = (indices + offset) * self.discretizer.l_c - (self.discretizer.l_c / 2.0)
            
            # 应用颜色表
            values = data_matrix[indices[:, 0], indices[:, 1], indices[:, 2]]
            # 强制使用绝对映射：假设可达率的范围是 0 到 100
            # 如果你的数据最大值不是 100，可以将这里的 vmax 修改为实际理论最大值
            vmin = 0.0
            vmax = 100.0 
            normalized_values = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
            
            # 使用标准的 jet 颜色表
            cmap = plt.get_cmap('jet_r')
            colors = cmap(normalized_values)[:, :3]

            # --- 4. 转换并创建带有间隙的体素网格 (Mesh) ---
            voxel_size = self.discretizer.l_c * 0.8
            
            # 手动构建立方体的 8 个相对顶点
            v_rel = np.array([
                [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], 
                [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
                [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], 
                [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5]
            ]) * voxel_size

            # 立方体的 12 个三角面
            faces_rel = np.array([
                [0, 2, 1], [1, 2, 3], # 底面
                [4, 5, 6], [5, 7, 6], # 顶面
                [0, 1, 4], [1, 5, 4], # 前面
                [2, 6, 3], [3, 6, 7], # 后面
                [0, 4, 2], [2, 4, 6], # 左面
                [1, 3, 5], [3, 7, 5]  # 右面
            ])

            n_voxels = len(centers)
            # 预分配内存以实现极速计算
            vertices = np.zeros((n_voxels * 8, 3))
            triangles = np.zeros((n_voxels * 12, 3), dtype=np.int32)
            vertex_colors = np.zeros((n_voxels * 8, 3))

            # 向量化生成所有顶点和面
            for i in range(n_voxels):
                vertices[i*8:(i+1)*8] = v_rel + centers[i]
                triangles[i*12:(i+1)*12] = faces_rel + i * 8
                # 为该立方体的 8 个顶点赋予相同的颜色
                vertex_colors[i*8:(i+1)*8] = colors[i]

            # 创建 Open3D TriangleMesh 几何体
            custom_voxel_mesh = o3d.geometry.TriangleMesh()
            custom_voxel_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            custom_voxel_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            custom_voxel_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            custom_voxel_mesh.compute_vertex_normals() # 计算法线以确保光照正确
            
            # (注意：在后面的代码中，将 vis.add_geometry(voxel_grid) 
            #  改为 vis.add_geometry(custom_voxel_mesh) )

            # --- 5. 可视化配置保持不变 ---
            # ... (保持原有的代码不变)
            print(f"--- Open3D 启动中: 渲染 {len(centers)} 个体素 ---")
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Capability Map - VoxelGrid Section", width=1280, height=720)
            
            # 添加几何体时，添加体素网格而不是点云
            vis.add_geometry(custom_voxel_mesh)
            
            # 获取渲染设置 (体素渲染不需要调 point_size 了)
            opt = vis.get_render_option()
            if opt is not None:
                opt.background_color = np.asarray([0.1, 0.1, 0.1]) # 深灰色背景
            
            # 添加坐标轴辅助观察
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.discretizer.l_ws/4, origin=[0, 0, 0])
            vis.add_geometry(axes)
            
            vis.run()
            vis.destroy_window()


if __name__ == "__main__":
    # file_path = "./data/csm_fk_cmap_multi.npz"
    file_path = "./data/ur5_fk_cmap_multi.npz"
    # file_path = "./data/panda_fk_cmap_multi.npz"
    analyzer = CmapAnalyzer(filepath=file_path)
    
    # 可视化
    # analyzer.visualize_meshcat(robot_name='panda', metric='D', 
    #                           threshold=0.1, slice_axis='y', 
    #                           cut_half=False, alpha=0.3) 

    analyzer.visualize_polyscope(robot_name='ur5', metric='D', 
                              threshold=0.1, slice_axis='y', 
                              cut_half=False, alpha=0.3) 

    # analyzer.visualize_open3d(metric='D', threshold=0.1 )

