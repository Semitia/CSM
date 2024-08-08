import numpy as np
import matplotlib.pyplot as plt

def find_intersection(p1, d1, p2, d2):
    # 计算两个方向向量的叉乘
    cross_d1_d2 = np.cross(d1, d2)
    norm_cross_d1_d2 = np.linalg.norm(cross_d1_d2)

    # 如果叉乘的范数接近于零，说明射线平行或共线
    if norm_cross_d1_d2 < 1e-8:
        print("射线平行或共线，无交点")
        return None

    # 计算线性方程组的右侧常数项
    diff_p = p2 - p1

    # 计算方程组的解，即参数 t 和 s
    t = np.linalg.det([diff_p, d2, cross_d1_d2]) / norm_cross_d1_d2**2
    s = np.linalg.det([diff_p, d1, cross_d1_d2]) / norm_cross_d1_d2**2

    # 计算交点坐标
    intersection1 = p1 + t * d1
    intersection2 = p2 + s * d2

    # 验证交点是否相同（即射线相交）
    if np.allclose(intersection1, intersection2):
        return intersection1
    else:
        print("射线不相交")
        return None
    
    
class LineGenerator:
    def __init__(self):
        self.segments = []
        self.debug_info = []

    def add_hermite_curve(self, p0, p1, m0, m1, num_points=100):
        t = np.linspace(0, 1, num_points)
        h00 = (2 * t ** 3) - (3 * t ** 2) + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = (-2 * t ** 3) + (3 * t ** 2)
        h11 = t ** 3 - t ** 2

        curve_points = np.outer(h00, p0) + np.outer(h10, m0) + np.outer(h01, p1) + np.outer(h11, m1)
        self.segments.append((curve_points, p0, p1))

    def add_line(self, p0, p1, num_points=100):
        t = np.linspace(0, 1, num_points)
        line_points = np.outer(1 - t, p0) + np.outer(t, p1)
        self.segments.append((line_points, p0, p1))
    

    def add_arc(self, p0, p1, m0, m1, num_points=100):
        # Ensure inputs are float arrays
        p0 = np.array(p0, dtype=np.float64)
        p1 = np.array(p1, dtype=np.float64)
        m0 = np.array(m0, dtype=np.float64)
        m1 = np.array(m1, dtype=np.float64)

        # Calculate normal vector of the plane
        normal = np.cross(m0, m1)
        normal /= np.linalg.norm(normal)
        # print("normal: ", normal)

        r0_vec = np.cross(m0, normal)
        r1_vec = np.cross(m1, normal)
        center = find_intersection(p0, r0_vec, p1, r1_vec)

        if center is None:
            self.debug_info.append((p0, m0, r0_vec, p1, m1, r1_vec, normal))  # 保存调试信息
            print("Arc center not found")
            return


        # Calculate the radius
        radius = np.linalg.norm(center - p0)

        # Calculate angle between the two points with respect to the center
        v0 = (p0 - center) / np.linalg.norm(p0 - center)
        v1 = (p1 - center) / np.linalg.norm(p1 - center)
        angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0))

        # Create the arc points
        theta = np.linspace(0, angle, num_points)
        arc_points = np.zeros((num_points, 3))
        cross_v0_normal = np.cross(normal, v0)
        cross_v0_normal /= np.linalg.norm(cross_v0_normal)

        for i in range(num_points):
            point = (
                center
                + radius * np.cos(theta[i]) * v0
                + radius * np.sin(theta[i]) * cross_v0_normal
            )
            arc_points[i] = point

        self.segments.append((arc_points, p0, p1))


    def plot_segments(self):
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        # 预定义的颜色列表
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'black', 'purple', 'brown']
        
        for i, segment in enumerate(self.segments):
            points = segment[0]
            # 从颜色列表中循环选择颜色
            color = colors[i % len(colors)]  # 使用模运算确保不会超出列表范围
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2)
        
        for p0, m0, r0_vec, p1, m1, r1_vec, normal in self.debug_info:
            t = np.linspace(-1, 1, 100)
            line1 = p0 + np.outer(t, r0_vec)
            line2 = p1 + np.outer(t, r1_vec)
            line3 = p0 + np.outer(t, m0)
            line4 = p1 + np.outer(t, m1)
            ax.plot(line3[:, 0], line3[:, 1], line3[:, 2], 'r--')
            ax.plot(line4[:, 0], line4[:, 1], line4[:, 2], 'g--')
            ax.plot(line1[:, 0], line1[:, 1], line1[:, 2], 'r--')
            ax.plot(line2[:, 0], line2[:, 1], line2[:, 2], 'g--')
            # 绘制法平面
            # d = -np.dot(normal, p0)
            # xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
            # zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            # ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)
            # # 绘制法向量
            # ax.quiver(p0[0], p0[1], p0[2], normal[0], normal[1], normal[2], length=0.5, color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
        plt.title('Combined Line Segments in 3D')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Create an instance of LineGenerator
    line_gen = LineGenerator()

    # Example points and directions
    p0 = np.array([0, 0, 0], dtype=np.float64)
    m0 = np.array([0, 0, 1], dtype=np.float64)
    p1 = np.array([0.5, 0.5, 0.5*np.sqrt(2)], dtype=np.float64)
    m1 = np.array([1, 1, 0], dtype=np.float64)

    # Add an arc segment
    line_gen.add_arc(p0, p1, m0, m1)

    # Plot the segments
    line_gen.plot_segments()