"""
Module: line_generator.py
Description: Helper class to generate 3D lines and arcs for visualization purposes.
"""
import numpy as np
import matplotlib.pyplot as plt

def find_intersection(p1, d1, p2, d2):
    cross_d1_d2 = np.cross(d1, d2)
    norm_cross_d1_d2 = np.linalg.norm(cross_d1_d2)
    if norm_cross_d1_d2 < 1e-8:
        print("射线平行或共线，无交点")
        return None
    diff_p = p2 - p1
    t = np.linalg.det([diff_p, d2, cross_d1_d2]) / norm_cross_d1_d2**2
    s = np.linalg.det([diff_p, d1, cross_d1_d2]) / norm_cross_d1_d2**2
    intersection1 = p1 + t * d1
    intersection2 = p2 + s * d2
    if np.allclose(intersection1, intersection2):
        return intersection1
    else:
        print("射线不相交")
        return None


class LineGenerator:
    def __init__(self):
        self.segments = []
        self.debug_info = []
        self.colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'black', 'purple', 'brown']

    def add_line(self, p0, p1, num_points=3):
        t = np.linspace(0, 1, num_points)
        line_points = np.outer(1 - t, p0) + np.outer(t, p1)
        self.segments.append((line_points, p0, p1))

    def add_hermite_curve(self, p0, p1, m0, m1, num_points=15):
        t = np.linspace(0, 1, num_points)
        h00 = (2 * t ** 3) - (3 * t ** 2) + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = (-2 * t ** 3) + (3 * t ** 2)
        h11 = t ** 3 - t ** 2
        curve_points = np.outer(h00, p0) + np.outer(h10, m0) + np.outer(h01, p1) + np.outer(h11, m1)
        self.segments.append((curve_points, p0, p1))

    def add_arc(self, p0, p1, m0, m1, num_points=15):
        p0 = np.array(p0, dtype=np.float64)
        p1 = np.array(p1, dtype=np.float64)
        m0 = np.array(m0, dtype=np.float64)
        m1 = np.array(m1, dtype=np.float64)
        if np.allclose(np.cross(m0, m1), 0):
            self.add_line(p0, p1, num_points)
            return
        normal = np.cross(m0, m1)
        normal /= np.linalg.norm(normal)
        r0_vec = np.cross(m0, normal)
        r1_vec = np.cross(m1, normal)
        center = find_intersection(p0, r0_vec, p1, r1_vec)
        if center is None:
            self.debug_info.append((p0, m0, r0_vec, p1, m1, r1_vec))
            print("Arc center not found")
            return
        radius = np.linalg.norm(center - p0)
        v0 = (p0 - center) / np.linalg.norm(p0 - center)
        v1 = (p1 - center) / np.linalg.norm(p1 - center)
        angle = np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0))
        theta = np.linspace(0, angle, num_points)
        arc_points = np.zeros((num_points, 3))
        cross_v0_normal = np.cross(normal, v0)
        cross_v0_normal /= np.linalg.norm(cross_v0_normal)
        for i in range(num_points):
            arc_points[i] = center + radius * np.cos(theta[i]) * v0 + radius * np.sin(theta[i]) * cross_v0_normal
        self.segments.append((arc_points, p0, p1))

    def draw(self, ax, reverse_color=False):
        colors = self.colors[::-1] if reverse_color else self.colors
        for i, segment in enumerate(self.segments):
            color = colors[(len(self.segments) - 1 - i) % len(colors)]
            ax.plot(segment[0][:, 0], segment[0][:, 1], segment[0][:, 2], color=color, linewidth=2)

    def plot_segments(self):
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        self.draw(ax)
        for p0, m0, r0_vec, p1, m1, r1_vec in self.debug_info:
            t = np.linspace(-1, 1, 100)
            for line, color in [(p0 + np.outer(t, m0), 'r--'), (p1 + np.outer(t, m1), 'g--'),
                                 (p0 + np.outer(t, r0_vec), 'r--'), (p1 + np.outer(t, r1_vec), 'g--')]:
                ax.plot(line[:, 0], line[:, 1], line[:, 2], color)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 1])
        plt.title('Combined Line Segments in 3D')
        plt.grid(True)
        plt.show()
