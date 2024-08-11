import numpy as np
import matplotlib.pyplot as plt
from LineGenerator import LineGenerator

class Visualizer:
    def __init__(self):
        self.lg = LineGenerator()  # 将 LineGenerator 作为 Visualizer 的一个成员

    def plot(self, csm, ax):
        """
        使用 LineGenerator 绘制 CSM 的当前状态
        """
        init_pos = np.array([0, 0, 0, 1])
        init_ori = np.array([0, 0, 1])
        ax.clear()  # 清除之前的绘图

        self.lg.segments = []  # 重置 segments

        if csm.mode == 1:
            self.lg.add_arc(init_pos[:3], csm.end2_pos[:3], init_ori, csm.end2_ori)

        elif csm.mode == 2:
            self.lg.add_line(init_pos[:3], csm.base2_pos[:3])
            self.lg.add_arc(csm.base2_pos[:3], csm.end2_pos[:3], csm.base2_ori, csm.end2_ori)

        elif csm.mode == 3:
            self.lg.add_arc(init_pos[:3], csm.end1_pos[:3], init_ori, csm.end1_ori)
            self.lg.add_line(csm.end1_pos[:3], csm.base2_pos[:3])
            self.lg.add_arc(csm.base2_pos[:3], csm.end2_pos[:3], csm.base2_ori, csm.end2_ori)

        elif csm.mode == 4:
            self.lg.add_line(init_pos[:3], csm.base1_pos[:3])
            self.lg.add_arc(csm.base1_pos[:3], csm.end1_pos[:3], csm.base1_ori, csm.end1_ori)
            self.lg.add_line(csm.end1_pos[:3], csm.base2_pos[:3])
            self.lg.add_arc(csm.base2_pos[:3], csm.end2_pos[:3], csm.base2_ori, csm.end2_ori)

        # 使用 LineGenerator 绘制生成的线段
        for segment in self.lg.segments:
            points = segment[0]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
        ax.set_box_aspect([1, 1, 1])
        plt.title('Manipulator Movement')
        plt.grid(True)
