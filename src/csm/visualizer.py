"""
Module: visualizer.py
Description: Visualizer class using LineGenerator to plot the CSM model and its movement.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .line_generator import LineGenerator


class Visualizer:
    def __init__(self, body_radius=0.0017, tendon_radius_ratio=0.72, disk_spacing=0.004,
                 default_render_mode="detailed"):
        self.body_radius = body_radius
        self.tendon_radius_ratio = tendon_radius_ratio
        self.disk_spacing = disk_spacing
        self.default_render_mode = default_render_mode
        self.palette = {
            "backbone": "#1f2937",
            "seg1": "#d97706",
            "seg2": "#1d4ed8",
            "straight": "#4b5563",
            "disk": "#9ca3af",
            "tool": "#0f766e",
        }
        self.tendon_colors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]

    def _robot_radius(self, csm):
        nonzero_lengths = [x for x in (csm.L_10, csm.L_20, csm.L_r0, csm.L_tool) if x > 0]
        if not nonzero_lengths:
            return self.body_radius
        return float(np.clip(min(nonzero_lengths) * 0.2, 8e-4, 1.7e-3))

    def _draw_disk(self, ax, center, rotation, radius, color, alpha=0.35, resolution=24):
        angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
        local = np.vstack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros_like(angles),
        ])
        world = center[:, None] + rotation @ local
        verts = [list(map(tuple, world.T))]
        patch = Poly3DCollection(verts, facecolors=color, edgecolors=color, linewidths=0.7, alpha=alpha)
        ax.add_collection3d(patch)

    def _draw_tool(self, ax, tool, radius):
        length = tool["length"]
        if length <= 0:
            return
        R = tool["rotation"]
        p = tool["start"]
        base_radius = radius * 0.82
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        base_circle_local = np.vstack([
            base_radius * np.cos(angles),
            base_radius * np.sin(angles),
            np.zeros_like(angles),
        ])
        base_circle_world = (p[:, None] + R @ base_circle_local).T
        tip_world = p + R @ np.array([0.0, 0.0, length])
        tool_color = self.palette["tool"]

        closed_circle = np.vstack([base_circle_world, base_circle_world[0]])
        ax.plot(
            closed_circle[:, 0],
            closed_circle[:, 1],
            closed_circle[:, 2],
            color=tool_color,
            linewidth=1.0,
            alpha=0.9,
        )

        for idx in range(0, len(base_circle_world), 2):
            edge = np.vstack([base_circle_world[idx], tip_world])
            ax.plot(
                edge[:, 0],
                edge[:, 1],
                edge[:, 2],
                color=tool_color,
                linewidth=1.0,
                alpha=0.9,
            )

    def _sample_disk_frames(self, segment):
        length = segment["length"]
        if segment["kind"] != "arc" or length <= 0:
            return []
        disk_count = max(2, int(np.floor(length / self.disk_spacing)) + 1)
        idx = np.linspace(0, len(segment["points"]) - 1, disk_count).round().astype(int)
        idx = np.unique(np.clip(idx, 0, len(segment["points"]) - 1))
        return [(segment["points"][i], segment["rotations"][i]) for i in idx]

    def _plot_simple(self, csm, ax, reverse_color=False):
        init_pos = np.array([0, 0, 0, 1], dtype=float)
        init_ori = np.array([0, 0, 1], dtype=float)
        lg = LineGenerator()

        if csm.mode == 1:
            lg.add_arc(init_pos[:3], csm.end2_pos[:3], init_ori, csm.end2_ori)
        elif csm.mode == 2:
            lg.add_line(init_pos[:3], csm.base2_pos[:3])
            lg.add_arc(csm.base2_pos[:3], csm.end2_pos[:3], csm.base2_ori, csm.end2_ori)
        elif csm.mode == 3:
            lg.add_arc(init_pos[:3], csm.end1_pos[:3], init_ori, csm.end1_ori)
            lg.add_line(csm.end1_pos[:3], csm.base2_pos[:3])
            lg.add_arc(csm.base2_pos[:3], csm.end2_pos[:3], csm.base2_ori, csm.end2_ori)
        elif csm.mode == 4:
            lg.add_line(init_pos[:3], csm.base1_pos[:3])
            lg.add_arc(csm.base1_pos[:3], csm.end1_pos[:3], csm.base1_ori, csm.end1_ori)
            lg.add_line(csm.end1_pos[:3], csm.base2_pos[:3])
            lg.add_arc(csm.base2_pos[:3], csm.end2_pos[:3], csm.base2_ori, csm.end2_ori)

        if csm.L_tool > 0:
            lg.add_line(csm.end2_pos[:3], csm.tool_pos)

        lg.draw(ax, reverse_color=reverse_color)

    def _plot_detailed(self, csm, ax, reverse_color=False):
        ax.clear()
        geometry = csm.get_visualization_segments()
        radius = self._robot_radius(csm)
        tendon_radius = radius * self.tendon_radius_ratio
        disk_radius = radius * 1.25
        tendon_offsets = [
            np.array([ tendon_radius, 0.0, 0.0]),
            np.array([ 0.0, tendon_radius, 0.0]),
            np.array([-tendon_radius, 0.0, 0.0]),
            np.array([ 0.0,-tendon_radius, 0.0]),
        ]

        centerline = []
        for seg_idx, segment in enumerate(geometry["segments"]):
            points = segment["points"]
            rotations = segment["rotations"]
            if seg_idx > 0:
                points = points[1:]
                rotations = rotations[1:]
            centerline.append(points)

            segment_color = self.palette["seg1"] if segment["label"] == "seg1" else \
                            self.palette["seg2"] if segment["label"] == "seg2" else \
                            self.palette["straight"]

            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=segment_color, linewidth=3.0, alpha=0.95)

            for tendon_idx, offset in enumerate(tendon_offsets):
                tendon_points = points + np.einsum("nij,j->ni", rotations, offset)
                ax.plot(
                    tendon_points[:, 0],
                    tendon_points[:, 1],
                    tendon_points[:, 2],
                    color=self.tendon_colors[tendon_idx],
                    linewidth=1.2,
                    alpha=0.9 if segment["kind"] == "arc" else 0.55,
                )

            for center, rotation in self._sample_disk_frames(segment):
                self._draw_disk(ax, center, rotation, disk_radius, self.palette["disk"])

            if segment["label"] == "rigid" and segment["length"] > 0:
                self._draw_disk(ax, segment["T_start"][:3, 3], segment["T_start"][:3, :3], disk_radius, segment_color, alpha=0.22)
                self._draw_disk(ax, segment["T_end"][:3, 3], segment["T_end"][:3, :3], disk_radius, segment_color, alpha=0.22)

        if centerline:
            merged = np.vstack(centerline)
            ax.plot(
                merged[:, 0],
                merged[:, 1],
                merged[:, 2],
                color=self.palette["backbone"],
                linewidth=1.4,
                linestyle="--",
                alpha=0.8 if not reverse_color else 0.55,
            )

        self._draw_tool(ax, geometry["tool"], radius * 1.05)

    def plot(self, csm, ax, reverse_color=False, render_mode=None):
        ax.clear()
        mode = self.default_render_mode if render_mode is None else render_mode
        if mode not in {"simple", "detailed"}:
            raise ValueError(f"Unsupported render_mode: {mode}")
        if mode == "simple":
            self._plot_simple(csm, ax, reverse_color=reverse_color)
        else:
            self._plot_detailed(csm, ax, reverse_color=reverse_color)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 1])
        ax.set_box_aspect([1, 1, 1])
        plt.title('Manipulator Movement')
        plt.grid(True)
