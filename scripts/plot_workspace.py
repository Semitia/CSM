"""
Module: plot_workspace.py
Description: Script to plot the generated workspace points in 3D.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ===== 配置 =====
json_path = "./data/workspace_data.json"
RENDER_MODE = "wire"   # ← 'wire' | 'surface' | 'both'
SEPARATE_PLOTS = True  # ← 是否分成四个子图绘制

MODE_COLORS = {1: "#E41A1C", 2: "#F2DF9588", 3: "#9ACDE5B1", 4: "#C7C7C7B9"}
MODE_LABELS = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
SURF_ALPHA = 0.30
WIRE_ALPHA = 0.9
WIRE_LW = 0.6

def load_points_by_mode(path):
    with open(path, "r") as f:
        data = json.load(f)
    by_mode = {}
    for m in (1, 2, 3, 4):
        pts = [np.asarray(d["pose"][:3], float) for d in data if int(d["mode"]) == m]
        by_mode[m] = np.vstack(pts) if pts else np.empty((0, 3))
        print(f"Mode {m}: {by_mode[m].shape[0]} points")
    return by_mode

def _unique_edges_from_tris(tris):
    edges = set()
    for a, b, c in tris:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return edges

def draw_shell(ax, points, color, label=None, mode="surface"):
    P = np.unique(points.round(9), axis=0)
    if P.shape[0] < 4:
        ax.scatter(P[:,0], P[:,1], P[:,2], s=3, color=color, label=label)
        return

    hull = ConvexHull(P, qhull_options="QJ")
    tris = hull.simplices

    if mode in ("surface", "both"):
        surf = ax.plot_trisurf(
            P[:,0], P[:,1], P[:,2],
            triangles=tris,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            alpha=SURF_ALPHA,
            color=color,
            shade=False
        )
        try:
            surf.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass

    if mode in ("wire", "both"):
        edges = _unique_edges_from_tris(tris)
        segments = [(P[i], P[j]) for (i, j) in edges]
        lc = Line3DCollection(segments, colors=color, linewidths=WIRE_LW,
                              alpha=WIRE_ALPHA if mode == "wire" else 0.6)
        ax.add_collection3d(lc)

    if label:
        ax.plot([], [], [], color=color, label=label)

def main():
    by_mode = load_points_by_mode(json_path)

    if SEPARATE_PLOTS:
        fig = plt.figure(figsize=(10, 10))
        axes = [fig.add_subplot(2, 2, i+1, projection="3d") for i in range(4)]
        # ===== 统一范围计算 =====
        all_pts = np.vstack([pts for pts in by_mode.values() if pts.size])
        if all_pts.size:
            pad = 0.05 * (all_pts.max() - all_pts.min())
            mins = all_pts.min(axis=0) - pad
            maxs = all_pts.max(axis=0) + pad
        for m, ax in zip((1, 2, 3, 4), axes):
            pts = by_mode[m]
            if pts.size == 0:
                continue
            draw_shell(ax, pts, MODE_COLORS[m], label=MODE_LABELS[m], mode=RENDER_MODE)
            ax.set_title(MODE_LABELS[m])
            ax.set_box_aspect([1, 1, 1])
            ax.grid(False)
            ax.set_proj_type('ortho')
            ax.view_init(elev=0, azim=90)
            # === 每个子图都用相同的坐标范围 ===
            if all_pts.size:
                ax.set_xlim(mins[0], maxs[0])
                ax.set_ylim(mins[1], maxs[1])
                ax.set_zlim(mins[2], maxs[2])
            # 背景透明优化
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((1, 1, 1, 0))
                axis.pane.fill = False
        plt.tight_layout()


    else:
        all_pts = np.vstack([pts for pts in by_mode.values() if pts.size])
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        for m in (1, 2, 3, 4):
            pts = by_mode[m]
            if pts.size == 0:
                continue
            draw_shell(ax, pts, MODE_COLORS[m], label=MODE_LABELS[m], mode=RENDER_MODE)

        if all_pts.size:
            pad = 0.05 * (all_pts.max() - all_pts.min())
            mins = all_pts.min(axis=0) - pad
            maxs = all_pts.max(axis=0) + pad
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

        ax.legend(loc="upper right", frameon=False)
        ax.set_proj_type('ortho')
        ax.view_init(elev=0, azim=90)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.fill = False
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
