import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ===== 配置 =====
json_path = "./data/workspace_data.json"  # ← 改成你的路径
WIRE_ONLY = False                  # 只画线框：True；半透明表面：False
MODE_COLORS = {1: "#E41A1C", 2: "#EED87D92", 3: "#9ACDE5D0", 4: "#C7C7C7AD"}  # C1~C4
MODE_LABELS = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}

def load_points_by_mode(path):
    with open(path, "r") as f:
        data = json.load(f)
    by_mode = {}
    for m in (1, 2, 3, 4):
        pts = [np.asarray(d["pose"][:3], float) for d in data if int(d["mode"]) == m]
        by_mode[m] = np.vstack(pts) if pts else np.empty((0, 3))
        print(f"Mode {m}: {by_mode[m].shape[0]} points")
    return by_mode

def draw_shell(ax, points, color, label=None, wire_only=False):
    # 去重避免 Qhull 数值问题；不足 4 点时退化成散点
    P = np.unique(points.round(9), axis=0)
    if P.shape[0] < 4:
        ax.scatter(P[:,0], P[:,1], P[:,2], s=3, color=color, label=label)
        return

    hull = ConvexHull(P, qhull_options="QJ")

    if not wire_only:
        # 半透明表面 + 轻边线
        surf = ax.plot_trisurf(P[:,0], P[:,1], P[:,2],
                               triangles=hull.simplices,
                               linewidth=0.35,
                               edgecolor=(0, 0, 0, 0.15),
                               antialiased=True,
                               alpha=0.30,
                               color=color,
                               shade=False)
        if label:
            ax.plot([], [], [], color=color, label=label)
        return

    # ---- 仅线框：从凸包三角面提取“唯一边”并绘制 ----
    tris = hull.simplices
    edges = set()
    for a, b, c in tris:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    segments = [(P[i], P[j]) for (i, j) in edges]
    lc = Line3DCollection(segments, colors=color, linewidths=0.6, alpha=0.9)
    ax.add_collection3d(lc)

    # 让坐标轴包含这些线段（有时 3D 集合不会自动扩展范围）
    mins = P.min(axis=0); maxs = P.max(axis=0)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])

    if label:
        ax.plot([], [], [], color=color, label=label)


def main():
    by_mode = load_points_by_mode(json_path)
    # 统计全局范围用于设定轴限
    all_pts = np.vstack([pts for pts in by_mode.values() if pts.size])
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # 逐模式绘制
    for m in (1, 2, 3, 4):
        pts = by_mode[m]
        if pts.size == 0:
            continue
        draw_shell(ax, pts, MODE_COLORS[m], label=MODE_LABELS[m], wire_only=WIRE_ONLY)

    # 轴/视觉优化
    if all_pts.size:
        pad = 0.05 * (all_pts.max() - all_pts.min())
        mins = all_pts.min(axis=0) - pad
        maxs = all_pts.max(axis=0) + pad
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    # 去掉背景面
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.fill = False

    ax.legend(loc="upper right", frameon=False)
    # ax.view_init(elev=22, azim=-45)     # 斜二测视角
    # ax.view_init(elev=0, azim=0)        # 沿x轴观察
    ax.view_init(elev=0, azim=90)       # 沿y轴观察
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
