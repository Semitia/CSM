"""
Module: plot_workspace.py
Description: Script to plot the generated workspace points in 3D.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LightSource

# ===== 配置 =====
json_path = "./data/workspace_data_csm_cfg_3.4mm_uniform.json"
RENDER_MODE = "surface"   # ← 'wire' | 'surface' | 'both' | 'smooth_surface' | 'cloud'
SEPARATE_PLOTS = True  # ← 是否分成四个子图绘制
SURFACE_GEOMETRY = "alpha_shape"  # ← 'convex_hull' | 'alpha_shape'
ALPHA_RADIUS = None               # None 时自动估计
ALPHA_MAX_POINTS = 4000           # alpha shape 前的最大点数，避免 3D Delaunay 爆炸

MODE_COLORS = {1: "#E41A1C", 2: "#F2DF95", 3: "#9ACDE5", 4: "#C7C7C7"}
MODE_LABELS = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
SURF_ALPHA = 0.35
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


def _downsample_points_for_alpha(points, max_points):
    if points.shape[0] <= max_points:
        return points

    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-12)
    voxel_size = (np.prod(spans) / max_points) ** (1.0 / 3.0)
    voxel_size = max(voxel_size, 1e-12)

    while True:
        grid_idx = np.floor((points - mins) / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(grid_idx, axis=0, return_index=True)
        reduced = points[np.sort(unique_idx)]
        if reduced.shape[0] <= max_points or voxel_size >= spans.max():
            return reduced
        voxel_size *= 1.15


def _tetra_circumsphere_radii(tetra):
    a = tetra[:, 0, :]
    b = tetra[:, 1, :]
    c = tetra[:, 2, :]
    d = tetra[:, 3, :]

    m = 2.0 * np.stack((b - a, c - a, d - a), axis=1)
    rhs = np.stack((
        np.sum(b * b, axis=1) - np.sum(a * a, axis=1),
        np.sum(c * c, axis=1) - np.sum(a * a, axis=1),
        np.sum(d * d, axis=1) - np.sum(a * a, axis=1),
    ), axis=1)

    radii = np.full(tetra.shape[0], np.inf, dtype=float)
    try:
        centers = np.linalg.solve(m, rhs[..., np.newaxis]).squeeze(-1)
        radii = np.linalg.norm(centers - a, axis=1)
    except np.linalg.LinAlgError:
        for i in range(tetra.shape[0]):
            try:
                center = np.linalg.solve(m[i], rhs[i])
                radii[i] = np.linalg.norm(center - a[i])
            except np.linalg.LinAlgError:
                continue
    return radii


def _estimate_alpha_radius(points):
    if points.shape[0] < 8:
        return np.inf
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(8, points.shape[0]))
    kth = distances[:, -1]
    return 2.5 * np.median(kth)


def _alpha_shape_mesh(points, alpha_radius=None, max_points=4000):
    reduced = _downsample_points_for_alpha(points, max_points)
    if reduced.shape[0] < 4:
        return reduced, np.empty((0, 3), dtype=int)

    delaunay = Delaunay(reduced, qhull_options="QJ")
    tetra_idx = delaunay.simplices
    tetra = reduced[tetra_idx]
    radii = _tetra_circumsphere_radii(tetra)
    alpha = _estimate_alpha_radius(reduced) if alpha_radius is None else alpha_radius
    keep = tetra_idx[radii <= alpha]

    if keep.size == 0:
        hull = ConvexHull(reduced, qhull_options="QJ")
        return reduced, hull.simplices

    faces = np.concatenate((
        keep[:, [0, 1, 2]],
        keep[:, [0, 1, 3]],
        keep[:, [0, 2, 3]],
        keep[:, [1, 2, 3]],
    ), axis=0)
    faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    boundary = unique_faces[counts == 1]

    if boundary.size == 0:
        hull = ConvexHull(reduced, qhull_options="QJ")
        return reduced, hull.simplices

    return reduced, boundary


def _build_surface_mesh(points, geometry_mode):
    if geometry_mode == "alpha_shape":
        return _alpha_shape_mesh(points, alpha_radius=ALPHA_RADIUS, max_points=ALPHA_MAX_POINTS)
    if geometry_mode == "convex_hull":
        hull = ConvexHull(points, qhull_options="QJ")
        return points, hull.simplices
    raise ValueError(f"Unsupported surface geometry: {geometry_mode}")

def draw_shell(ax, points, color, label=None, mode="surface"):
    P = np.unique(points.round(9), axis=0)

    if mode == "cloud":
        ax.scatter(
            P[:, 0],
            P[:, 1],
            P[:, 2],
            c=color,
            s=0.5,
            alpha=0.03,
            edgecolors="none",
            label=label,
        )
        return

    if P.shape[0] < 4:
        ax.scatter(P[:,0], P[:,1], P[:,2], s=3, color=color, label=label)
        return

    surface_points, tris = _build_surface_mesh(P, SURFACE_GEOMETRY)
    if tris.size == 0:
        ax.scatter(P[:,0], P[:,1], P[:,2], s=3, color=color, label=label)
        return

    if mode in ("surface", "both"):
        surf = ax.plot_trisurf(
            surface_points[:,0], surface_points[:,1], surface_points[:,2],
            triangles=tris,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            alpha=SURF_ALPHA,
            color=color,
            shade=False
        )
    elif mode == "smooth_surface":
        ls = LightSource(azdeg=30, altdeg=60)
        surf = ax.plot_trisurf(
            surface_points[:,0], surface_points[:,1], surface_points[:,2],
            triangles=tris,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            alpha=SURF_ALPHA,
            color=color,
            shade=True,
            lightsource=ls,
        )

    if mode in ("surface", "both", "smooth_surface"):
        try:
            surf.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass

    if mode in ("wire", "both"):
        edges = _unique_edges_from_tris(tris)
        segments = [(surface_points[i], surface_points[j]) for (i, j) in edges]
        lc = Line3DCollection(segments, colors=color, linewidths=WIRE_LW,
                              alpha=WIRE_ALPHA if mode == "wire" else 0.6)
        ax.add_collection3d(lc)

    if label and mode != "cloud":
        ax.plot([], [], [], color=color, label=label)

def main():
    by_mode = load_points_by_mode(json_path)

    if SEPARATE_PLOTS:
        fig = plt.figure(figsize=(10, 10))
        axes = [fig.add_subplot(2, 2, i+1, projection="3d") for i in range(4)]
        # ===== 统一范围计算 =====
        all_pts = np.vstack([pts for pts in by_mode.values() if pts.size])
        for m, ax in zip((1, 2, 3, 4), axes):
            pts = by_mode[m]
            if pts.size == 0:
                continue
            draw_shell(ax, pts, MODE_COLORS[m], label=MODE_LABELS[m], mode=RENDER_MODE)
            ax.set_title(MODE_LABELS[m])
            ax.set_box_aspect([1, 1, 1])
            # ax.grid(False)
            # ax.set_proj_type('ortho')
            # ax.view_init(elev=0, azim=90)
            # === 每个子图都用相同的坐标范围 ===
            if all_pts.size:
                # 计算数据范围并设置相等的轴范围以保持 1:1:1 比例
                data_range = all_pts.max() - all_pts.min()
                center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
                max_range = data_range.max() * 1.1 / 2  # 10% 留白
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[1] - max_range, center[1] + max_range)
                ax.set_zlim(center[2] - max_range, center[2] + max_range)
            # 背景透明优化
            # for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            #     axis.pane.set_facecolor((1, 1, 1, 0))
            #     axis.pane.fill = False
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
            # 计算数据范围并设置相等的轴范围以保持 1:1:1 比例
            data_range = all_pts.max() - all_pts.min()
            center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
            max_range = data_range.max() * 1.1 / 2  # 10% 留白
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=False)
        # ax.set_proj_type('ortho')
        # ax.view_init(elev=0, azim=90)
        # ax.grid(False)
        # for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        #     axis.pane.set_facecolor((1, 1, 1, 0))
        #     axis.pane.fill = False
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
