"""
Module: plot_workspace.py
Description: Plot translation workspace in a paper-like style from NPZ data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.ndimage import binary_fill_holes, gaussian_filter, label
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from skimage.measure import marching_cubes

NPZ_PATH = Path("./data/workspace_data_csm_cfg_3.4mm_uniform.npz")
TARGET_FIELD = "all_points"
OUTPUT_FIGURE = None

VOXEL_GRID_N = 112
PADDING_MM = 6.0
SMOOTH_SIGMA = 1.2
REACH_ISO_LEVEL = 0.42
UNREACH_ISO_LEVEL = 0.50
ALPHA_NEIGHBOR_K = 10
REACH_ALPHA_SCALE = 1.35
KEEP_LARGEST_UNREACHABLE = True
MIN_COMPONENT_RATIO = 0.015

REACH_COLOR = "#b8dfb4"
REACH_ALPHA = 0.34
UNREACH_COLOR = "#c69a74"
UNREACH_ALPHA = 0.42
EDGE_ALPHA = 0.0
POINT_COLOR = "#3b82f6"
BACKGROUND = "white"

VIEW_ELEV = 18
VIEW_AZIM = -38
SHOW_AXES = True

ALPHA_FACE_BATCH = 1024
ALPHA_QUERY_BATCH = 8192


def load_points(npz_path, field_name):
    payload = np.load(npz_path)
    if field_name not in payload:
        raise KeyError(f"Field '{field_name}' not found in {npz_path}")
    pts = np.asarray(payload[field_name], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Field '{field_name}' must have shape (N, 3), got {pts.shape}")
    return pts


def convert_points_to_mm(points):
    span = points.max(axis=0) - points.min(axis=0)
    scale = 1000.0 if np.max(np.abs(points)) < 1.0 and np.max(span) < 1.0 else 1.0
    return points * scale, scale


def unique_points(points, decimals=6):
    return np.unique(np.round(points, decimals=decimals), axis=0)


def padded_bounds(points, padding_mm):
    mins = points.min(axis=0) - padding_mm
    maxs = points.max(axis=0) + padding_mm
    return mins, maxs


def build_grid(points, grid_n, padding_mm):
    mins, maxs = padded_bounds(points, padding_mm)
    x = np.linspace(mins[0], maxs[0], grid_n)
    y = np.linspace(mins[1], maxs[1], grid_n)
    z = np.linspace(mins[2], maxs[2], grid_n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    origin = np.array([x[0], y[0], z[0]], dtype=float)
    return X, Y, Z, spacing, origin


def estimate_reach_alpha(points, k=ALPHA_NEIGHBOR_K, scale=REACH_ALPHA_SCALE):
    if points.shape[0] < 8:
        return np.inf
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(k, points.shape[0]))
    kth = distances[:, -1]
    return float(scale * np.median(kth))


def tetra_circumsphere_radii(tetra):
    a = tetra[:, 0, :]
    b = tetra[:, 1, :]
    c = tetra[:, 2, :]
    d = tetra[:, 3, :]

    m = 2.0 * np.stack((b - a, c - a, d - a), axis=1)
    rhs = np.stack(
        (
            np.sum(b * b, axis=1) - np.sum(a * a, axis=1),
            np.sum(c * c, axis=1) - np.sum(a * a, axis=1),
            np.sum(d * d, axis=1) - np.sum(a * a, axis=1),
        ),
        axis=1,
    )

    radii = np.full(tetra.shape[0], np.inf, dtype=float)
    for idx in range(tetra.shape[0]):
        try:
            center = np.linalg.solve(m[idx], rhs[idx])
            radii[idx] = np.linalg.norm(center - a[idx])
        except np.linalg.LinAlgError:
            continue
    return radii


def build_alpha_boundary_faces(points, alpha_radius):
    if points.shape[0] < 4:
        return np.empty((0, 3), dtype=int)

    tetra_idx = Delaunay(points, qhull_options="QJ").simplices
    tetra = points[tetra_idx]
    radii = tetra_circumsphere_radii(tetra)
    keep = tetra_idx[radii <= alpha_radius]
    if keep.size == 0:
        return np.empty((0, 3), dtype=int)

    faces = np.concatenate(
        (
            keep[:, [0, 1, 2]],
            keep[:, [0, 1, 3]],
            keep[:, [0, 2, 3]],
            keep[:, [1, 2, 3]],
        ),
        axis=0,
    )
    faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    return unique_faces[counts == 1]


def points_in_hull(query_points, hull_points):
    if hull_points.shape[0] < 4:
        return np.zeros(query_points.shape[0], dtype=bool)
    delaunay = Delaunay(hull_points, qhull_options="QJ")
    return delaunay.find_simplex(query_points) >= 0


def alpha_shape_mask(points, query_points, alpha_radius):
    boundary_faces = build_alpha_boundary_faces(points, alpha_radius)
    if boundary_faces.size == 0:
        return points_in_hull(query_points, points)

    hull_mask = points_in_hull(query_points, points)
    if not np.any(hull_mask):
        return hull_mask

    boundary = points[boundary_faces]
    tri_a = boundary[:, 0, :]
    tri_b = boundary[:, 1, :]
    tri_c = boundary[:, 2, :]

    selected = query_points[hull_mask]
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    eps = 1e-9
    crossings = np.zeros(selected.shape[0], dtype=np.int32)

    for start in range(0, boundary.shape[0], ALPHA_FACE_BATCH):
        stop = min(start + ALPHA_FACE_BATCH, boundary.shape[0])
        a = tri_a[start:stop]
        b = tri_b[start:stop]
        c = tri_c[start:stop]
        edge1 = b - a
        edge2 = c - a
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        det = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(det) > eps
        if not np.any(valid):
            continue

        a = a[valid]
        edge1 = edge1[valid]
        edge2 = edge2[valid]
        h = h[valid]
        inv_det = (1.0 / det[valid]).astype(selected.dtype, copy=False)
        ray_cross = np.broadcast_to(direction, edge1.shape)

        for query_start in range(0, selected.shape[0], ALPHA_QUERY_BATCH):
            query_stop = min(query_start + ALPHA_QUERY_BATCH, selected.shape[0])
            query_chunk = selected[query_start:query_stop]

            s = query_chunk[:, None, :] - a[None, :, :]
            u = np.einsum("qti,ti->qt", s, h) * inv_det[None, :]
            qvec = np.cross(s, edge1[None, :, :])
            v = np.einsum("ti,qti->qt", edge2, qvec) * inv_det[None, :]
            t = np.einsum("ti,qti->qt", ray_cross, qvec) * inv_det[None, :]

            hits = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps) & (t > eps)
            crossings[query_start:query_stop] += np.count_nonzero(hits, axis=1)

    inside = (crossings % 2) == 1
    mask = np.zeros(query_points.shape[0], dtype=bool)
    mask[hull_mask] = inside
    return mask


def largest_components(mask, min_ratio=MIN_COMPONENT_RATIO, keep_largest_only=False):
    labeled, count = label(mask)
    if count == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    if keep_largest_only:
        keep = int(np.argmax(component_sizes))
        return labeled == keep

    max_size = component_sizes.max()
    keep_labels = np.flatnonzero(component_sizes >= max_size * min_ratio)
    if keep_labels.size == 0:
        keep_labels = np.array([int(np.argmax(component_sizes))])
    return np.isin(labeled, keep_labels)


def smooth_mask(mask, sigma):
    return gaussian_filter(mask.astype(float), sigma=sigma)


def mesh_from_scalar(scalar, spacing, origin, iso_level):
    if scalar.max() <= iso_level:
        return None, None
    verts, faces, _, _ = marching_cubes(scalar, level=iso_level, spacing=spacing)
    verts = verts + origin
    return verts, faces


def compute_equal_limits(points, padding_mm):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * np.max(maxs - mins) + padding_mm
    return np.array(
        [
            center[0] - half_range,
            center[0] + half_range,
            center[1] - half_range,
            center[1] + half_range,
            center[2] - half_range,
            center[2] + half_range,
        ],
        dtype=float,
    )


def plot_mesh(ax, vertices, faces, color, alpha):
    tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles=faces)
    surf = ax.plot_trisurf(
        tri,
        vertices[:, 2],
        color=color,
        alpha=alpha,
        linewidth=0.0,
        edgecolor=(0, 0, 0, EDGE_ALPHA),
        antialiased=True,
        shade=False,
    )
    try:
        surf.set_edgecolor((0, 0, 0, EDGE_ALPHA))
    except Exception:
        pass


def style_axes(ax, points):
    ax.set_box_aspect([1, 1, 1])
    lims = compute_equal_limits(points, PADDING_MM)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_zlim(lims[4], lims[5])
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    if SHOW_AXES:
        ax.set_xlabel("X Axis (mm)")
        ax.set_ylabel("Y Axis (mm)")
        ax.set_zlabel("Z Axis (mm)")
        ax.grid(True, alpha=0.25)
    else:
        ax.set_axis_off()

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))


def build_workspace_shells(points_mm):
    X, Y, Z, spacing, origin = build_grid(points_mm, VOXEL_GRID_N, PADDING_MM)
    query_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    alpha_radius = estimate_reach_alpha(points_mm)
    reach_mask = alpha_shape_mask(points_mm, query_points, alpha_radius).reshape(X.shape)
    reach_mask = binary_fill_holes(reach_mask)

    outer_mask = points_in_hull(query_points, points_mm).reshape(X.shape)
    outer_mask = binary_fill_holes(outer_mask)

    unreach_mask = outer_mask & ~reach_mask
    unreach_mask = largest_components(
        unreach_mask,
        min_ratio=MIN_COMPONENT_RATIO,
        keep_largest_only=KEEP_LARGEST_UNREACHABLE,
    )

    reach_scalar = smooth_mask(reach_mask, SMOOTH_SIGMA)
    unreach_scalar = smooth_mask(unreach_mask, SMOOTH_SIGMA)

    reach_vertices, reach_faces = mesh_from_scalar(reach_scalar, spacing, origin, REACH_ISO_LEVEL)
    unreach_vertices, unreach_faces = mesh_from_scalar(unreach_scalar, spacing, origin, UNREACH_ISO_LEVEL)

    return {
        "reach_vertices": reach_vertices,
        "reach_faces": reach_faces,
        "unreach_vertices": unreach_vertices,
        "unreach_faces": unreach_faces,
        "alpha_radius": alpha_radius,
        "grid_shape": X.shape,
        "reach_voxels": int(np.count_nonzero(reach_mask)),
        "outer_voxels": int(np.count_nonzero(outer_mask)),
        "unreach_voxels": int(np.count_nonzero(unreach_mask)),
    }


def print_debug_info(points_raw, points_mm, scale_to_mm, shell):
    raw_min = points_raw.min(axis=0)
    raw_max = points_raw.max(axis=0)
    mm_min = points_mm.min(axis=0)
    mm_max = points_mm.max(axis=0)
    print(f"Loaded points            : {points_mm.shape[0]}")
    print(f"Unit scale to mm         : x{scale_to_mm:g}")
    print(f"Raw min/max              : {raw_min} / {raw_max}")
    print(f"MM min/max               : {mm_min} / {mm_max}")
    print(f"MM span                  : {mm_max - mm_min}")
    print(f"Estimated reach alpha    : {shell['alpha_radius']:.6f} mm")
    print(f"Voxel grid size          : {shell['grid_shape']}")
    print(f"Outer voxels             : {shell['outer_voxels']}")
    print(f"Reachable voxels         : {shell['reach_voxels']}")
    print(f"Unreachable voxels       : {shell['unreach_voxels']}")


def main():
    raw_points = load_points(NPZ_PATH, TARGET_FIELD)
    points_mm, scale_to_mm = convert_points_to_mm(raw_points)
    points_mm = unique_points(points_mm)
    shell = build_workspace_shells(points_mm)

    fig = plt.figure(figsize=(8.6, 7.2), facecolor=BACKGROUND)
    ax = fig.add_subplot(111, projection="3d")

    if shell["unreach_vertices"] is not None and shell["unreach_faces"] is not None:
        plot_mesh(ax, shell["unreach_vertices"], shell["unreach_faces"], UNREACH_COLOR, UNREACH_ALPHA)
    if shell["reach_vertices"] is not None and shell["reach_faces"] is not None:
        plot_mesh(ax, shell["reach_vertices"], shell["reach_faces"], REACH_COLOR, REACH_ALPHA)

    if shell["reach_vertices"] is None and shell["unreach_vertices"] is None:
        ax.scatter(points_mm[:, 0], points_mm[:, 1], points_mm[:, 2], s=1.0, c=POINT_COLOR, alpha=0.18)

    style_axes(ax, points_mm)
    ax.set_title("Translation Workspace", pad=14)
    plt.tight_layout()

    if OUTPUT_FIGURE:
        fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight", facecolor=BACKGROUND)

    print_debug_info(raw_points, points_mm, scale_to_mm, shell)
    plt.show()


if __name__ == "__main__":
    main()
