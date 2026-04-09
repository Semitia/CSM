"""
Paper-style plotting helpers for local dexterous probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
from scipy.spatial import ConvexHull

from csm.model import CSM
from .kinematics import DexterousMode3State, make_mode3_display_csm


@dataclass
class DexterousPlotOptions:
    output_path: Path | None = None
    show_figure: bool = True
    figsize: tuple[float, float] = (6.2, 7.2)
    sphere_alpha: float = 0.24
    patch_alpha: float = 0.58
    sphere_color: str = "#57E36D"
    patch_color: str = "#C329B8"
    robot_colors: tuple[str, str, str] = ("#1E40AF", "#22C55E", "#DC2626")
    elev: float = 17.0
    azim: float = -58.0
    save_debug_figure: bool = False
    debug_output_dir: Path | None = None
    region_grid_n: int = 220
    region_sigma: float = 1.2
    region_close_iters: int = 2
    region_threshold: float = 0.34
    use_cap_if_fit_below_deg: float = 4.0
    prefer_analytic_boundary: bool = True
    prefer_cap_over_patch: bool = False
    boundary_circle_tol: float = 0.025
    hull_arc_tol: float = 0.03
    line_simplify_tol: float = 0.01


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(helper, z)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    x = np.cross(helper, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return x, y


def _set_equal_3d_axes(ax, all_points: np.ndarray, pad: float = 0.006) -> None:
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(np.max(maxs - mins), 1e-3)
    half = 0.5 * span + pad
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _plot_base_sphere(ax, center: np.ndarray, radius: float, color: str, alpha: float) -> None:
    u = np.linspace(0.0, 2.0 * math.pi, 60)
    v = np.linspace(0.0, math.pi, 40)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(uu) * np.sin(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(vv)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=False)


def _plot_cap(ax, center: np.ndarray, direction: np.ndarray, angular_radius: float, radius: float, color: str, alpha: float) -> None:
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    bx, by = _orthonormal_basis(direction)
    rho = np.linspace(0.0, angular_radius, 28)
    phi = np.linspace(0.0, 2.0 * math.pi, 64)
    rr, pp = np.meshgrid(rho, phi, indexing="ij")
    dirs = (
        np.cos(rr)[..., None] * direction[None, None, :]
        + np.sin(rr)[..., None] * (
            np.cos(pp)[..., None] * bx[None, None, :]
            + np.sin(pp)[..., None] * by[None, None, :]
        )
    )
    x = center[0] + radius * dirs[..., 0]
    y = center[1] + radius * dirs[..., 1]
    z = center[2] + radius * dirs[..., 2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=False)


def _plot_patch(ax, center: np.ndarray, radius: float, directions_world: np.ndarray, color: str, alpha: float) -> None:
    if directions_world.shape[0] < 3:
        return
    centroid = np.mean(directions_world, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    bx, by = _orthonormal_basis(centroid)
    uv = np.column_stack([directions_world @ bx, directions_world @ by])
    tri = mtri.Triangulation(uv[:, 0], uv[:, 1])
    points = center[None, :] + radius * directions_world
    surf = ax.plot_trisurf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        triangles=tri.triangles,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        surf.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return labeled == keep


def _extract_longest_contour(mask: np.ndarray, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(xs, zs, mask.astype(float).T, levels=[0.5])
        segments = []
        for level_segments in getattr(cs, "allsegs", []):
            segments.extend(level_segments)
        if not segments:
            return np.zeros((0, 2), dtype=float)
        vertices = max(segments, key=lambda verts: len(verts))
        return np.asarray(vertices, dtype=float)
    finally:
        plt.close(fig)


def _smooth_closed_curve(points: np.ndarray, passes: int = 6) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 5:
        return pts
    out = pts.copy()
    for _ in range(passes):
        out = 0.25 * np.roll(out, 1, axis=0) + 0.5 * out + 0.25 * np.roll(out, -1, axis=0)
    radii = np.linalg.norm(out, axis=1)
    over = radii > 1.0
    if np.any(over):
        out[over] = out[over] / radii[over][:, None]
    return out


def _dedupe_rows(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, pts.shape[-1] if pts.ndim > 1 else 0)
    decimals = max(0, int(round(-math.log10(eps))))
    _, idx = np.unique(np.round(pts, decimals=decimals), axis=0, return_index=True)
    return pts[np.sort(idx)]


def _sample_unit_arc(start: np.ndarray, end: np.ndarray, n_samples: int = 96) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    a0 = float(math.atan2(start[1], start[0]))
    a1 = float(math.atan2(end[1], end[0]))
    diff = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) < 1e-10:
        return np.repeat(start.reshape(1, 2), max(2, n_samples), axis=0)
    angles = a0 + np.linspace(0.0, diff, max(2, n_samples))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _rdp_indices(points: np.ndarray, epsilon: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 2:
        return np.arange(pts.shape[0], dtype=int)

    start = pts[0]
    end = pts[-1]
    seg = end - start
    seg_norm = float(np.linalg.norm(seg))
    if seg_norm < 1e-12:
        dists = np.linalg.norm(pts[1:-1] - start[None, :], axis=1)
    else:
        rel = pts[1:-1] - start[None, :]
        proj = (rel @ seg) / seg_norm
        closest = start[None, :] + (proj[:, None] / seg_norm) * seg[None, :]
        dists = np.linalg.norm(pts[1:-1] - closest, axis=1)
    if dists.size == 0:
        return np.array([0, pts.shape[0] - 1], dtype=int)
    split = int(np.argmax(dists)) + 1
    if float(np.max(dists)) <= epsilon:
        return np.array([0, pts.shape[0] - 1], dtype=int)
    left = _rdp_indices(pts[: split + 1], epsilon)
    right = _rdp_indices(pts[split:], epsilon) + split
    return np.concatenate([left[:-1], right])


def _fit_line_hesse(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    offset = -float(normal @ centroid)
    return normal, offset


def _intersect_lines(normal1: np.ndarray, offset1: float, normal2: np.ndarray, offset2: float) -> np.ndarray | None:
    A = np.vstack([normal1, normal2])
    b = -np.array([offset1, offset2], dtype=float)
    if abs(float(np.linalg.det(A))) < 1e-10:
        return None
    return np.linalg.solve(A, b)


def _intersect_line_unit_circle(normal: np.ndarray, offset: float, hint: np.ndarray) -> np.ndarray | None:
    a, b = float(normal[0]), float(normal[1])
    pts: list[np.ndarray] = []
    if abs(b) > abs(a):
        xs = np.roots([1.0 + (a / b) ** 2, 2.0 * offset * a / (b * b), (offset / b) ** 2 - 1.0])
        for x in xs:
            if abs(x.imag) > 1e-8:
                continue
            x = float(x.real)
            z = float(-(a * x + offset) / b)
            pts.append(np.array([x, z], dtype=float))
    else:
        zs = np.roots([1.0 + (b / a) ** 2, 2.0 * offset * b / (a * a), (offset / a) ** 2 - 1.0])
        for z in zs:
            if abs(z.imag) > 1e-8:
                continue
            z = float(z.real)
            x = float(-(b * z + offset) / a)
            pts.append(np.array([x, z], dtype=float))
    if not pts:
        return None
    pts_arr = np.vstack(pts)
    idx = int(np.argmin(np.linalg.norm(pts_arr - np.asarray(hint, dtype=float)[None, :], axis=1)))
    return pts_arr[idx]


def _sample_polyline(vertices: np.ndarray, samples_per_seg: int = 48) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    if verts.shape[0] < 2:
        return verts
    parts = []
    for idx in range(verts.shape[0] - 1):
        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)
        seg = (1.0 - t)[:, None] * verts[idx][None, :] + t[:, None] * verts[idx + 1][None, :]
        parts.append(seg)
    parts.append(verts[-1:].copy())
    return np.vstack(parts)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    if not np.any(mask):
        return None
    n = len(mask)
    doubled = np.concatenate([mask, mask])
    best = None
    start = None
    for idx, value in enumerate(doubled):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(doubled) - 1):
            end = idx - 1 if not value else idx
            length = end - start + 1
            if length <= n:
                if best is None or length > best[1] - best[0] + 1:
                    best = (start, end)
            start = None
    if best is None:
        return None
    return best[0] % n, best[1] % n


def _extract_circular_slice(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return points[start : end + 1]
    return np.vstack([points[start:], points[: end + 1]])


def _sort_open_path(points: np.ndarray, start_hint: np.ndarray | None = None, end_hint: np.ndarray | None = None) -> np.ndarray:
    pts = _dedupe_rows(points, eps=5e-5)
    if pts.shape[0] <= 2:
        return pts
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    scores = centered @ tangent
    order = np.argsort(scores)
    ordered = pts[order]
    if start_hint is None:
        return ordered
    start_hint = np.asarray(start_hint, dtype=float)
    dist_front = float(np.linalg.norm(ordered[0] - start_hint))
    dist_back = float(np.linalg.norm(ordered[-1] - start_hint))
    if dist_back < dist_front:
        ordered = ordered[::-1]
    if end_hint is not None and ordered.shape[0] > 1:
        end_hint = np.asarray(end_hint, dtype=float)
        if np.linalg.norm(ordered[-1] - end_hint) > np.linalg.norm(ordered[0] - end_hint):
            ordered = ordered[::-1]
    return ordered


def _triangulate_planar_region(boundary: np.ndarray, interior_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    polygon = np.asarray(boundary, dtype=float)
    if polygon.shape[0] < 3:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)
    polygon = _dedupe_rows(polygon, eps=5e-5)
    if polygon.shape[0] < 3:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)
    interior = np.asarray(interior_points, dtype=float)
    if interior.ndim == 1:
        interior = interior.reshape(-1, 2)
    if interior.size:
        poly = MplPath(polygon, closed=True)
        keep = poly.contains_points(interior, radius=-1e-6)
        interior = _dedupe_rows(interior[keep], eps=5e-5)
    vertices_2d = polygon if interior.size == 0 else np.vstack([polygon, interior])
    tri = mtri.Triangulation(vertices_2d[:, 0], vertices_2d[:, 1])
    triangles = tri.triangles
    poly = MplPath(polygon, closed=True)
    centroids = np.mean(vertices_2d[triangles], axis=1)
    keep = poly.contains_points(centroids, radius=-1e-6)
    return vertices_2d, triangles[keep]


def _lift_patch_vertices_to_world(vertices_2d: np.ndarray, triangles: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    if vertices_2d.shape[0] < 3 or triangles.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    x = vertices_2d[:, 0]
    z = vertices_2d[:, 1]
    y_abs = np.sqrt(np.maximum(1.0 - x * x - z * z, 0.0))
    vertices_pos = np.column_stack([x, y_abs, z])
    vertices_neg = np.column_stack([x, -y_abs, z])
    vertices_sym_3d = np.vstack([vertices_pos, vertices_neg])
    n_vertices = vertices_2d.shape[0]
    triangles_dual = np.vstack([triangles, triangles[:, ::-1] + n_vertices])
    R_ws = np.array(
        [
            [math.cos(gamma), -math.sin(gamma), 0.0],
            [math.sin(gamma), math.cos(gamma), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    vertices_3d_world = (R_ws @ vertices_sym_3d.T).T
    return vertices_3d_world, triangles_dual


def _build_patch_geometry_from_boundary_families(
    probe,
    options: DexterousPlotOptions,
) -> dict:
    feasible = np.asarray(probe.feasible_directions_sym, dtype=float)
    if feasible.shape[0] < 8:
        return {}
    try:
        hull = ConvexHull(feasible)
    except Exception:
        return {}
    hull_pts = feasible[hull.vertices]
    if hull_pts.shape[0] < 6:
        return {}
    radii = np.linalg.norm(hull_pts, axis=1)
    arc_run = _longest_true_run(np.abs(radii - 1.0) <= options.hull_arc_tol)
    if arc_run is None:
        return {}
    arc_start_idx, arc_end_idx = arc_run
    arc_hull = _extract_circular_slice(hull_pts, arc_start_idx, arc_end_idx)
    poly_hull = _extract_circular_slice(hull_pts, arc_end_idx, arc_start_idx)
    if arc_hull.shape[0] < 2 or poly_hull.shape[0] < 3:
        return {}

    keep_idx = _rdp_indices(poly_hull, options.line_simplify_tol)
    keep_idx = np.unique(keep_idx)
    if keep_idx[0] != 0:
        keep_idx = np.concatenate([[0], keep_idx])
    if keep_idx[-1] != poly_hull.shape[0] - 1:
        keep_idx = np.concatenate([keep_idx, [poly_hull.shape[0] - 1]])
    if keep_idx.size < 3:
        return {}

    segment_normals: list[np.ndarray] = []
    segment_offsets: list[float] = []
    for i0, i1 in zip(keep_idx[:-1], keep_idx[1:]):
        raw_seg = poly_hull[i0 : i1 + 1]
        if raw_seg.shape[0] < 2:
            continue
        normal, offset = _fit_line_hesse(raw_seg)
        segment_normals.append(normal)
        segment_offsets.append(offset)
    if len(segment_normals) < 2:
        return {}

    poly_vertices = []
    start_intersection = _intersect_line_unit_circle(segment_normals[0], segment_offsets[0], poly_hull[0])
    end_intersection = _intersect_line_unit_circle(segment_normals[-1], segment_offsets[-1], poly_hull[-1])
    if start_intersection is None or end_intersection is None:
        return {}
    poly_vertices.append(start_intersection)
    for idx in range(len(segment_normals) - 1):
        corner = _intersect_lines(
            segment_normals[idx],
            segment_offsets[idx],
            segment_normals[idx + 1],
            segment_offsets[idx + 1],
        )
        if corner is None:
            return {}
        poly_vertices.append(corner)
    poly_vertices.append(end_intersection)
    poly_vertices = np.asarray(poly_vertices, dtype=float)

    arc_start = end_intersection
    arc_end = start_intersection
    arc_points = _sample_unit_arc(arc_start, arc_end, n_samples=max(64, arc_hull.shape[0] * 12))
    poly_points = _sample_polyline(poly_vertices, samples_per_seg=56)
    boundary = np.vstack([arc_points, poly_points[1:-1], arc_points[:1]])
    interior_stride = max(1, feasible.shape[0] // 2400)
    vertices_2d, triangles = _triangulate_planar_region(boundary, feasible[::interior_stride])
    vertices_3d_world, triangles_3d = _lift_patch_vertices_to_world(vertices_2d, triangles, float(probe.gamma))
    return {
        "mask": np.zeros((0, 0), dtype=bool),
        "xs": np.array([], dtype=float),
        "zs": np.array([], dtype=float),
        "contour_raw": boundary,
        "contour_smooth": boundary,
        "vertices_2d": vertices_2d,
        "triangles_2d": triangles,
        "vertices_3d_world": vertices_3d_world,
        "triangles": triangles_3d,
        "method": "analytic_boundary",
    }


def _build_patch_geometry_from_symmetry_region(
    probe,
    options: DexterousPlotOptions,
) -> dict:
    if options.prefer_analytic_boundary:
        analytic_geom = _build_patch_geometry_from_boundary_families(probe, options)
        if analytic_geom:
            return analytic_geom
    points = np.asarray(probe.feasible_directions_sym, dtype=float)
    if points.shape[0] < 8:
        return {
            "mask": np.zeros((0, 0), dtype=bool),
            "xs": np.array([], dtype=float),
            "zs": np.array([], dtype=float),
            "contour_raw": np.zeros((0, 2), dtype=float),
            "contour_smooth": np.zeros((0, 2), dtype=float),
            "vertices_2d": np.zeros((0, 2), dtype=float),
            "triangles_2d": np.zeros((0, 3), dtype=int),
            "vertices_3d_world": np.zeros((0, 3), dtype=float),
            "triangles": np.zeros((0, 3), dtype=int),
        }

    xs = np.linspace(-1.0, 1.0, options.region_grid_n)
    zs = np.linspace(-1.0, 1.0, options.region_grid_n)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Z.ravel()])

    step = 2.0 / max(options.region_grid_n - 1, 1)
    radius2 = max(2.5 * step, 0.02) ** 2
    d2 = np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    nearest = np.min(d2, axis=1)
    density = np.exp(-nearest / max(radius2, 1e-12)).reshape(options.region_grid_n, options.region_grid_n)
    disk_mask = (X**2 + Z**2) <= 1.0 + 1e-8
    density *= disk_mask
    smooth = gaussian_filter(density.astype(float), sigma=options.region_sigma)
    mask = smooth >= options.region_threshold
    mask &= disk_mask
    if options.region_close_iters > 0:
        structure = np.ones((3, 3), dtype=bool)
        for _ in range(options.region_close_iters):
            mask = binary_closing(mask, structure=structure)
    mask = binary_fill_holes(mask)
    mask &= disk_mask
    mask = _largest_component(mask)

    contour_raw = _extract_longest_contour(mask, xs, zs)
    contour_smooth = _smooth_closed_curve(contour_raw) if contour_raw.size else contour_raw
    if contour_smooth.shape[0] < 3:
        return {
            "mask": mask,
            "xs": xs,
            "zs": zs,
            "contour_raw": contour_raw,
            "contour_smooth": contour_smooth,
            "vertices_2d": np.zeros((0, 2), dtype=float),
            "triangles_2d": np.zeros((0, 3), dtype=int),
            "vertices_3d_world": np.zeros((0, 3), dtype=float),
            "triangles": np.zeros((0, 3), dtype=int),
        }

    inside_points = grid_points[mask.ravel()]
    sample_stride = max(1, inside_points.shape[0] // 1800)
    inside_points = inside_points[::sample_stride]
    vertices_2d = np.vstack([contour_smooth, inside_points])
    tri = mtri.Triangulation(vertices_2d[:, 0], vertices_2d[:, 1])
    poly = MplPath(contour_smooth, closed=True)
    triangles = tri.triangles
    centroids = np.mean(vertices_2d[triangles], axis=1)
    keep = poly.contains_points(centroids, radius=-1e-6)
    triangles = triangles[keep]

    vertices_3d_world, triangles = _lift_patch_vertices_to_world(vertices_2d, triangles, float(probe.gamma))
    return {
        "mask": mask,
        "xs": xs,
        "zs": zs,
        "contour_raw": contour_raw,
        "contour_smooth": contour_smooth,
        "vertices_2d": vertices_2d,
        "triangles_2d": triangles,
        "vertices_3d_world": vertices_3d_world,
        "triangles": triangles,
        "method": "raster_mask",
    }


def _plot_patch_mesh(ax, center: np.ndarray, radius: float, vertices_world: np.ndarray, triangles: np.ndarray, color: str, alpha: float) -> None:
    if vertices_world.shape[0] < 3 or triangles.shape[0] == 0:
        return
    points = center[None, :] + radius * vertices_world
    surf = ax.plot_trisurf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        triangles=triangles,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        surf.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass


def _plot_point_cloud(ax, center: np.ndarray, radius: float, directions_world: np.ndarray, color: str, alpha: float, size: float = 10.0) -> None:
    if directions_world.size == 0:
        return
    points = center[None, :] + radius * directions_world
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, color=color, alpha=alpha)


def _plot_robot(ax, csm: CSM, probe_state: DexterousMode3State | None, colors: tuple[str, str, str]) -> None:
    if probe_state is None:
        return
    display_csm = make_mode3_display_csm(csm, probe_state)
    vis = display_csm.get_visualization_segments(arc_points=28, straight_points=6)
    for seg in vis["segments"]:
        pts = np.asarray(seg["points"], dtype=float)
        color = colors[0]
        if seg["label"] == "rigid":
            color = colors[1]
        elif seg["label"] == "seg2":
            color = colors[2]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=3.0, alpha=0.95)
    tool = vis["tool"]
    ax.plot(
        [tool["start"][0], tool["end"][0]],
        [tool["start"][1], tool["end"][1]],
        [tool["start"][2], tool["end"][2]],
        color="#111827",
        linewidth=3.0,
        alpha=0.95,
    )


def plot_dexterous_probe(ax, probe, *, csm: CSM | None = None, options: DexterousPlotOptions | None = None) -> None:
    options = options or DexterousPlotOptions()
    center = np.asarray(probe.position_xyz, dtype=float)
    _plot_base_sphere(ax, center, float(probe.sphere_radius_m), options.sphere_color, options.sphere_alpha)
    patch_geom = _build_patch_geometry_from_symmetry_region(probe, options)
    use_cap = (
        probe.cap_center_world is not None
        and probe.cap_angular_radius is not None
        and probe.cap_fit_error is not None
        and np.degrees(probe.cap_fit_error) <= options.use_cap_if_fit_below_deg
    )
    if use_cap and options.prefer_cap_over_patch:
        _plot_cap(ax, center, probe.cap_center_world, probe.cap_angular_radius, float(probe.sphere_radius_m), options.patch_color, options.patch_alpha)
    elif patch_geom["vertices_3d_world"].shape[0] >= 3 and patch_geom["triangles"].shape[0] > 0:
        _plot_patch_mesh(
            ax,
            center,
            float(probe.sphere_radius_m),
            patch_geom["vertices_3d_world"],
            patch_geom["triangles"],
            options.patch_color,
            options.patch_alpha,
        )
    elif probe.feasible_directions_world.shape[0] >= 3:
        _plot_patch(ax, center, float(probe.sphere_radius_m), probe.feasible_directions_world, options.patch_color, options.patch_alpha)
    if csm is not None:
        _plot_robot(ax, csm, probe.display_state, options.robot_colors)

    all_points = [center[None, :]]
    if probe.feasible_directions_world.size:
        all_points.append(center[None, :] + probe.sphere_radius_m * probe.feasible_directions_world)
    if csm is not None and probe.display_state is not None:
        display_csm = make_mode3_display_csm(csm, probe.display_state)
        vis = display_csm.get_visualization_segments(arc_points=28, straight_points=6)
        for seg in vis["segments"]:
            all_points.append(np.asarray(seg["points"], dtype=float))
        all_points.append(np.asarray([vis["tool"]["start"], vis["tool"]["end"]], dtype=float))
    _set_equal_3d_axes(ax, np.vstack(all_points))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=options.elev, azim=options.azim)
    ax.grid(True, alpha=0.25)
    ax.set_title(probe.label or "Dexterous Probe")


def _plot_symmetry_debug(ax, probe) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#94A3B8", linewidth=1.4)
    if probe.feasible_directions_sym.size:
        ax.scatter(
            probe.feasible_directions_sym[:, 0],
            probe.feasible_directions_sym[:, 1],
            s=12,
            color="#FACC15",
            alpha=0.65,
            label="Feasible area",
        )
    if probe.type1_boundary_sym.size:
        ax.scatter(
            probe.type1_boundary_sym[:, 0],
            probe.type1_boundary_sym[:, 1],
            s=10,
            color="#06B6D4",
            alpha=0.7,
            label="Type-I boundary",
        )
    if probe.type2_boundary_sym.size:
        ax.scatter(
            probe.type2_boundary_sym[:, 0],
            probe.type2_boundary_sym[:, 1],
            s=10,
            color="#EF4444",
            alpha=0.7,
            label="Type-II boundary",
        )
    if probe.debug_data is not None:
        ax.set_title(
            f"Symmetry Plane\nfeasible={probe.debug_data['feasible_count_sym']} "
            f"cap_err={probe.debug_data['cap_fit_error_deg']:.2f} deg"
            if probe.debug_data.get("cap_fit_error_deg") is not None
            else "Symmetry Plane"
        )
    else:
        ax.set_title("Symmetry Plane")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$a_{sx}$")
    ax.set_ylabel(r"$a_{sz}$")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", frameon=False)


def _plot_sphere_mapping_debug(ax, probe, options: DexterousPlotOptions) -> None:
    center = np.asarray(probe.position_xyz, dtype=float)
    _plot_base_sphere(ax, center, float(probe.sphere_radius_m), options.sphere_color, 0.12)
    _plot_point_cloud(ax, center, float(probe.sphere_radius_m), probe.feasible_directions_world, "#D946EF", 0.65, size=10.0)
    if probe.type1_boundary_world.size:
        _plot_point_cloud(ax, center, float(probe.sphere_radius_m), probe.type1_boundary_world, "#06B6D4", 0.75, size=14.0)
    if probe.type2_boundary_world.size:
        _plot_point_cloud(ax, center, float(probe.sphere_radius_m), probe.type2_boundary_world, "#EF4444", 0.75, size=14.0)
    if probe.cap_center_world is not None and probe.cap_angular_radius is not None:
        _plot_cap(
            ax,
            center,
            probe.cap_center_world,
            probe.cap_angular_radius,
            float(probe.sphere_radius_m),
            "#111827",
            0.16,
        )
    all_points = [center[None, :]]
    if probe.feasible_directions_world.size:
        all_points.append(center[None, :] + probe.sphere_radius_m * probe.feasible_directions_world)
    _set_equal_3d_axes(ax, np.vstack(all_points))
    ax.set_title("Mapped 3D Directions")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=options.elev, azim=options.azim)
    ax.grid(True, alpha=0.25)


def _plot_final_debug(ax, probe, options: DexterousPlotOptions) -> None:
    plot_dexterous_probe(ax, probe, csm=None, options=options)
    ax.set_title("Final Render Primitive")


def _plot_mask_debug(ax, probe, options: DexterousPlotOptions) -> dict:
    geom = _build_patch_geometry_from_symmetry_region(probe, options)
    xs = geom["xs"]
    zs = geom["zs"]
    mask = geom["mask"]
    if mask.size:
        ax.imshow(
            mask.T.astype(float),
            extent=[xs[0], xs[-1], zs[0], zs[-1]],
            origin="lower",
            cmap="Greens",
            alpha=0.65,
            interpolation="nearest",
        )
    theta = np.linspace(0.0, 2.0 * math.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#94A3B8", linewidth=1.2)
    if geom["contour_raw"].size:
        ax.plot(geom["contour_raw"][:, 0], geom["contour_raw"][:, 1], color="#0EA5E9", linewidth=1.6, label="raw contour")
    if geom["contour_smooth"].size:
        ax.plot(geom["contour_smooth"][:, 0], geom["contour_smooth"][:, 1], color="#111827", linewidth=1.8, label="smooth contour")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Raster Mask / Contour" if geom.get("method") != "analytic_boundary" else "Analytic Boundary / Contour")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", frameon=False)
    return geom


def _plot_triangulation_debug(ax, probe, geom: dict) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#94A3B8", linewidth=1.2)
    verts = geom["vertices_2d"]
    triangles = geom.get("triangles_2d", geom["triangles"])
    if verts.shape[0] >= 3 and triangles.shape[0] > 0:
        tri = mtri.Triangulation(verts[:, 0], verts[:, 1], triangles=triangles)
        ax.triplot(tri, color="#7C3AED", linewidth=0.35, alpha=0.85)
    if geom["contour_smooth"].size:
        ax.plot(geom["contour_smooth"][:, 0], geom["contour_smooth"][:, 1], color="#111827", linewidth=1.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("2D Triangulation")


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def save_probe_debug_figure(probe, options: DexterousPlotOptions) -> None:
    if not options.save_debug_figure or options.debug_output_dir is None:
        return
    output_dir = Path(options.debug_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = (probe.label or "probe").replace(" ", "_")

    fig = plt.figure(figsize=(15.0, 11.0))
    ax1 = fig.add_subplot(2, 3, 1)
    _plot_symmetry_debug(ax1, probe)
    ax2 = fig.add_subplot(2, 3, 2)
    geom = _plot_mask_debug(ax2, probe, options)
    ax3 = fig.add_subplot(2, 3, 3)
    _plot_triangulation_debug(ax3, probe, geom)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    _plot_sphere_mapping_debug(ax4, probe, options)
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    _plot_final_debug(ax5, probe, options)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    debug_text = probe.debug_data or {}
    lines = [
        f"status: {debug_text.get('status')}",
        f"method: {debug_text.get('method')}",
        f"feasible_sym: {debug_text.get('feasible_count_sym')}",
        f"feasible_world: {debug_text.get('feasible_count_world')}",
        f"type1_sym: {debug_text.get('type1_boundary_count_sym')}",
        f"type2_sym: {debug_text.get('type2_boundary_count_sym')}",
        f"type1_world: {debug_text.get('type1_boundary_count_world')}",
        f"type2_world: {debug_text.get('type2_boundary_count_world')}",
        f"cap_fit_error_deg: {debug_text.get('cap_fit_error_deg')}",
        f"cap_radius_deg: {debug_text.get('cap_angular_radius_deg')}",
        f"use_cap: {debug_text.get('fit_used_as_cap')}",
        "",
        f"grid_n: {options.region_grid_n}",
        f"sigma: {options.region_sigma}",
        f"threshold: {options.region_threshold}",
        f"close_iters: {options.region_close_iters}",
        f"mask_pixels: {int(np.count_nonzero(geom['mask'])) if geom['mask'].size else 0}",
        f"contour_raw_pts: {len(geom['contour_raw'])}",
        f"contour_smooth_pts: {len(geom['contour_smooth'])}",
        f"triangles: {len(geom['triangles'])}",
    ]
    fallback = debug_text.get("fallback")
    if fallback is not None:
        lines.extend(
            [
                "",
                f"fallback.samples: {fallback.get('direction_samples')}",
                f"fallback.reachable: {fallback.get('reachable_count')}",
                f"fallback.has_q_seed: {fallback.get('has_q_seed')}",
            ]
        )
    ax6.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_label}_debug.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / f"{safe_label}_debug.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready(debug_text), f, ensure_ascii=False, indent=2)


def render_dexterous_figure(probes, *, csm: CSM, options: DexterousPlotOptions | None = None):
    options = options or DexterousPlotOptions()
    probes = list(probes)
    if not probes:
        raise ValueError("At least one probe is required.")

    n = len(probes)
    cols = min(2, n)
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(options.figsize[0] * cols, options.figsize[1] * rows))
    for idx, probe in enumerate(probes, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        plot_dexterous_probe(ax, probe, csm=csm, options=options)
        save_probe_debug_figure(probe, options)

    fig.tight_layout()
    if options.output_path is not None:
        options.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(options.output_path, dpi=220, bbox_inches="tight")
    if options.show_figure:
        plt.show()
    else:
        plt.close(fig)
    return fig
