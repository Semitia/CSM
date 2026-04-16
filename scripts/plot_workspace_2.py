"""
Module: plot_workspace_2.py
Description: Directly sample a side-view workspace profile from the robot model,
smooth the reachable/unreachable contours, and revolve them into a paper-like 3D plot.
"""
import json
import os
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

from csm.model import CSM

# ===== Data source =====
CONFIG_NAME = "csm_cfg_3mm.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
# PLOT_MODES = [3]
PLOT_MODES = [3]
COMBINED_SOURCE_MODES = (1, 2, 3, 4)
CACHE_DIR = Path("./data/profile_cache")
USE_CACHE = True
FORCE_REBUILD_CACHE = False

# ===== Direct sampling =====
# We sample only the side-view generating variables.
# phi / delta_1 / delta_2 are fixed to 0 and are reintroduced by revolving the profile.
MODE_SAMPLE_RES = {
    1: {"length": 220, "theta1": 1, "theta2": 240},
    2: {"length": 180, "theta1": 1, "theta2": 220},
    3: {"length": 60, "theta1": 60, "theta2": 60},
    4: {"length": 90, "theta1": 90, "theta2": 90},
}

# ===== Profile extraction =====
PROFILE_Z_BINS = 220
PROFILE_R_BINS = 180
PROFILE_SMOOTH_SIGMA = 1.2
PROFILE_ROW_THRESHOLD = 0.10
PROFILE_GLOBAL_THRESHOLD = 0.02
PROFILE_MIN_VOID_RATIO = 0.15
PROFILE_MIN_INNER_RUN = 8
PROFILE_CURVE_SMOOTH_SIGMA = 2.2
PROFILE_INNER_APEX_MAX_EXTEND_BINS = 12
PROFILE_ENDPOINT_FIT_SAMPLES = 7
PROFILE_ENDPOINT_ARC_SAMPLES = 48
PROFILE_ENDPOINT_LINE_RADIUS_RATIO = 24.0
PROFILE_ENDPOINT_MAX_TURN = 0.5 * np.pi
PROFILE_ENDPOINT_FIT_WINDOWS = (5, 7, 9, 11, 15, 21, 31)
PROFILE_MODE0_VOID_MIN_HEIGHT_BINS = 2
MODE0_DEBUG = False
# ===== Surface rendering =====
REVOLVE_SAMPLES = 40
CAP_RADIAL_SAMPLES = 18
SHOW_UNREACHABLE = True
MAIN_VIEW_STYLE = "real_3d"   # "real_3d" | "pseudo_3d"
SIDE_VIEW_STYLE = "pseudo_3d"   # "pseudo_3d" | "real_3d"

# MODE_COLORS = {1: "#E9A3A7", 2: "#F0E0AA", 3: "#9FD4EA", 4: "#CFCFCF"}
MODE_COLORS = {1: "#9FD4EA", 2: "#9FD4EA", 3: "#9FD4EA", 4: "#9FD4EA"}
MODE_LABELS = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
UNREACHABLE_COLOR = "#D79A6B"
COMBINED_REACHABLE_COLOR = "#9FD4EA"

REACH_ALPHA = 0.34
UNREACHABLE_ALPHA = 0.40

# ===== Figure =====
SEPARATE_PLOTS = False
OUTPUT_PATH = None
FALLBACK_OUTPUT_PATH = Path("./data/plot_workspace_2_fallback.png")
RAW_SCATTER_OUTPUT_PATH = Path("./data/plot_workspace_2_raw_scatter.png")
FIGSIZE = (10, 10)
VIEW_ELEV = 18
VIEW_AZIM = -40
SHOW_AXES = True
DISPLAY_IN_MM = True
SHOW_SIDE_VIEW = True
SHOW_SAMPLE_SCATTER = True
SHOW_SCATTER_PROFILE_CURVES = True
PSEUDO_VIEW_ELEV = 18
PSEUDO_VIEW_AZIM = -40
PSEUDO_MERIDIAN_SAMPLES = 240
SIDE_REAL_VIEW_ELEV = 0
SIDE_REAL_VIEW_AZIM = -90
SHOW_MANIPULATOR = True
MANIPULATOR_RENDER_MODE = "detailed"
MANIPULATOR_RNG_SEED = 20260406
UNREACHABLE_LINE_INSET_RADIAL_RES = 2.2
UNREACHABLE_LINEWIDTH = 0.6
UNREACHABLE_SCATTER_LINEWIDTH = 1.1


def _has_interactive_display():
    display = os.environ.get("DISPLAY")
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    backend = plt.get_backend().lower()
    if "agg" in backend:
        return False
    return bool(display or wayland_display)


def _linspace_from_zero(stop, num):
    if num <= 1 or stop <= 0:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, float(stop), int(num))


def _estimated_radial_resolution(r_vals):
    r_vals = np.asarray(r_vals, dtype=float)
    finite = r_vals[np.isfinite(r_vals)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite)) / max(PROFILE_R_BINS, 1)


def _inset_unreachable_curve_r(r_vals, reference_r=None, inset_scale=UNREACHABLE_LINE_INSET_RADIAL_RES):
    r_vals = np.asarray(r_vals, dtype=float)
    if reference_r is None:
        reference_r = r_vals
    radial_res = _estimated_radial_resolution(reference_r)
    if radial_res <= 0.0:
        return r_vals.copy()
    inset = float(inset_scale * radial_res)
    return np.maximum(r_vals - inset, 0.0)


def _cache_file_path(mode):
    return CACHE_DIR / f"{CONFIG_PATH.stem}_mode{mode}_profile_cache.npz"


def _display_color_for_mode(mode):
    if mode == 0:
        return COMBINED_REACHABLE_COLOR
    return MODE_COLORS[mode]


def _display_label_for_mode(mode):
    if mode == 0:
        return "Combined"
    return MODE_LABELS[mode]


def _current_cache_metadata(mode):
    config_sha256 = None
    if CONFIG_PATH.exists():
        config_sha256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    return {
        "config_name": CONFIG_NAME,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha256,
        "mode": int(mode),
        "sample_cfg": MODE_SAMPLE_RES[mode],
    }


def _segment_theta_upper_bound(csm, segment_index, length):
    length = max(float(length), 0.0)
    if segment_index == 1:
        if hasattr(csm, "max_theta1_for_length"):
            return float(csm.max_theta1_for_length(length))
        return float(csm.kappa_10 * length)
    if segment_index == 2:
        if hasattr(csm, "max_theta2_for_length"):
            return float(csm.max_theta2_for_length(length))
        return float(csm.kappa_20 * length)
    raise ValueError(f"Unsupported segment index: {segment_index}")


def _load_profile_cache(mode):
    cache_path = _cache_file_path(mode)
    if not USE_CACHE or FORCE_REBUILD_CACHE or not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as payload:
        cached_meta = json.loads(str(payload["metadata"].item()))
        current_meta = _current_cache_metadata(mode)
        for key, value in current_meta.items():
            if cached_meta.get(key) != value:
                return None
        if "side_points" not in payload:
            return None
        return np.asarray(payload["side_points"], dtype=float)


def _save_profile_cache(mode, side_points):
    if not USE_CACHE:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_path(mode)
    np.savez_compressed(
        cache_path,
        metadata=np.asarray(json.dumps(_current_cache_metadata(mode))),
        side_points=np.asarray(side_points, dtype=float),
    )
    print(f"Mode {mode}: cached sampled points -> {cache_path.resolve()}")


def _prepare_csm_for_mode(csm, mode):
    if mode == 1:
        csm.set_state(mode=1, phi=0, L1=0, L2=csm.L_20, Lr=0, Ls=0, theta_1=0, theta_2=0, delta_1=0, delta_2=0)
    elif mode == 2:
        csm.set_state(mode=2, phi=0, L1=0, L2=csm.L_20, Lr=0, Ls=0, theta_1=0, theta_2=0, delta_1=0, delta_2=0)
    elif mode == 3:
        csm.set_state(mode=3, phi=0, L1=0, L2=csm.L_20, Lr=csm.L_r0, Ls=0, theta_1=0, theta_2=0, delta_1=0, delta_2=0)
    elif mode == 4:
        csm.set_state(
            mode=4,
            phi=0,
            L1=csm.L_10,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=0,
            theta_1=0,
            theta_2=0,
            delta_1=0,
            delta_2=0,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _side_radius(point):
    return float(np.hypot(point[0], point[1]))


def _resolve_side_view_reference_angle(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        return 0.0

    radii = np.hypot(points[:, 0], points[:, 1])
    valid_indices = np.flatnonzero(radii > 1e-9)
    if valid_indices.size == 0:
        return 0.0

    ref_point = points[int(valid_indices[-1])]
    return float(np.arctan2(ref_point[0], ref_point[1]))


def _map_points_to_side_view(points, reference_angle=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        return np.empty((0, 2), dtype=float)

    if reference_angle is None:
        reference_angle = _resolve_side_view_reference_angle(points)

    cos_a = float(np.cos(reference_angle))
    sin_a = float(np.sin(reference_angle))
    side_coord = -points[:, 0] * sin_a + points[:, 1] * cos_a
    return np.column_stack((side_coord, points[:, 2]))


def _count_mode_side_samples(csm, mode, sample_cfg):
    length_res = int(sample_cfg["length"])
    theta1_res = int(sample_cfg["theta1"])
    theta2_res = int(sample_cfg["theta2"])

    total = 0
    if mode == 1:
        for L2 in _linspace_from_zero(csm.L_20, length_res):
            total += len(_linspace_from_zero(_segment_theta_upper_bound(csm, 2, L2), theta2_res))
    elif mode == 2:
        for _Lr in _linspace_from_zero(csm.L_r0, length_res):
            total += len(_linspace_from_zero(_segment_theta_upper_bound(csm, 2, csm.L2), theta2_res))
    elif mode == 3:
        for L1 in _linspace_from_zero(csm.L_10, length_res):
            theta1_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 1, L1), theta1_res)
            total += len(theta1_vals) * len(_linspace_from_zero(_segment_theta_upper_bound(csm, 2, csm.L2), theta2_res))
    elif mode == 4:
        for _Ls in _linspace_from_zero(csm.L_s0, length_res):
            theta1_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 1, csm.L1), theta1_res)
            total += len(theta1_vals) * len(_linspace_from_zero(_segment_theta_upper_bound(csm, 2, csm.L2), theta2_res))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return total


def sample_mode_side_points(csm, mode, sample_cfg):
    if mode == 0:
        raise ValueError("Mode 0 is a combined visualization mode and does not support direct sampling.")
    _prepare_csm_for_mode(csm, mode)

    length_res = int(sample_cfg["length"])
    theta1_res = int(sample_cfg["theta1"])
    theta2_res = int(sample_cfg["theta2"])

    side_points = []
    total_samples = _count_mode_side_samples(csm, mode, sample_cfg)

    with tqdm(total=total_samples, desc=f"Mode {mode} sampling", unit="sample") as pbar:
        if mode == 1:
            for L2 in _linspace_from_zero(csm.L_20, length_res):
                csm.L2 = L2
                theta2_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 2, L2), theta2_res)
                for theta_2 in theta2_vals:
                    csm.theta_2 = theta_2
                    csm.update()
                    side_points.append([_side_radius(csm.pose[:3]), float(csm.pose[2])])
                    pbar.update(1)

        elif mode == 2:
            for Lr in _linspace_from_zero(csm.L_r0, length_res):
                csm.Lr = Lr
                theta2_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 2, csm.L2), theta2_res)
                for theta_2 in theta2_vals:
                    csm.theta_2 = theta_2
                    csm.update()
                    side_points.append([_side_radius(csm.pose[:3]), float(csm.pose[2])])
                    pbar.update(1)

        elif mode == 3:
            csm.delta_1 = 0.0
            csm.delta_2 = 0.0
            for L1 in _linspace_from_zero(csm.L_10, length_res):
                csm.L1 = L1
                theta1_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 1, L1), theta1_res)
                for theta_1 in theta1_vals:
                    csm.theta_1 = theta_1
                    theta2_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 2, csm.L2), theta2_res)
                    for theta_2 in theta2_vals:
                        csm.theta_2 = theta_2
                        csm.update()
                        side_points.append([_side_radius(csm.pose[:3]), float(csm.pose[2])])
                        pbar.update(1)

        elif mode == 4:
            csm.delta_1 = 0.0
            csm.delta_2 = 0.0
            for Ls in _linspace_from_zero(csm.L_s0, length_res):
                csm.Ls = Ls
                theta1_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 1, csm.L1), theta1_res)
                for theta_1 in theta1_vals:
                    csm.theta_1 = theta_1
                    theta2_vals = _linspace_from_zero(_segment_theta_upper_bound(csm, 2, csm.L2), theta2_res)
                    for theta_2 in theta2_vals:
                        csm.theta_2 = theta_2
                        csm.update()
                        side_points.append([_side_radius(csm.pose[:3]), float(csm.pose[2])])
                        pbar.update(1)

    side_points = np.asarray(side_points, dtype=float)
    if side_points.size == 0:
        return np.empty((0, 2), dtype=float)

    side_points = np.unique(np.round(side_points, decimals=9), axis=0)
    print(f"Mode {mode}: sampled {side_points.shape[0]} side-view points")
    return side_points


def _longest_true_run(mask, weights):
    best_start = best_end = None
    best_score = -np.inf
    start = None

    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if start is not None and ((not flag) or idx == len(mask) - 1):
            end = idx if flag and idx == len(mask) - 1 else idx - 1
            score = float(np.sum(weights[start:end + 1]))
            if score > best_score:
                best_score = score
                best_start, best_end = start, end
            start = None

    keep = np.zeros_like(mask, dtype=bool)
    if best_start is not None and best_end - best_start + 1 >= PROFILE_MIN_INNER_RUN:
        keep[best_start:best_end + 1] = True
    return keep


def _estimate_dr_dz(z_vals, r_vals, at_end=False, samples=5):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 2:
        return None

    sample_count = min(samples, z_vals.size)
    if at_end:
        z_fit = z_vals[-sample_count:]
        r_fit = r_vals[-sample_count:]
    else:
        z_fit = z_vals[:sample_count]
        r_fit = r_vals[:sample_count]
    return float(np.polyfit(z_fit, r_fit, deg=1)[0])


def _append_axis_arc(z_vals, r_vals, max_extend_bins=12, arc_samples=24):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 3:
        return z_vals, r_vals

    slope = _estimate_dr_dz(z_vals, r_vals, at_end=True)
    if slope is None or slope >= -1e-9:
        return z_vals, r_vals

    r_end = float(r_vals[-1])
    z_end = float(z_vals[-1])
    z_center = z_end + slope * r_end
    radius = float(np.hypot(r_end, z_end - z_center))
    z_axis = z_center + radius
    if not np.isfinite(z_axis) or z_axis <= z_end:
        return z_vals, r_vals

    dz = np.median(np.diff(z_vals)) if z_vals.size >= 2 else 0.0
    if dz > 0 and z_axis - z_end > max_extend_bins * dz:
        return z_vals, r_vals

    theta_end = float(np.arctan2(z_end - z_center, r_end))
    theta_arc = np.linspace(theta_end, 0.5 * np.pi, arc_samples)
    arc_r = radius * np.cos(theta_arc)
    arc_z = z_center + radius * np.sin(theta_arc)
    return (
        np.concatenate([z_vals, arc_z[1:]]),
        np.concatenate([r_vals, arc_r[1:]]),
    )


def _fit_axis_arc_from_endpoint(z_vals, r_vals, axis_side="upper", fit_samples=5, arc_samples=120):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 3:
        return z_vals, r_vals

    sample_count = min(fit_samples, z_vals.size)
    z_fit = z_vals[-sample_count:]
    r_fit = r_vals[-sample_count:]
    slope = float(np.polyfit(z_fit, r_fit, deg=1)[0])
    r_end = float(r_vals[-1])
    z_end = float(z_vals[-1])
    z_center = z_end + slope * r_end
    radius = float(np.hypot(r_end, z_end - z_center))
    if not np.isfinite(radius) or radius <= 0:
        return z_vals, r_vals

    theta_end = float(np.arctan2(z_end - z_center, r_end))
    if axis_side == "upper":
        z_axis = z_center + radius
        if not np.isfinite(z_axis) or z_axis <= z_end:
            return z_vals, r_vals
        theta_arc = np.linspace(theta_end, 0.5 * np.pi, arc_samples)
        arc_r = radius * np.cos(theta_arc)
        arc_z = z_center + radius * np.sin(theta_arc)
        return (
            np.concatenate([z_vals, arc_z[1:]]),
            np.concatenate([r_vals, arc_r[1:]]),
        )

    if axis_side == "lower":
        z_axis = z_center - radius
        if not np.isfinite(z_axis) or z_axis >= z_vals[0]:
            return z_vals, r_vals
        theta_arc = np.linspace(-0.5 * np.pi, theta_end, arc_samples)
        arc_r = radius * np.cos(theta_arc)
        arc_z = z_center + radius * np.sin(theta_arc)
        return arc_z, arc_r

    raise ValueError(f"Unsupported axis_side: {axis_side}")


def _smooth_curve_for_plot(z_vals, r_vals, upsample_factor=4, sigma=1.2):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 4:
        return z_vals, r_vals

    dense_count = max(int(z_vals.size * upsample_factor), z_vals.size)
    z_dense = np.linspace(z_vals[0], z_vals[-1], dense_count)
    r_dense = np.interp(z_dense, z_vals, r_vals)
    if sigma > 0:
        r_dense = gaussian_filter1d(r_dense, sigma=sigma, mode="nearest")
    return z_dense, r_dense


def _smooth_scalar_curve(x_vals, y_vals, upsample_factor=4, sigma=1.2):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if x_vals.size < 4:
        return x_vals, y_vals

    dense_count = max(int(x_vals.size * upsample_factor), x_vals.size)
    x_dense = np.linspace(x_vals[0], x_vals[-1], dense_count)
    y_dense = np.interp(x_dense, x_vals, y_vals)
    if sigma > 0:
        y_dense = gaussian_filter1d(y_dense, sigma=sigma, mode="nearest")
    return x_dense, y_dense


def _normalize_vector(vec):
    vec = np.asarray(vec, dtype=float)
    norm = float(np.hypot(vec[0], vec[1]))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    return vec / norm


def _fit_circle_to_points(x_vals, y_vals):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if x_vals.size < 3:
        return None

    a_mat = np.column_stack((2.0 * x_vals, 2.0 * y_vals, np.ones_like(x_vals)))
    b_vec = x_vals * x_vals + y_vals * y_vals
    try:
        cx, cy, c_term = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    radius_sq = cx * cx + cy * cy + c_term
    if not np.isfinite(radius_sq) or radius_sq <= 1e-12:
        return None

    radius = float(np.sqrt(radius_sq))
    residual = np.hypot(x_vals - cx, y_vals - cy) - radius
    rms = float(np.sqrt(np.mean(residual * residual)))
    return {
        "center": np.array([float(cx), float(cy)], dtype=float),
        "radius": radius,
        "rms": rms,
    }


def _build_endpoint_helper(x_vals, y_vals, at_end=False, fit_samples=PROFILE_ENDPOINT_FIT_SAMPLES):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if x_vals.size < 2:
        return None

    sample_count = min(int(fit_samples), x_vals.size)
    if at_end:
        x_local = x_vals[-sample_count:][::-1]
        y_local = y_vals[-sample_count:][::-1]
        endpoint = np.array([x_vals[-1], y_vals[-1]], dtype=float)
    else:
        x_local = x_vals[:sample_count]
        y_local = y_vals[:sample_count]
        endpoint = np.array([x_vals[0], y_vals[0]], dtype=float)

    tangent_in = _normalize_vector(np.array([x_local[1] - x_local[0], y_local[1] - y_local[0]], dtype=float))
    if tangent_in is None:
        return None
    tangent_out = -tangent_in

    helper = {
        "kind": "line",
        "point": endpoint,
        "dir_out": tangent_out,
        "dir_in": tangent_in,
    }

    circle_fit = _fit_circle_to_points(x_local, y_local)
    if circle_fit is None:
        return helper

    chord = float(np.hypot(x_local[-1] - x_local[0], y_local[-1] - y_local[0]))
    if chord <= 1e-9:
        return helper

    radius = circle_fit["radius"]
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or radius > PROFILE_ENDPOINT_LINE_RADIUS_RATIO * chord
        or circle_fit["rms"] > 0.08 * chord
    ):
        return helper

    center = circle_fit["center"]
    angles = np.unwrap(np.arctan2(y_local - center[1], x_local - center[0]))
    angle_diffs = np.diff(angles)
    if not np.all(np.isfinite(angle_diffs)) or np.max(np.abs(angle_diffs)) <= 1e-6:
        return helper

    a0 = float(angles[0])
    tangent_ccw = _normalize_vector(np.array([-np.sin(a0), np.cos(a0)], dtype=float))
    if tangent_ccw is None:
        return helper

    turn_in_sign = 1.0 if float(np.dot(tangent_ccw, tangent_in)) >= 0.0 else -1.0
    turn_out_sign = -turn_in_sign
    return {
        "kind": "circle",
        "point": endpoint,
        "center": center,
        "radius": radius,
        "angle0": a0,
        "turn_out_sign": turn_out_sign,
    }


def _cross_2d(a_vec, b_vec):
    return float(a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0])


def _trim_endpoint_flat_run(x_vals, y_vals, axis="x", at_end=False, tol=1e-8, min_run_points=4):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if x_vals.size < min_run_points + 2:
        return x_vals, y_vals, False

    coord = x_vals if axis == "x" else y_vals
    if at_end:
        start = x_vals.size - 1
        while start - 1 >= 0 and abs(coord[start] - coord[start - 1]) <= tol:
            start -= 1
        trimmed_points = x_vals.size - start - 1
        if trimmed_points < min_run_points:
            return x_vals, y_vals, False
        return x_vals[:start + 1], y_vals[:start + 1], True

    end = 0
    while end + 1 < x_vals.size and abs(coord[end + 1] - coord[end]) <= tol:
        end += 1
    trimmed_points = end
    if trimmed_points < min_run_points:
        return x_vals, y_vals, False
    return x_vals[end:], y_vals[end:], True


def _helper_progress(helper, point):
    point = np.asarray(point, dtype=float)
    tol = 1e-7

    if helper["kind"] == "line":
        delta = point - helper["point"]
        length = float(np.dot(delta, helper["dir_out"]))
        distance = abs(_cross_2d(delta, helper["dir_out"]))
        if length < -tol or distance > 1e-5:
            return None
        return max(length, 0.0)

    delta = point - helper["center"]
    distance = float(np.hypot(delta[0], delta[1]))
    if not np.isfinite(distance) or abs(distance - helper["radius"]) > 1e-5:
        return None

    angle = float(np.arctan2(delta[1], delta[0]))
    signed_delta = float(np.arctan2(np.sin(angle - helper["angle0"]), np.cos(angle - helper["angle0"])))
    turn = helper["turn_out_sign"] * signed_delta
    if turn < -tol or turn > PROFILE_ENDPOINT_MAX_TURN + tol:
        return None
    return max(turn, 0.0) * helper["radius"]


def _line_line_intersections(helper_a, helper_b):
    dir_a = helper_a["dir_out"]
    dir_b = helper_b["dir_out"]
    denom = _cross_2d(dir_a, dir_b)
    if abs(denom) <= 1e-10:
        return []

    delta = helper_b["point"] - helper_a["point"]
    length_a = _cross_2d(delta, dir_b) / denom
    length_b = _cross_2d(delta, dir_a) / denom
    if length_a < -1e-9 or length_b < -1e-9:
        return []
    return [helper_a["point"] + length_a * dir_a]


def _line_circle_intersections(line_helper, circle_helper):
    origin = line_helper["point"]
    direction = line_helper["dir_out"]
    offset = origin - circle_helper["center"]

    b_term = 2.0 * float(np.dot(direction, offset))
    c_term = float(np.dot(offset, offset) - circle_helper["radius"] * circle_helper["radius"])
    disc = b_term * b_term - 4.0 * c_term
    if disc < -1e-10:
        return []
    disc = max(disc, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    roots = [(-b_term - sqrt_disc) / 2.0, (-b_term + sqrt_disc) / 2.0]

    points = []
    for root in roots:
        if root < -1e-9:
            continue
        points.append(origin + max(root, 0.0) * direction)
    return points


def _circle_circle_intersections(helper_a, helper_b):
    center_delta = helper_b["center"] - helper_a["center"]
    center_dist = float(np.hypot(center_delta[0], center_delta[1]))
    r0 = helper_a["radius"]
    r1 = helper_b["radius"]
    if (
        center_dist <= 1e-10
        or center_dist > r0 + r1 + 1e-9
        or center_dist < abs(r0 - r1) - 1e-9
    ):
        return []

    a_term = (r0 * r0 - r1 * r1 + center_dist * center_dist) / (2.0 * center_dist)
    h_sq = r0 * r0 - a_term * a_term
    if h_sq < -1e-10:
        return []
    h_sq = max(h_sq, 0.0)
    h_term = float(np.sqrt(h_sq))

    base = helper_a["center"] + (a_term / center_dist) * center_delta
    normal = np.array([-center_delta[1], center_delta[0]], dtype=float) / center_dist
    if h_term <= 1e-10:
        return [base]
    return [base + h_term * normal, base - h_term * normal]


def _find_shared_endpoint_target(helper_a, helper_b):
    if helper_a is None or helper_b is None:
        return None

    if helper_a["kind"] == "line" and helper_b["kind"] == "line":
        candidates = _line_line_intersections(helper_a, helper_b)
    elif helper_a["kind"] == "line" and helper_b["kind"] == "circle":
        candidates = _line_circle_intersections(helper_a, helper_b)
    elif helper_a["kind"] == "circle" and helper_b["kind"] == "line":
        candidates = _line_circle_intersections(helper_b, helper_a)
    else:
        candidates = _circle_circle_intersections(helper_a, helper_b)

    best_point = None
    best_cost = np.inf
    for candidate in candidates:
        prog_a = _helper_progress(helper_a, candidate)
        prog_b = _helper_progress(helper_b, candidate)
        if prog_a is None or prog_b is None:
            continue
        total_cost = float(prog_a + prog_b)
        if total_cost < best_cost:
            best_cost = total_cost
            best_point = np.asarray(candidate, dtype=float)
    return best_point


def _extend_curve_with_helper(x_vals, y_vals, helper, target_point, at_end=False, arc_samples=PROFILE_ENDPOINT_ARC_SAMPLES):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    target_point = np.asarray(target_point, dtype=float)
    if helper is None or target_point.size != 2:
        return x_vals, y_vals

    if helper["kind"] == "line":
        if at_end:
            return (
                np.concatenate([x_vals, [target_point[0]]]),
                np.concatenate([y_vals, [target_point[1]]]),
            )
        return (
            np.concatenate([[target_point[0]], x_vals]),
            np.concatenate([[target_point[1]], y_vals]),
        )

    target_delta = target_point - helper["center"]
    target_angle = float(np.arctan2(target_delta[1], target_delta[0]))
    signed_delta = float(
        np.arctan2(np.sin(target_angle - helper["angle0"]), np.cos(target_angle - helper["angle0"]))
    )
    turn = helper["turn_out_sign"] * signed_delta
    if turn <= 1e-9:
        return x_vals, y_vals

    sample_count = max(10, int(np.ceil(arc_samples * turn / max(0.25 * np.pi, turn))) + 1)
    if at_end:
        arc_angles = np.linspace(helper["angle0"], target_angle, sample_count)
        arc_x = helper["center"][0] + helper["radius"] * np.cos(arc_angles)
        arc_y = helper["center"][1] + helper["radius"] * np.sin(arc_angles)
        return (
            np.concatenate([x_vals, arc_x[1:]]),
            np.concatenate([y_vals, arc_y[1:]]),
        )

    arc_angles = np.linspace(target_angle, helper["angle0"], sample_count)
    arc_x = helper["center"][0] + helper["radius"] * np.cos(arc_angles)
    arc_y = helper["center"][1] + helper["radius"] * np.sin(arc_angles)
    return (
        np.concatenate([arc_x[:-1], x_vals]),
        np.concatenate([arc_y[:-1], y_vals]),
    )


def _build_endpoint_variants(x_vals, y_vals, at_end=False):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if x_vals.size < 2:
        return []

    curve_variants = [(x_vals, y_vals)]
    for axis in ("x", "y"):
        trimmed_x, trimmed_y, changed = _trim_endpoint_flat_run(x_vals, y_vals, axis=axis, at_end=at_end)
        if changed and trimmed_x.size >= 2:
            curve_variants.append((trimmed_x, trimmed_y))

    helper_variants = []
    seen_keys = set()
    for curve_x, curve_y in curve_variants:
        for fit_samples in PROFILE_ENDPOINT_FIT_WINDOWS:
            if curve_x.size < 2:
                continue
            helper = _build_endpoint_helper(curve_x, curve_y, at_end=at_end, fit_samples=min(fit_samples, curve_x.size))
            if helper is None:
                continue

            if helper["kind"] == "line":
                rounded = tuple(np.round(np.concatenate([helper["point"], helper["dir_out"]]), decimals=7))
                key = ("line", curve_x.size, rounded)
            else:
                rounded = tuple(
                    np.round(
                        np.concatenate([helper["point"], helper["center"], [helper["radius"], helper["turn_out_sign"]]]),
                        decimals=7,
                    )
                )
                key = ("circle", curve_x.size, rounded)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            helper_variants.append(
                {
                    "x_vals": curve_x,
                    "y_vals": curve_y,
                    "helper": helper,
                }
            )
    return helper_variants


def _connect_curve_endpoints(x_a, y_a, x_b, y_b, at_end_a=False, at_end_b=False):
    variants_a = _build_endpoint_variants(x_a, y_a, at_end=at_end_a)
    variants_b = _build_endpoint_variants(x_b, y_b, at_end=at_end_b)
    best_result = None
    best_cost = np.inf

    for variant_a in variants_a:
        for variant_b in variants_b:
            target_point = _find_shared_endpoint_target(variant_a["helper"], variant_b["helper"])
            if target_point is None or not np.all(np.isfinite(target_point)):
                continue
            prog_a = _helper_progress(variant_a["helper"], target_point)
            prog_b = _helper_progress(variant_b["helper"], target_point)
            if prog_a is None or prog_b is None:
                continue

            total_cost = float(prog_a + prog_b)
            if total_cost >= best_cost:
                continue

            x_a_ext, y_a_ext = _extend_curve_with_helper(
                variant_a["x_vals"],
                variant_a["y_vals"],
                variant_a["helper"],
                target_point,
                at_end=at_end_a,
            )
            x_b_ext, y_b_ext = _extend_curve_with_helper(
                variant_b["x_vals"],
                variant_b["y_vals"],
                variant_b["helper"],
                target_point,
                at_end=at_end_b,
            )
            best_cost = total_cost
            best_result = (x_a_ext, y_a_ext, x_b_ext, y_b_ext)

    return best_result


def _build_mode1_profile_from_side_points(side_points):
    if side_points.shape[0] < 8:
        return None

    r = np.asarray(side_points[:, 0], dtype=float)
    z = np.asarray(side_points[:, 1], dtype=float)
    r_max = float(np.max(r))
    if np.isclose(r_max, 0.0):
        return None

    r_edges = np.linspace(0.0, r_max, PROFILE_R_BINS + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_bin_idx = np.clip(np.digitize(r, r_edges) - 1, 0, len(r_centers) - 1)

    upper_z = np.full(r_centers.shape, np.nan, dtype=float)
    lower_z = np.full(r_centers.shape, np.nan, dtype=float)
    for i in range(len(r_centers)):
        row_points = z[r_bin_idx == i]
        if row_points.size > 0:
            upper_z[i] = np.max(row_points)
            lower_z[i] = np.min(row_points)

    valid = np.isfinite(upper_z) & np.isfinite(lower_z)
    if np.count_nonzero(valid) < 4:
        return None

    first = int(np.flatnonzero(valid)[0])
    last = int(np.flatnonzero(valid)[-1])
    r_valid = r_centers[first:last + 1]

    upper_raw = np.interp(r_valid, r_centers[valid], upper_z[valid])
    lower_raw = np.interp(r_valid, r_centers[valid], lower_z[valid])

    if PROFILE_CURVE_SMOOTH_SIGMA > 0:
        upper_smooth = gaussian_filter1d(upper_raw, sigma=PROFILE_CURVE_SMOOTH_SIGMA, mode="nearest")
        lower_smooth = gaussian_filter1d(lower_raw, sigma=PROFILE_CURVE_SMOOTH_SIGMA, mode="nearest")
    else:
        upper_smooth = upper_raw.copy()
        lower_smooth = lower_raw.copy()

    # Keep the upper/lower envelopes conservative with respect to sampled points.
    upper_smooth = np.maximum(upper_smooth, upper_raw)
    lower_smooth = np.minimum(lower_smooth, lower_raw)

    # Include the axis apex and the outermost sampled radius explicitly.
    axis_mask = r <= r_edges[1]
    axis_upper = float(np.max(z[axis_mask])) if np.any(axis_mask) else float(np.max(z))
    axis_lower = float(np.min(z[axis_mask])) if np.any(axis_mask) else float(np.min(z))
    outer_mask = r >= r_edges[-2]
    outer_upper = float(np.max(z[outer_mask])) if np.any(outer_mask) else float(upper_smooth[-1])
    outer_lower = float(np.min(z[outer_mask])) if np.any(outer_mask) else float(lower_smooth[-1])

    r_plot = np.concatenate(([0.0], r_valid, [r_max]))
    upper_plot = np.concatenate(([axis_upper], upper_smooth, [outer_upper]))
    lower_plot = np.concatenate(([axis_lower], lower_smooth, [outer_lower]))

    upper_raw_plot = np.concatenate(([axis_upper], upper_raw, [outer_upper]))
    lower_raw_plot = np.concatenate(([axis_lower], lower_raw, [outer_lower]))
    r_plot, upper_plot = _smooth_scalar_curve(
        r_plot,
        upper_plot,
        upsample_factor=6,
        sigma=max(1.0, 0.75 * PROFILE_CURVE_SMOOTH_SIGMA),
    )
    _, lower_plot = _smooth_scalar_curve(
        np.concatenate(([0.0], r_valid, [r_max])),
        lower_plot,
        upsample_factor=6,
        sigma=max(1.0, 0.75 * PROFILE_CURVE_SMOOTH_SIGMA),
    )

    upper_raw_dense = np.interp(r_plot, np.concatenate(([0.0], r_valid, [r_max])), upper_raw_plot)
    lower_raw_dense = np.interp(r_plot, np.concatenate(([0.0], r_valid, [r_max])), lower_raw_plot)
    upper_plot = np.maximum(upper_plot, upper_raw_dense)
    lower_plot = np.minimum(lower_plot, lower_raw_dense)
    upper_plot[0] = axis_upper
    lower_plot[0] = axis_lower
    upper_plot[-1] = outer_upper
    lower_plot[-1] = outer_lower

    void_floor_z = _build_mode1_void_floor(lower_plot)

    # Drawing helpers expect r as a function of z for regular modes; mode1 uses upper/lower z(r).
    profile = {
        "z": np.concatenate((lower_plot, upper_plot)),
        "outer_r": np.concatenate((r_plot, r_plot)),
        "inner_r": np.zeros_like(np.concatenate((r_plot, r_plot))),
        "mode1_r": r_plot,
        "mode1_upper_z": upper_plot,
        "mode1_lower_z": lower_plot,
    }
    if void_floor_z is not None:
        profile["mode1_void_r"] = r_plot
        profile["mode1_void_floor_z"] = float(void_floor_z)
    return profile


def _build_mode1_curve_envelope(r_samples, z_samples, num_bins=None):
    r_samples = np.asarray(r_samples, dtype=float)
    z_samples = np.asarray(z_samples, dtype=float)
    valid = np.isfinite(r_samples) & np.isfinite(z_samples)
    if np.count_nonzero(valid) < 4:
        return None, None

    r_samples = r_samples[valid]
    z_samples = z_samples[valid]
    max_r = float(np.max(r_samples))
    if not np.isfinite(max_r) or max_r <= 0.0:
        return None, None

    if num_bins is None:
        num_bins = max(220, 2 * PROFILE_R_BINS)
    num_bins = int(max(num_bins, 16))

    r_edges = np.linspace(0.0, max_r, num_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_bin_idx = np.clip(np.digitize(r_samples, r_edges) - 1, 0, len(r_centers) - 1)

    env_z = np.full(r_centers.shape, np.nan, dtype=float)
    for i in range(len(r_centers)):
        row_points = z_samples[r_bin_idx == i]
        if row_points.size > 0:
            env_z[i] = np.max(row_points)

    valid_bins = np.isfinite(env_z)
    if np.count_nonzero(valid_bins) < 4:
        return None, None

    first = int(np.flatnonzero(valid_bins)[0])
    last = int(np.flatnonzero(valid_bins)[-1])
    r_valid = r_centers[first:last + 1]
    env_raw = np.interp(r_valid, r_centers[valid_bins], env_z[valid_bins])

    if PROFILE_CURVE_SMOOTH_SIGMA > 0:
        env_smooth = gaussian_filter1d(env_raw, sigma=PROFILE_CURVE_SMOOTH_SIGMA, mode="nearest")
    else:
        env_smooth = env_raw.copy()

    env_smooth = np.maximum(env_smooth, env_raw)

    axis_mask = r_samples <= r_edges[1]
    axis_z = float(np.max(z_samples[axis_mask])) if np.any(axis_mask) else float(np.max(z_samples))
    outer_mask = r_samples >= r_edges[-2]
    outer_z = float(np.max(z_samples[outer_mask])) if np.any(outer_mask) else float(env_smooth[-1])

    r_plot = np.concatenate(([0.0], r_valid, [max_r]))
    z_plot = np.concatenate(([axis_z], env_smooth, [outer_z]))
    raw_plot = np.concatenate(([axis_z], env_raw, [outer_z]))
    r_plot, z_plot = _smooth_scalar_curve(
        r_plot,
        z_plot,
        upsample_factor=6,
        sigma=max(1.0, 0.75 * PROFILE_CURVE_SMOOTH_SIGMA),
    )
    raw_dense = np.interp(r_plot, np.concatenate(([0.0], r_valid, [max_r])), raw_plot)
    z_plot = np.maximum(z_plot, raw_dense)
    z_plot[0] = axis_z
    z_plot[-1] = outer_z
    return r_plot, z_plot


def _sin_over_theta(theta_vals):
    theta_vals = np.asarray(theta_vals, dtype=float)
    result = np.ones_like(theta_vals, dtype=float)
    nonzero = np.abs(theta_vals) > 1e-9
    result[nonzero] = np.sin(theta_vals[nonzero]) / theta_vals[nonzero]
    return result


def _one_minus_cos_over_theta(theta_vals):
    theta_vals = np.asarray(theta_vals, dtype=float)
    result = np.zeros_like(theta_vals, dtype=float)
    nonzero = np.abs(theta_vals) > 1e-9
    result[nonzero] = (1.0 - np.cos(theta_vals[nonzero])) / theta_vals[nonzero]
    return result


def _build_mode1_profile_from_geometry(csm, num_samples=None):
    theta_max = _segment_theta_upper_bound(csm, 2, csm.L_20)
    tool_length = float(csm.L_tool)
    kappa = float(csm.kappa_20)
    arc_length = float(csm.L_20)
    if arc_length <= 0.0:
        return None

    if num_samples is None:
        num_samples = max(600, 3 * PROFILE_R_BINS, 3 * MODE_SAMPLE_RES[1]["theta2"])
    num_samples = int(max(num_samples, 8))

    theta_vals = np.linspace(0.0, theta_max, num_samples)
    sin_over_theta = _sin_over_theta(theta_vals)
    one_minus_cos_over_theta = _one_minus_cos_over_theta(theta_vals)
    sin_theta = np.sin(theta_vals)
    cos_theta = np.cos(theta_vals)

    upper_r_samples = arc_length * one_minus_cos_over_theta + tool_length * sin_theta
    upper_z_samples = arc_length * sin_over_theta + tool_length * cos_theta

    if kappa <= 0.0:
        lower_r_samples = tool_length * sin_theta
        lower_z_samples = tool_length * cos_theta
    else:
        lower_r_samples = (1.0 - cos_theta) / kappa + tool_length * sin_theta
        lower_z_samples = sin_theta / kappa + tool_length * cos_theta

    upper_r_samples[0] = 0.0
    upper_z_samples[0] = arc_length + tool_length
    lower_r_samples[0] = 0.0
    lower_z_samples[0] = tool_length

    void_r, void_upper_z = _build_mode1_curve_envelope(lower_r_samples, lower_z_samples, num_bins=num_samples // 2)
    if void_r is None or void_upper_z is None:
        return None

    void_floor_z = _build_mode1_void_floor(lower_z_samples)
    profile = {
        "z": np.concatenate((lower_z_samples, upper_z_samples)),
        "outer_r": np.concatenate((upper_r_samples, upper_r_samples)),
        "inner_r": np.zeros_like(np.concatenate((lower_r_samples, upper_r_samples))),
        "mode1_r": upper_r_samples,
        "mode1_upper_r": upper_r_samples,
        "mode1_lower_r": lower_r_samples,
        "mode1_upper_z": upper_z_samples,
        "mode1_lower_z": lower_z_samples,
        "mode1_analytic": True,
    }
    if void_floor_z is not None:
        profile["mode1_void_r"] = void_r
        profile["mode1_void_upper_z"] = void_upper_z
        profile["mode1_void_floor_z"] = float(void_floor_z)
    return profile


def _build_mode1_void_floor(lower_z_vals):
    lower_z_vals = np.asarray(lower_z_vals, dtype=float)
    if lower_z_vals.size < 4:
        return None

    finite = lower_z_vals[np.isfinite(lower_z_vals)]
    if finite.size < 4:
        return None

    floor_z = float(np.min(finite))
    if np.max(lower_z_vals - floor_z) <= 1e-6:
        return None
    return floor_z


def _build_regular_profile_from_side_points(side_points):
    if side_points.shape[0] < 8:
        return None

    r = side_points[:, 0]
    z = side_points[:, 1]
    z_min, z_max = z.min(), z.max()
    r_max = r.max()
    if np.isclose(z_min, z_max) or np.isclose(r_max, 0.0):
        return None

    z_edges = np.linspace(z_min, z_max, PROFILE_Z_BINS + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    outer_r = np.full(z_centers.shape, np.nan, dtype=float)
    inner_r = np.zeros(z_centers.shape, dtype=float)

    z_bin_idx = np.clip(np.digitize(z, z_edges) - 1, 0, len(z_centers) - 1)
    row_point_max = np.full(z_centers.shape, np.nan, dtype=float)
    row_point_min = np.full(z_centers.shape, np.nan, dtype=float)
    for i in range(len(z_centers)):
        row_points = r[z_bin_idx == i]
        if row_points.size > 0:
            row_point_max[i] = np.max(row_points)
            row_point_min[i] = np.min(row_points)

    valid_outer = np.isfinite(row_point_max)
    if np.count_nonzero(valid_outer) < 4:
        return None

    first_outer = int(np.flatnonzero(valid_outer)[0])
    last_outer = int(np.flatnonzero(valid_outer)[-1])
    outer_core = np.interp(
        z_centers[first_outer:last_outer + 1],
        z_centers[valid_outer],
        row_point_max[valid_outer],
    )
    z_valid = z_centers[first_outer:last_outer + 1]

    outer_valid = outer_core.copy()
    outer_raw = outer_core.copy()
    if PROFILE_CURVE_SMOOTH_SIGMA > 0:
        outer_valid = gaussian_filter1d(outer_valid, sigma=PROFILE_CURVE_SMOOTH_SIGMA, mode="nearest")
    outer_valid = np.maximum(outer_valid, outer_raw)

    outer_valid = np.concatenate(([row_point_max[first_outer]], outer_valid, [row_point_max[last_outer]]))
    z_valid = np.concatenate(([z_min], z_valid, [z_max]))

    inner_candidate = np.full(z_centers.shape, np.nan, dtype=float)
    for i in np.flatnonzero(valid_outer):
        outer_here = row_point_max[i]
        inner_here = row_point_min[i]
        if inner_here <= 0 or outer_here <= 0:
            continue
        if inner_here / outer_here < PROFILE_MIN_VOID_RATIO:
            continue
        inner_candidate[i] = inner_here

    valid_inner = np.isfinite(inner_candidate)
    if np.count_nonzero(valid_inner) >= PROFILE_MIN_INNER_RUN:
        inner_keep = _longest_true_run(valid_inner, np.nan_to_num(inner_candidate, nan=0.0) ** 2)
        valid_inner = valid_inner & inner_keep

    if np.count_nonzero(valid_inner) >= 2:
        first_inner = int(np.flatnonzero(valid_inner)[0])
        last_inner = int(np.flatnonzero(valid_inner)[-1])
        inner_z = z_centers[first_inner:last_inner + 1]
        inner_raw = np.interp(
            inner_z,
            z_centers[valid_inner],
            inner_candidate[valid_inner],
        )
        inner_segment = inner_raw.copy()
        if PROFILE_CURVE_SMOOTH_SIGMA > 0:
            inner_segment = gaussian_filter1d(inner_segment, sigma=PROFILE_CURVE_SMOOTH_SIGMA, mode="nearest")
        inner_segment = np.minimum(inner_segment, inner_raw)

        inner_segment_full = np.interp(
            z_centers[first_inner:last_inner + 1],
            inner_z,
            inner_segment,
            left=np.nan,
            right=np.nan,
        )
        write_count = min(inner_segment_full.size, inner_r[first_inner:].size)
        finite_write = np.isfinite(inner_segment_full[:write_count])
        target_slice = inner_r[first_inner:first_inner + write_count]
        target_slice[finite_write] = inner_segment_full[:write_count][finite_write]
        inner_r[first_inner:first_inner + write_count] = target_slice

    radial_res = r_max / max(PROFILE_R_BINS, 1)
    inner_valid = np.interp(z_valid, z_centers, inner_r, left=0.0, right=0.0)
    inner_upper = np.maximum(outer_valid - 0.5 * radial_res, 0.0)
    inner_valid = np.minimum(np.maximum(inner_valid, 0.0), inner_upper)
    inner_valid[inner_valid < 0.5 * radial_res] = 0.0
    inner_keep = _longest_true_run(inner_valid > 0.0, inner_valid ** 2)
    inner_valid = np.where(inner_keep, inner_valid, 0.0)

    return {
        "z": z_valid,
        "outer_r": outer_valid,
        "inner_r": inner_valid,
    }


def _build_mode0_profile_from_side_points(side_points):
    regular_profile = _build_regular_profile_from_side_points(side_points)
    if regular_profile is None:
        return None

    r = np.asarray(side_points[:, 0], dtype=float)
    z = np.asarray(side_points[:, 1], dtype=float)
    r_max = float(np.max(r))
    z_floor = float(np.min(z))
    if np.isclose(r_max, 0.0):
        return regular_profile

    r_edges = np.linspace(0.0, r_max, PROFILE_R_BINS + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_bin_idx = np.clip(np.digitize(r, r_edges) - 1, 0, len(r_centers) - 1)

    lower_z = np.full(r_centers.shape, np.nan, dtype=float)
    for i in range(len(r_centers)):
        col_points = z[r_bin_idx == i]
        if col_points.size > 0:
            lower_z[i] = np.min(col_points)

    valid = np.isfinite(lower_z)
    if np.count_nonzero(valid) < 4:
        return regular_profile

    lower_raw = np.interp(r_centers, r_centers[valid], lower_z[valid])
    lower_smooth = lower_raw.copy()
    if PROFILE_CURVE_SMOOTH_SIGMA > 0:
        lower_smooth = gaussian_filter1d(lower_smooth, sigma=0.85 * PROFILE_CURVE_SMOOTH_SIGMA, mode="nearest")
    lower_smooth = np.minimum(lower_smooth, lower_raw)

    radial_res = r_max / max(PROFILE_R_BINS, 1)
    z_range = max(float(np.max(z) - z_floor), 1e-9)
    z_tol = max(0.5 * radial_res, PROFILE_MODE0_VOID_MIN_HEIGHT_BINS * z_range / max(PROFILE_Z_BINS, 1))
    void_mask = lower_smooth > z_floor + z_tol
    if np.count_nonzero(void_mask) < 4:
        return regular_profile

    void_keep = _longest_true_run(void_mask, np.maximum(lower_smooth - z_floor, 0.0))
    if np.count_nonzero(void_keep) < 4:
        return regular_profile

    first = int(np.flatnonzero(void_keep)[0])
    last = int(np.flatnonzero(void_keep)[-1])
    void_r_core = r_centers[first:last + 1]
    void_upper_core = lower_smooth[first:last + 1]

    void_r = np.concatenate(([0.0], void_r_core))
    axis_mask = r <= r_edges[1]
    axis_lower = float(np.min(z[axis_mask])) if np.any(axis_mask) else float(np.min(z))
    void_upper = np.concatenate(([axis_lower], void_upper_core))
    void_upper = np.maximum(void_upper, z_floor)

    regular_profile["mode0_void_r"] = void_r
    regular_profile["mode0_void_upper_z"] = void_upper
    regular_profile["mode0_void_floor_z"] = float(z_floor)
    regular_profile["inner_r"] = np.zeros_like(regular_profile["inner_r"])
    return regular_profile


def _curve_points(r_vals, z_vals):
    r_vals = np.asarray(r_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)
    if r_vals.size == 0 or z_vals.size == 0 or r_vals.size != z_vals.size:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((r_vals, z_vals))


def _trim_mode0_outer_curve_prefix(mode0_outer_curve, source_profiles):
    curve = np.asarray(mode0_outer_curve, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 3 or curve.shape[1] != 2:
        return curve
    if not source_profiles:
        return curve

    source_outer_curves = {}
    global_max_r = 0.0
    for mode, profile in source_profiles.items():
        z_vals, r_vals, _, _ = _build_plot_curves(profile, mode)
        outer_curve = _curve_points(r_vals, z_vals)
        if outer_curve.shape[0] < 2:
            continue
        source_outer_curves[mode] = outer_curve
        global_max_r = max(global_max_r, float(np.max(outer_curve[:, 0])))

    if global_max_r <= 0.0 or not source_outer_curves:
        return curve

    shell_modes = [
        outer_curve
        for outer_curve in source_outer_curves.values()
        if float(np.max(outer_curve[:, 0])) >= 0.98 * global_max_r
    ]
    if not shell_modes:
        return curve

    shell_min_z = min(float(np.min(shell_curve[:, 1])) for shell_curve in shell_modes)
    start_z = float(curve[0, 1])
    if start_z >= shell_min_z - 1e-6:
        return curve

    start_r = float(curve[0, 0])
    shell_start_r = max(
        float(shell_curve[np.argmin(shell_curve[:, 1]), 0])
        for shell_curve in shell_modes
    )
    if start_r >= 0.9 * shell_start_r:
        return curve

    z_vals = curve[:, 1]
    r_vals = curve[:, 0]
    keep_mask = z_vals >= shell_min_z
    if not np.any(keep_mask):
        return curve

    first_keep = int(np.flatnonzero(keep_mask)[0])
    trimmed = curve[first_keep:].copy()
    if first_keep > 0 and not np.isclose(trimmed[0, 1], shell_min_z, atol=1e-9):
        shell_r = max(
            float(np.interp(shell_min_z, shell_curve[:, 1], shell_curve[:, 0]))
            for shell_curve in shell_modes
        )
        trimmed = np.vstack((np.array([[shell_r, shell_min_z]], dtype=float), trimmed))
    elif trimmed.shape[0] > 0:
        shell_r = max(
            float(shell_curve[np.argmin(shell_curve[:, 1]), 0])
            for shell_curve in shell_modes
        )
        if trimmed[0, 0] < shell_r:
            trimmed[0, 0] = shell_r
    return trimmed


def _closest_curve_points(curve_a, curve_b):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    if curve_a.ndim != 2 or curve_b.ndim != 2 or curve_a.shape[1] != 2 or curve_b.shape[1] != 2:
        return None
    if curve_a.shape[0] == 0 or curve_b.shape[0] == 0:
        return None

    delta = curve_a[:, np.newaxis, :] - curve_b[np.newaxis, :, :]
    dist_sq = np.sum(delta * delta, axis=2)
    idx_a, idx_b = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    point = 0.5 * (curve_a[idx_a] + curve_b[idx_b])
    return int(idx_a), int(idx_b), point


def _intersect_endpoint_helper_with_vertical(helper, r_value):
    if helper is None:
        return None

    r_value = float(r_value)
    tol = 1e-9

    if helper["kind"] == "line":
        dir_x = float(helper["dir_out"][0])
        if abs(dir_x) <= tol:
            if abs(float(helper["point"][0]) - r_value) > 1e-6:
                return None
            return np.array([r_value, float(helper["point"][1])], dtype=float)
        length = (r_value - float(helper["point"][0])) / dir_x
        if length < -1e-9:
            return None
        return helper["point"] + max(length, 0.0) * helper["dir_out"]

    dx = r_value - float(helper["center"][0])
    radius_sq = float(helper["radius"] * helper["radius"])
    rem = radius_sq - dx * dx
    if rem < -1e-9:
        return None
    rem = max(rem, 0.0)
    dz = float(np.sqrt(rem))
    candidates = [
        np.array([r_value, float(helper["center"][1]) + dz], dtype=float),
        np.array([r_value, float(helper["center"][1]) - dz], dtype=float),
    ]

    best_point = None
    best_progress = np.inf
    for candidate in candidates:
        progress = _helper_progress(helper, candidate)
        if progress is None:
            continue
        if progress < best_progress:
            best_progress = progress
            best_point = candidate
    return best_point


def _concat_curve_segments(segments):
    merged = []
    for segment in segments:
        segment = np.asarray(segment, dtype=float)
        if segment.ndim != 2 or segment.shape[0] == 0:
            continue
        if merged and np.allclose(merged[-1][-1], segment[0], atol=1e-9):
            merged.append(segment[1:])
        else:
            merged.append(segment)

    if not merged:
        return np.empty((0, 2), dtype=float)
    return np.vstack(merged)


def _should_prepend_mode0_outer_join(curve, join_point):
    curve = np.asarray(curve, dtype=float)
    join_point = np.asarray(join_point, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] != 2 or join_point.shape != (2,):
        return False

    start = curve[0]
    z_span = float(np.max(curve[:, 1]) - np.min(curve[:, 1]))
    r_span = float(np.max(curve[:, 0]) - np.min(curve[:, 0]))
    z_tol = max(1e-6, 2e-3 * max(z_span, 1.0))
    r_tol = max(1e-6, 2e-3 * max(r_span, 1.0))

    # Only keep the synthetic connector when it behaves like a smooth extension
    # of the original outer curve instead of pushing the envelope outward.
    if join_point[1] > start[1] + z_tol:
        return True
    return abs(join_point[1] - start[1]) <= z_tol and join_point[0] <= start[0] + r_tol


def _extract_axis_flat_segment_near_max_radius(curve, tol=1e-5, min_points=4):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 2:
        return curve

    r_vals = curve[:, 0]
    max_r = float(np.max(r_vals))

    # Small workspaces can turn the ideal vertical segment into a lightly
    # jittered band, so use an adaptive tolerance around the outer endpoint.
    endpoint_window = r_vals[:min(24, r_vals.size)]
    endpoint_jitter = 0.0
    if endpoint_window.size >= 3:
        endpoint_jitter = float(np.percentile(np.abs(np.diff(endpoint_window)), 90))
    adaptive_tol = max(float(tol), 2.5 * endpoint_jitter, 5e-3 * max_r)

    mask = r_vals >= max_r - adaptive_tol
    if not np.any(mask):
        return curve.copy()

    prefix_false = np.flatnonzero(~mask)
    if prefix_false.size == 0:
        prefix_len = curve.shape[0]
    else:
        prefix_len = int(prefix_false[0])
    if prefix_len >= 2:
        return curve[:prefix_len].copy()

    indices = np.flatnonzero(mask)
    splits = np.where(np.diff(indices) > 1)[0]
    best_run = indices[:1]
    start_idx = 0
    for split_idx in list(splits) + [len(indices) - 1]:
        run = indices[start_idx:split_idx + 1]
        if run.size > best_run.size:
            best_run = run
        start_idx = split_idx + 1

    if best_run.size >= 2:
        return curve[int(best_run[0]):int(best_run[-1]) + 1].copy()

    fallback_len = min(max(int(min_points), 2), curve.shape[0])
    return curve[:fallback_len].copy()


def _build_mode0_void_outline_from_source_profiles(source_profiles, mode0_outer_curve=None):
    if not all(mode in source_profiles for mode in COMBINED_SOURCE_MODES):
        return None

    _upper_r, _upper_z, mode1_lower_r, mode1_lower_z = _build_mode1_plot_curves(source_profiles[1])
    mode2_z, mode2_outer_r, _, _ = _build_plot_curves(source_profiles[2], 2)
    mode4_outer_z, mode4_outer_r, _, _ = _build_plot_curves(source_profiles[4], 4)
    _, _, mode3_inner_z, mode3_inner_r = _build_plot_curves(source_profiles[3], 3)
    _, _, mode4_inner_z, mode4_inner_r = _build_plot_curves(source_profiles[4], 4)

    if mode3_inner_z is None or mode4_inner_z is None:
        return None

    curve1 = _curve_points(mode1_lower_r, mode1_lower_z)
    curve2 = _curve_points(mode2_outer_r, mode2_z)  # outer -> axis
    curve2 = _extract_axis_flat_segment_near_max_radius(curve2)
    curve3 = _curve_points(mode3_inner_r[::-1], mode3_inner_z[::-1])  # axis -> outer
    curve4 = _curve_points(mode4_inner_r, mode4_inner_z)[::-1].copy()  # axis -> outer
    outer_curve = _curve_points(mode4_outer_r, mode4_outer_z)  # outer -> axis
    target_outer_curve = outer_curve
    if mode0_outer_curve is not None:
        candidate_curve = np.asarray(mode0_outer_curve, dtype=float)
        if candidate_curve.ndim == 2 and candidate_curve.shape[0] >= 2 and candidate_curve.shape[1] == 2:
            target_outer_curve = candidate_curve

    if curve1.shape[0] < 2 or curve2.shape[0] < 2:
        return None

    point12 = 0.5 * (curve1[-1] + curve2[0])
    seg1 = curve1.copy()
    seg1[-1] = point12

    seg2_seed = curve2.copy()
    seg2_seed[0] = point12
    join23 = _closest_curve_points(seg2_seed, curve3)
    if join23 is None:
        return None
    idx2_end, idx3_start, point23 = join23
    seg2 = seg2_seed[:idx2_end + 1].copy()
    seg2[-1] = point23
    seg3_seed = curve3[idx3_start:].copy()
    seg3_seed[0] = point23

    if seg2.shape[0] < 2 or seg3_seed.shape[0] < 2:
        return None

    join34 = _closest_curve_points(seg3_seed, curve4)
    if join34 is None:
        return None
    idx3_end, idx4_start, point34 = join34
    seg3 = seg3_seed[:idx3_end + 1].copy()
    seg3[-1] = point34

    seg4_seed = curve4[idx4_start:].copy()
    if seg4_seed.shape[0] < 2:
        return None
    seg4_seed[0] = point34
    seg4 = seg4_seed
    if seg4.shape[0] < 2:
        return None

    outer_helper = _build_endpoint_helper(target_outer_curve[:, 0], target_outer_curve[:, 1], at_end=False)
    tail_window = seg4[-min(12, seg4.shape[0]):]
    vertical_r = float(np.min(tail_window[:, 0]))
    join_point = _intersect_endpoint_helper_with_vertical(outer_helper, vertical_r)

    if join_point is None or not np.all(np.isfinite(join_point)):
        join_point = np.asarray([vertical_r, float(target_outer_curve[0, 1])], dtype=float)

    flat_idx = np.flatnonzero(np.isclose(seg4[:, 0], vertical_r, atol=1e-9))
    if flat_idx.size > 0:
        seg4 = seg4[:flat_idx[-1] + 1].copy()
    if not np.allclose(seg4[-1], join_point, atol=1e-9):
        seg4 = np.vstack((seg4, join_point[np.newaxis, :]))
    else:
        seg4[-1] = join_point
    if _should_prepend_mode0_outer_join(target_outer_curve, join_point):
        target_outer_curve = np.vstack((join_point[np.newaxis, :], target_outer_curve))

    outline = _concat_curve_segments((seg1, seg2, seg3, seg4))
    if outline.shape[0] < 4:
        return None
    return outline, target_outer_curve


def _build_mode0_profile_from_sources(side_points, source_profiles):
    profile = _build_mode0_profile_from_side_points(side_points)
    if profile is None:
        return None

    mode0_outer_z, mode0_outer_r, _, _ = _build_plot_curves(profile, 0)
    mode0_outer_curve = _curve_points(mode0_outer_r, mode0_outer_z)
    mode0_outer_curve = _trim_mode0_outer_curve_prefix(mode0_outer_curve, source_profiles)
    built = _build_mode0_void_outline_from_source_profiles(
        source_profiles,
        mode0_outer_curve=mode0_outer_curve,
    )
    if built is None:
        profile["mode0_outer_r"] = mode0_outer_curve[:, 0]
        profile["mode0_outer_z"] = mode0_outer_curve[:, 1]
        return profile
    outline, outer_override = built

    profile["mode0_void_outline_r"] = outline[:, 0]
    profile["mode0_void_outline_z"] = outline[:, 1]
    profile["mode0_void_floor_z"] = float(np.min(outline[:, 1]))
    profile["mode0_outer_r"] = outer_override[:, 0]
    profile["mode0_outer_z"] = outer_override[:, 1]
    return profile


def _build_mode1_plot_curves(profile):
    upper_r = np.asarray(profile.get("mode1_upper_r", profile["mode1_r"]), dtype=float)
    lower_r = np.asarray(profile.get("mode1_lower_r", profile["mode1_r"]), dtype=float)
    upper_z = np.asarray(profile["mode1_upper_z"], dtype=float)
    lower_z = np.asarray(profile["mode1_lower_z"], dtype=float)

    if bool(profile.get("mode1_analytic", False)):
        return upper_r, upper_z, lower_r, lower_z

    connected = _connect_curve_endpoints(
        upper_r,
        upper_z,
        lower_r,
        lower_z,
        at_end_a=True,
        at_end_b=True,
    )
    if connected is None:
        return upper_r, upper_z, lower_r, lower_z
    return connected


def _build_mode1_void_curve(profile):
    if "mode1_void_r" not in profile or "mode1_void_floor_z" not in profile:
        return None, None, None

    void_r = np.asarray(profile["mode1_void_r"], dtype=float)
    void_floor_z = float(profile["mode1_void_floor_z"])
    if void_r.size < 4 or not np.isfinite(void_floor_z):
        return None, None, None
    if "mode1_void_upper_z" in profile:
        void_upper_z = np.asarray(profile["mode1_void_upper_z"], dtype=float)
        if void_upper_z.size == void_r.size:
            return void_r, void_upper_z, void_floor_z
    _upper_r, _upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
    void_upper_z = np.interp(void_r, lower_r, lower_z)
    return void_r, void_upper_z, void_floor_z


def _build_mode1_void_tail(profile):
    _upper_r, _upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
    if lower_r.size < 3 or lower_z.size != lower_r.size:
        return None, None

    tail_start = int(np.argmax(lower_r))
    tail_r = np.asarray(lower_r[tail_start:], dtype=float)
    tail_z = np.asarray(lower_z[tail_start:], dtype=float)
    if tail_r.size < 2 or tail_z.size != tail_r.size:
        return None, None
    return tail_r, tail_z


def _mode1_outer_lower_draw_curve(profile):
    _upper_r, _upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
    tail_r, tail_z = _build_mode1_void_tail(profile)
    if tail_r is None or tail_z is None or tail_r.size < 2:
        return lower_r, lower_z

    split_idx = max(1, len(lower_r) - len(tail_r))
    return lower_r[:split_idx], lower_z[:split_idx]


def _mode1_void_polygon(profile):
    void_r, void_upper_z, void_floor_z = _build_mode1_void_curve(profile)
    if void_r is None or void_upper_z is None or void_r.size < 4:
        return None

    tail_r, tail_z = _build_mode1_void_tail(profile)
    if tail_r is not None and tail_z is not None:
        polygon_r = np.concatenate([void_r, tail_r, [0.0]])
        polygon_z = np.concatenate([void_upper_z, tail_z, [void_floor_z]])
        return polygon_r, polygon_z, void_floor_z

    polygon_r = np.concatenate([void_r, [void_r[-1], 0.0]])
    polygon_z = np.concatenate([void_upper_z, [void_floor_z, void_floor_z]])
    return polygon_r, polygon_z, void_floor_z


def _build_plot_curves(profile, mode):
    if mode == 0 and "mode0_outer_z" in profile and "mode0_outer_r" in profile:
        z_outer = np.asarray(profile["mode0_outer_z"], dtype=float)
        r_outer = np.asarray(profile["mode0_outer_r"], dtype=float)
        return z_outer, r_outer, None, None

    if mode == 1 and "mode1_r" in profile:
        return _build_mode1_plot_curves(profile)

    z_outer = np.asarray(profile["z"], dtype=float)
    r_outer = np.asarray(profile["outer_r"], dtype=float)

    inner_mask = np.asarray(profile["inner_r"], dtype=float) > 0.0
    if np.count_nonzero(inner_mask) < 3:
        if mode == 0:
            z_outer, r_outer = _append_axis_arc(
                z_outer,
                r_outer,
                max_extend_bins=PROFILE_INNER_APEX_MAX_EXTEND_BINS,
            )
        return z_outer, r_outer, None, None

    z_inner = np.asarray(profile["z"][inner_mask], dtype=float)
    r_inner = np.asarray(profile["inner_r"][inner_mask], dtype=float)

    if mode == 1:
        z_outer, r_outer = _fit_axis_arc_from_endpoint(z_outer, r_outer, axis_side="upper")
        z_outer, r_outer = _smooth_curve_for_plot(z_outer, r_outer, upsample_factor=5, sigma=1.0)
        z_inner, r_inner = _fit_axis_arc_from_endpoint(z_inner, r_inner, axis_side="lower")
        z_inner, r_inner = _smooth_curve_for_plot(z_inner, r_inner, upsample_factor=5, sigma=0.8)
        return z_outer, r_outer, z_inner, r_inner

    z_outer, r_outer = _append_axis_arc(
        z_outer,
        r_outer,
        max_extend_bins=PROFILE_INNER_APEX_MAX_EXTEND_BINS,
    )
    z_inner, r_inner = _append_axis_arc(
        z_inner,
        r_inner,
        max_extend_bins=PROFILE_INNER_APEX_MAX_EXTEND_BINS,
    )

    connected = _connect_curve_endpoints(r_outer, z_outer, r_inner, z_inner, at_end_a=False, at_end_b=False)
    if connected is not None:
        r_outer, z_outer, r_inner, z_inner = connected

    return z_outer, r_outer, z_inner, r_inner


def build_profile_from_side_points(side_points, mode=None, csm=None):
    if mode == 1 and csm is not None:
        return _build_mode1_profile_from_geometry(csm)
    if mode == 1:
        return _build_mode1_profile_from_side_points(side_points)
    if mode == 0:
        return _build_mode0_profile_from_side_points(side_points)
    return _build_regular_profile_from_side_points(side_points)


def _build_revolve_faces(n_axial, n_theta):
    faces = []
    for i in range(n_axial - 1):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            a = i * n_theta + j
            b = i * n_theta + jn
            c = (i + 1) * n_theta + j
            d = (i + 1) * n_theta + jn
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(faces, dtype=int)


def _decimate_curve(z_vals, r_vals, max_axial_samples=120):
    if len(z_vals) <= max_axial_samples:
        return z_vals, r_vals
    sample_idx = np.linspace(0, len(z_vals) - 1, max_axial_samples).astype(int)
    sample_idx = np.unique(sample_idx)
    return z_vals[sample_idx], r_vals[sample_idx]


def plot_revolved_profile(ax, z_vals, r_vals, color, alpha, label=None, cap_ends=False):
    z_vals, r_vals = _decimate_curve(np.asarray(z_vals), np.asarray(r_vals))
    theta = np.linspace(0.0, 2.0 * np.pi, REVOLVE_SAMPLES, endpoint=False)
    theta_grid, z_grid = np.meshgrid(theta, z_vals)
    r_grid = np.repeat(r_vals[:, np.newaxis], theta.size, axis=1)

    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    verts = np.column_stack((x.ravel(), y.ravel(), z_grid.ravel()))
    faces = _build_revolve_faces(len(z_vals), theta.size)
    surface = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        surface.set_edgecolor((0, 0, 0, 0))
        surface.set_zsort("min")
    except Exception:
        pass

    if cap_ends:
        for z_cap, r_cap in ((z_vals[0], r_vals[0]), (z_vals[-1], r_vals[-1])):
            if r_cap <= 0:
                continue
            radial = np.linspace(0.0, r_cap, CAP_RADIAL_SAMPLES)
            theta_cap, radial_cap = np.meshgrid(theta, radial)
            x_cap = radial_cap * np.cos(theta_cap)
            y_cap = radial_cap * np.sin(theta_cap)
            z_cap_grid = np.full_like(x_cap, z_cap)
            verts_cap = np.column_stack((x_cap.ravel(), y_cap.ravel(), z_cap_grid.ravel()))
            faces_cap = _build_revolve_faces(len(radial), theta.size)
            cap = ax.plot_trisurf(
                verts_cap[:, 0],
                verts_cap[:, 1],
                verts_cap[:, 2],
                triangles=faces_cap,
                color=color,
                alpha=alpha,
                linewidth=0,
                edgecolor="none",
                antialiased=True,
                shade=False,
            )
            try:
                cap.set_edgecolor((0, 0, 0, 0))
                cap.set_zsort("min")
            except Exception:
                pass

    if label:
        ax.plot([], [], [], color=color, alpha=alpha, label=label)


def _plot_horizontal_disk(ax, z_level, radius, color, alpha):
    radius = float(radius)
    if radius <= 0.0:
        return

    theta = np.linspace(0.0, 2.0 * np.pi, REVOLVE_SAMPLES, endpoint=False)
    radial = np.linspace(0.0, radius, CAP_RADIAL_SAMPLES)
    theta_grid, radial_grid = np.meshgrid(theta, radial)
    x = radial_grid * np.cos(theta_grid)
    y = radial_grid * np.sin(theta_grid)
    z = np.full_like(x, float(z_level))
    verts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    faces = _build_revolve_faces(len(radial), theta.size)
    disk = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        disk.set_edgecolor((0, 0, 0, 0))
        disk.set_zsort("min")
    except Exception:
        pass


def _plot_vertical_revolve_wall(ax, radius, z_start, z_end, color, alpha):
    radius = float(radius)
    z_start = float(z_start)
    z_end = float(z_end)
    if radius <= 0.0 or np.isclose(z_start, z_end):
        return

    theta = np.linspace(0.0, 2.0 * np.pi, REVOLVE_SAMPLES, endpoint=False)
    z_vals = np.linspace(min(z_start, z_end), max(z_start, z_end), CAP_RADIAL_SAMPLES)
    theta_grid, z_grid = np.meshgrid(theta, z_vals)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    verts = np.column_stack((x.ravel(), y.ravel(), z_grid.ravel()))
    faces = _build_revolve_faces(len(z_vals), theta.size)
    wall = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        wall.set_edgecolor((0, 0, 0, 0))
        wall.set_zsort("min")
    except Exception:
        pass


def _plot_profile_wall(ax, z_vals, r_vals, color, alpha, label=None, y_plane=0.0):
    z_vals, r_vals = _decimate_curve(np.asarray(z_vals), np.asarray(r_vals))
    x_poly = np.concatenate([r_vals, -r_vals[::-1]])
    z_poly = np.concatenate([z_vals, z_vals[::-1]])
    y_poly = np.full_like(x_poly, y_plane)
    verts = np.column_stack((x_poly, y_poly, z_poly))
    poly = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=np.array([[0, i, i + 1] for i in range(1, len(verts) - 1)], dtype=int),
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        poly.set_edgecolor((0, 0, 0, 0))
        poly.set_zsort("min")
    except Exception:
        pass
    if label:
        ax.plot([], [], [], color=color, alpha=alpha, label=label)


def _pseudo_plane_basis(elev_deg=PSEUDO_VIEW_ELEV, azim_deg=PSEUDO_VIEW_AZIM):
    elev = np.deg2rad(float(elev_deg))
    azim = np.deg2rad(float(azim_deg))
    right = np.array([-np.sin(azim), np.cos(azim), 0.0], dtype=float)
    up = np.array(
        [-np.sin(elev) * np.cos(azim), -np.sin(elev) * np.sin(azim), np.cos(elev)],
        dtype=float,
    )
    return right, up


def _embed_pseudo_plane_polygon(x_vals, z_vals):
    x_vals = np.asarray(x_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)
    if x_vals.size == 0 or z_vals.size != x_vals.size:
        return np.empty((0, 3), dtype=float)

    right, up = _pseudo_plane_basis()
    return x_vals[:, None] * right[None, :] + z_vals[:, None] * up[None, :]


def _plot_pseudo_plane_polygon(ax, x_vals, z_vals, color, alpha, label=None):
    verts = _embed_pseudo_plane_polygon(x_vals, z_vals)
    if verts.shape[0] < 3:
        return None

    poly = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=np.array([[0, i, i + 1] for i in range(1, len(verts) - 1)], dtype=int),
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        poly.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass
    if label:
        ax.plot([], [], [], color=color, alpha=alpha, label=label)
    return poly


def _plot_profile_on_pseudo_plane(ax, z_vals, r_vals, color, alpha, label=None):
    z_vals, r_vals = _decimate_curve(np.asarray(z_vals), np.asarray(r_vals))
    x_poly = np.concatenate([r_vals, -r_vals[::-1]])
    z_poly = np.concatenate([z_vals, z_vals[::-1]])
    return _plot_pseudo_plane_polygon(ax, x_poly, z_poly, color=color, alpha=alpha, label=label)


def _plot_symmetric_band_on_pseudo_plane(ax, r_vals, lower_z_vals, upper_z_vals, color, alpha, label=None):
    r_vals = np.asarray(r_vals, dtype=float)
    lower_z_vals = np.asarray(lower_z_vals, dtype=float)
    upper_z_vals = np.asarray(upper_z_vals, dtype=float)
    if r_vals.size < 2 or lower_z_vals.size != r_vals.size or upper_z_vals.size != r_vals.size:
        return None

    x_poly = np.concatenate([r_vals, r_vals[::-1], -r_vals, -r_vals[::-1]])
    z_poly = np.concatenate([upper_z_vals, lower_z_vals[::-1], lower_z_vals, upper_z_vals[::-1]])
    return _plot_pseudo_plane_polygon(ax, x_poly, z_poly, color=color, alpha=alpha, label=label)


def _plot_symmetric_band_wall(ax, r_vals, lower_z_vals, upper_z_vals, color, alpha, label=None, y_plane=0.0):
    r_vals = np.asarray(r_vals, dtype=float)
    lower_z_vals = np.asarray(lower_z_vals, dtype=float)
    upper_z_vals = np.asarray(upper_z_vals, dtype=float)
    if r_vals.size < 2 or lower_z_vals.size != r_vals.size or upper_z_vals.size != r_vals.size:
        return

    x_poly = np.concatenate([r_vals, r_vals[::-1], -r_vals, -r_vals[::-1]])
    z_poly = np.concatenate([upper_z_vals, lower_z_vals[::-1], lower_z_vals, upper_z_vals[::-1]])
    y_poly = np.full_like(x_poly, y_plane)
    verts = np.column_stack((x_poly, y_poly, z_poly))
    poly = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=np.array([[0, i, i + 1] for i in range(1, len(verts) - 1)], dtype=int),
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        poly.set_edgecolor((0, 0, 0, 0))
        poly.set_zsort("min")
    except Exception:
        pass
    if label:
        ax.plot([], [], [], color=color, alpha=alpha, label=label)


def _mode0_void_polygon(profile):
    if "mode0_void_outline_r" not in profile or "mode0_void_outline_z" not in profile:
        return None

    outline_r = np.asarray(profile["mode0_void_outline_r"], dtype=float)
    outline_z = np.asarray(profile["mode0_void_outline_z"], dtype=float)
    if outline_r.size < 2 or outline_z.size != outline_r.size:
        return None

    floor_z = float(profile.get("mode0_void_floor_z", np.min(outline_z)))
    polygon_r = np.concatenate([outline_r, [outline_r[-1], 0.0]])
    polygon_z = np.concatenate([outline_z, [floor_z, floor_z]])
    return polygon_r, polygon_z, floor_z


def _plot_mode0_void_wall(ax, profile, color, alpha, label=None, y_plane=0.0):
    polygon = _mode0_void_polygon(profile)
    if polygon is None:
        return

    polygon_r, polygon_z, _floor_z = polygon
    x_poly = np.concatenate([polygon_r, -polygon_r[::-1]])
    z_poly = np.concatenate([polygon_z, polygon_z[::-1]])
    y_poly = np.full_like(x_poly, y_plane)
    verts = np.column_stack((x_poly, y_poly, z_poly))
    poly = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=np.array([[0, i, i + 1] for i in range(1, len(verts) - 1)], dtype=int),
        color=color,
        alpha=alpha,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
    )
    try:
        poly.set_edgecolor((0, 0, 0, 0))
        poly.set_zsort("min")
    except Exception:
        pass
    if label:
        ax.plot([], [], [], color=color, alpha=alpha, label=label)


def _plot_mode0_void_on_pseudo_plane(ax, profile, color, alpha, label=None):
    polygon = _mode0_void_polygon(profile)
    if polygon is None:
        return None
    polygon_r, polygon_z, _floor_z = polygon
    x_poly = np.concatenate([polygon_r, -polygon_r[::-1]])
    z_poly = np.concatenate([polygon_z, polygon_z[::-1]])
    return _plot_pseudo_plane_polygon(ax, x_poly, z_poly, color=color, alpha=alpha, label=label)


def draw_mode_workspace(ax, profile, mode, style="real_3d"):
    display_color = _display_color_for_mode(mode)
    display_label = _display_label_for_mode(mode)
    if mode == 0 and "mode0_void_r" in profile:
        outer_z_plot, outer_r_plot, _, _ = _build_plot_curves(profile, mode)
        void_r = np.asarray(profile["mode0_void_r"], dtype=float)
        void_upper_z = np.asarray(profile["mode0_void_upper_z"], dtype=float)
        void_floor_z = float(profile["mode0_void_floor_z"])
        void_outline_r = np.asarray(profile.get("mode0_void_outline_r", void_r), dtype=float)
        void_outline_z = np.asarray(profile.get("mode0_void_outline_z", void_upper_z), dtype=float)
        void_outline_draw_r = _inset_unreachable_curve_r(void_outline_r, reference_r=outer_r_plot)
        void_lower_z = np.full_like(void_r, void_floor_z)

        if style == "pseudo_3d":
            poly = _plot_profile_on_pseudo_plane(
                ax,
                outer_z_plot,
                outer_r_plot,
                color=display_color,
                alpha=1.0,
                label=display_label,
            )
            if poly is not None:
                try:
                    poly.set_zorder(1)
                except Exception:
                    pass
            if SHOW_UNREACHABLE and void_r.size >= 4:
                if "mode0_void_outline_r" in profile:
                    x_poly = np.concatenate([void_outline_draw_r, -void_outline_draw_r[::-1]])
                    z_poly = np.concatenate([void_outline_z, void_outline_z[::-1]])
                    poly = _plot_pseudo_plane_polygon(
                        ax,
                        x_poly,
                        z_poly,
                        color=UNREACHABLE_COLOR,
                        alpha=1.0,
                        label=None,
                    )
                    if poly is not None:
                        try:
                            poly.set_zorder(2)
                        except Exception:
                            pass
                else:
                    poly = _plot_symmetric_band_on_pseudo_plane(
                        ax,
                        void_outline_draw_r,
                        void_lower_z,
                        void_upper_z,
                        color=UNREACHABLE_COLOR,
                        alpha=1.0,
                        label=None,
                    )
                    if poly is not None:
                        try:
                            poly.set_zorder(2)
                        except Exception:
                            pass
            return

        plot_revolved_profile(
            ax,
            outer_z_plot,
            outer_r_plot,
            color=display_color,
            alpha=REACH_ALPHA,
            label=display_label,
            cap_ends=False,
        )
        if SHOW_UNREACHABLE and void_r.size >= 4:
            plot_revolved_profile(
                ax,
                void_outline_z,
                void_outline_draw_r,
                color=UNREACHABLE_COLOR,
                alpha=UNREACHABLE_ALPHA,
                label=None,
                cap_ends=False,
            )
        return

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        void_r, void_upper_z, void_floor_z = _build_mode1_void_curve(profile)
        void_tail_r, void_tail_z = _build_mode1_void_tail(profile)
        void_draw_r = _inset_unreachable_curve_r(void_r, reference_r=upper_r) if void_r is not None else None
        void_tail_draw_r = _inset_unreachable_curve_r(void_tail_r, reference_r=upper_r) if void_tail_r is not None else None
        lower_draw_r, lower_draw_z = _mode1_outer_lower_draw_curve(profile)

        if style == "pseudo_3d":
            x_poly = np.concatenate([upper_r, lower_r[::-1], -lower_r, -upper_r[::-1]])
            z_poly = np.concatenate([upper_z, lower_z[::-1], lower_z, upper_z[::-1]])
            poly = _plot_pseudo_plane_polygon(
                ax,
                x_poly,
                z_poly,
                color=display_color,
                alpha=1.0,
                label=display_label,
            )
            if poly is not None:
                try:
                    poly.set_zorder(1)
                except Exception:
                    pass
            if SHOW_UNREACHABLE and void_r is not None:
                void_polygon = _mode1_void_polygon(profile)
                poly = None
                if void_polygon is not None:
                    polygon_r, polygon_z, _ = void_polygon
                    polygon_draw_r = _inset_unreachable_curve_r(polygon_r, reference_r=upper_r)
                    x_poly = np.concatenate([polygon_draw_r, -polygon_draw_r[::-1]])
                    z_poly = np.concatenate([polygon_z, polygon_z[::-1]])
                    poly = _plot_pseudo_plane_polygon(
                        ax,
                        x_poly,
                        z_poly,
                        color=UNREACHABLE_COLOR,
                        alpha=1.0,
                        label=None,
                    )
                else:
                    poly = _plot_symmetric_band_on_pseudo_plane(
                        ax,
                        void_draw_r,
                        np.full_like(void_r, void_floor_z),
                        void_upper_z,
                        color=UNREACHABLE_COLOR,
                        alpha=1.0,
                        label=None,
                    )
                if poly is not None:
                    try:
                        poly.set_zorder(2)
                    except Exception:
                        pass
            return

        plot_revolved_profile(
            ax,
            upper_z,
            upper_r,
            color=display_color,
            alpha=REACH_ALPHA,
            label=display_label,
            cap_ends=False,
        )
        plot_revolved_profile(
            ax,
            lower_draw_z,
            lower_draw_r,
            color=display_color,
            alpha=REACH_ALPHA,
            label=None,
            cap_ends=False,
        )
        if SHOW_UNREACHABLE and void_r is not None:
            void_polygon = _mode1_void_polygon(profile)
            if void_polygon is not None:
                polygon_r, polygon_z, _ = void_polygon
                polygon_draw_r = _inset_unreachable_curve_r(polygon_r, reference_r=upper_r)
                plot_revolved_profile(
                    ax,
                    polygon_z,
                    polygon_draw_r,
                    color=UNREACHABLE_COLOR,
                    alpha=UNREACHABLE_ALPHA,
                    label=None,
                    cap_ends=False,
                )
            else:
                plot_revolved_profile(
                    ax,
                    void_upper_z,
                    void_draw_r,
                    color=UNREACHABLE_COLOR,
                    alpha=UNREACHABLE_ALPHA,
                    label=None,
                    cap_ends=False,
                )
        return

    outer_z_plot, outer_r_plot, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)
    inner_draw_r = None
    if inner_z_plot is not None and inner_z_plot.size >= 4:
        inner_draw_r = _inset_unreachable_curve_r(inner_r_plot, reference_r=outer_r_plot)

    if style == "pseudo_3d":
        poly = _plot_profile_on_pseudo_plane(
            ax,
            outer_z_plot,
            outer_r_plot,
            color=display_color,
            alpha=1.0,
            label=display_label,
        )
        if poly is not None:
            try:
                poly.set_zorder(1)
            except Exception:
                pass
        if SHOW_UNREACHABLE:
            if inner_z_plot is not None and inner_z_plot.size >= 4:
                poly = _plot_profile_on_pseudo_plane(
                    ax,
                    inner_z_plot,
                    inner_draw_r,
                    color=UNREACHABLE_COLOR,
                    alpha=1.0,
                    label=None,
                )
                if poly is not None:
                    try:
                        poly.set_zorder(2)
                    except Exception:
                        pass
        return

    plot_revolved_profile(
        ax,
        outer_z_plot,
        outer_r_plot,
        color=display_color,
        alpha=REACH_ALPHA,
        label=display_label,
        cap_ends=False,
    )

    if SHOW_UNREACHABLE:
        if inner_z_plot is not None and inner_z_plot.size >= 4:
            plot_revolved_profile(
                ax,
                inner_z_plot,
                inner_draw_r,
                color=UNREACHABLE_COLOR,
                alpha=UNREACHABLE_ALPHA,
                label=None,
                cap_ends=False,
            )


def _pseudo_projection_setup(all_profiles, manipulator_states_by_mode=None):
    outer_max = 0.0
    z_min = np.inf
    z_max = -np.inf
    for profile in all_profiles:
        if len(profile["z"]) == 0:
            continue
        if "mode1_r" in profile:
            upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
            outer_max = max(outer_max, float(np.max([np.max(upper_r), np.max(lower_r)])))
            z_min = min(z_min, float(np.min([np.min(upper_z), np.min(lower_z)])))
            z_max = max(z_max, float(np.max([np.max(upper_z), np.max(lower_z)])))
        else:
            outer_max = max(outer_max, float(np.max(profile["outer_r"])))
            z_min = min(z_min, float(np.min(profile["z"])))
            z_max = max(z_max, float(np.max(profile["z"])))

    manip_bounds = _collect_side_view_manipulator_bounds(manipulator_states_by_mode)
    if manip_bounds is not None:
        y_min_manip, y_max_manip, z_min_manip, z_max_manip = manip_bounds
        outer_max = max(outer_max, abs(y_min_manip), abs(y_max_manip))
        z_min = min(z_min, z_min_manip)
        z_max = max(z_max, z_max_manip)

    if outer_max <= 0.0 or not np.isfinite(z_min) or not np.isfinite(z_max):
        return None

    x_center = 0.0
    y_center = 0.0
    z_center = 0.5 * (z_min + z_max)
    half_range = 1.08 * max(outer_max, 0.5 * (z_max - z_min))
    return {
        "xlim": (x_center - half_range, x_center + half_range),
        "ylim": (y_center - half_range, y_center + half_range),
        "zlim": (z_center - half_range, z_center + half_range),
    }


def _pseudo_projection_matrix(all_profiles, manipulator_states_by_mode=None):
    setup = _pseudo_projection_setup(all_profiles, manipulator_states_by_mode)
    if setup is None:
        return None

    fig = plt.figure(figsize=(2, 2))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.set_xlim(*setup["xlim"])
    ax3d.set_ylim(*setup["ylim"])
    ax3d.set_zlim(*setup["zlim"])
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.view_init(elev=PSEUDO_VIEW_ELEV, azim=PSEUDO_VIEW_AZIM)
    try:
        ax3d.set_proj_type("ortho")
    except Exception:
        pass
    proj = ax3d.get_proj()
    plt.close(fig)
    return proj


def _project_points_to_pseudo_view(points, proj_matrix):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    if proj_matrix is None:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    x_proj, y_proj, z_depth = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], proj_matrix)
    return np.column_stack((x_proj, y_proj)), np.asarray(z_depth, dtype=float)


def _revolve_section_points(r_vals, z_vals, theta):
    r_vals = np.asarray(r_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)
    if r_vals.size < 2 or z_vals.size != r_vals.size:
        return np.empty((0, 3), dtype=float)

    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    pos = np.column_stack((r_vals * cos_t, r_vals * sin_t, z_vals))
    neg = np.column_stack((-r_vals[::-1] * cos_t, -r_vals[::-1] * sin_t, z_vals[::-1]))
    return np.vstack((pos, neg))


def _mode_outer_section_points(profile, mode, theta):
    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        pos_upper = np.column_stack((upper_r * cos_t, upper_r * sin_t, upper_z))
        pos_lower = np.column_stack((lower_r[::-1] * cos_t, lower_r[::-1] * sin_t, lower_z[::-1]))
        neg_lower = np.column_stack((-lower_r * cos_t, -lower_r * sin_t, lower_z))
        neg_upper = np.column_stack((-upper_r[::-1] * cos_t, -upper_r[::-1] * sin_t, upper_z[::-1]))
        return np.vstack((pos_upper, pos_lower, neg_lower, neg_upper))

    outer_z, outer_r, _, _ = _build_plot_curves(profile, mode)
    return _revolve_section_points(outer_r, outer_z, theta)


def _mode_unreachable_section_points(profile, mode, theta):
    if mode == 0 and "mode0_void_r" in profile:
        polygon = _mode0_void_polygon(profile)
        if polygon is None:
            return np.empty((0, 3), dtype=float)
        polygon_r, polygon_z, _floor_z = polygon
        return _revolve_section_points(polygon_r, polygon_z, theta)

    if mode == 1 and "mode1_r" in profile:
        polygon = _mode1_void_polygon(profile)
        if polygon is None:
            return np.empty((0, 3), dtype=float)
        polygon_r, polygon_z, _floor_z = polygon
        return _revolve_section_points(polygon_r, polygon_z, theta)

    _outer_z, _outer_r, inner_z, inner_r = _build_plot_curves(profile, mode)
    if inner_z is None or inner_z.size < 4:
        return np.empty((0, 3), dtype=float)
    return _revolve_section_points(inner_r, inner_z, theta)


def _render_projected_polygons(ax, polygons_3d, proj_matrix, color, alpha=1.0, zorder=1):
    projected_items = []
    for poly3d in polygons_3d:
        poly3d = np.asarray(poly3d, dtype=float)
        if poly3d.ndim != 2 or poly3d.shape[0] < 3 or poly3d.shape[1] != 3:
            continue
        projected, depth = _project_points_to_pseudo_view(poly3d, proj_matrix)
        if projected.shape[0] < 3:
            continue
        projected_items.append((float(np.mean(depth)), projected))

    for _depth, projected in sorted(projected_items, key=lambda item: item[0]):
        ax.fill(projected[:, 0], projected[:, 1], color=color, alpha=alpha, linewidth=0, zorder=zorder)


def _surface_quads_from_curve(z_vals, r_vals, theta_samples=PSEUDO_MERIDIAN_SAMPLES):
    z_vals, r_vals = _decimate_curve(np.asarray(z_vals, dtype=float), np.asarray(r_vals, dtype=float), max_axial_samples=40)
    if z_vals.size < 2 or r_vals.size != z_vals.size:
        return []

    theta_vals = np.linspace(0.0, 2.0 * np.pi, int(max(theta_samples, 12)), endpoint=False)
    quads = []
    for theta_idx, theta0 in enumerate(theta_vals):
        theta1 = theta_vals[(theta_idx + 1) % theta_vals.size]
        c0, s0 = float(np.cos(theta0)), float(np.sin(theta0))
        c1, s1 = float(np.cos(theta1)), float(np.sin(theta1))
        for i in range(z_vals.size - 1):
            r0 = float(r_vals[i])
            r1 = float(r_vals[i + 1])
            z0 = float(z_vals[i])
            z1 = float(z_vals[i + 1])
            quad = np.array(
                [
                    [r0 * c0, r0 * s0, z0],
                    [r1 * c0, r1 * s0, z1],
                    [r1 * c1, r1 * s1, z1],
                    [r0 * c1, r0 * s1, z0],
                ],
                dtype=float,
            )
            quads.append(quad)
    return quads


def _render_projected_surface_quads(ax, quads, proj_matrix, color, alpha=1.0, zorder=1):
    projected_items = []
    for quad in quads:
        projected, depth = _project_points_to_pseudo_view(quad, proj_matrix)
        if projected.shape[0] < 4:
            continue
        projected_items.append((float(np.mean(depth)), projected))

    for _depth, projected in sorted(projected_items, key=lambda item: item[0]):
        ax.fill(projected[:, 0], projected[:, 1], color=color, alpha=alpha, linewidth=0, zorder=zorder)


def draw_mode_main_pseudo(ax, profile, mode, proj_matrix):
    display_color = _display_color_for_mode(mode)
    outer_quads = []
    unreachable_quads = []

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        outer_quads.extend(_surface_quads_from_curve(upper_z, upper_r))
        outer_quads.extend(_surface_quads_from_curve(lower_z, lower_r))
        if SHOW_UNREACHABLE:
            void_r, void_upper_z, _void_floor_z = _build_mode1_void_curve(profile)
            if void_r is not None and void_r.size >= 4:
                unreachable_quads.extend(_surface_quads_from_curve(void_upper_z, void_r))
            void_tail_r, void_tail_z = _build_mode1_void_tail(profile)
            if void_tail_r is not None and void_tail_z is not None and void_tail_r.size >= 2:
                connector_r = np.array([void_tail_r[0], void_tail_r[0]], dtype=float)
                connector_z = np.array([void_upper_z[-1], void_tail_z[0]], dtype=float)
                unreachable_quads.extend(_surface_quads_from_curve(connector_z, connector_r))
                unreachable_quads.extend(_surface_quads_from_curve(void_tail_z, void_tail_r))
    else:
        outer_z, outer_r, inner_z, inner_r = _build_plot_curves(profile, mode)
        outer_quads.extend(_surface_quads_from_curve(outer_z, outer_r))
        if SHOW_UNREACHABLE:
            if mode == 0 and "mode0_void_outline_r" in profile:
                void_outline_r = np.asarray(profile["mode0_void_outline_r"], dtype=float)
                void_outline_z = np.asarray(profile["mode0_void_outline_z"], dtype=float)
                void_outline_draw_r = _inset_unreachable_curve_r(void_outline_r, reference_r=outer_r)
                unreachable_quads.extend(_surface_quads_from_curve(void_outline_z, void_outline_draw_r))
            elif inner_z is not None and inner_z.size >= 4:
                inner_draw_r = _inset_unreachable_curve_r(inner_r, reference_r=outer_r)
                unreachable_quads.extend(_surface_quads_from_curve(inner_z, inner_draw_r))

    _render_projected_surface_quads(ax, outer_quads, proj_matrix, color=display_color, alpha=1.0, zorder=1)
    if unreachable_quads:
        _render_projected_surface_quads(ax, unreachable_quads, proj_matrix, color=UNREACHABLE_COLOR, alpha=1.0, zorder=2)


def draw_mode_side_view(ax, profile, mode):
    display_color = _display_color_for_mode(mode)
    if mode == 0 and "mode0_void_r" in profile:
        z, outer_r, _, _ = _build_plot_curves(profile, mode)
        void_r = np.asarray(profile["mode0_void_r"], dtype=float)
        void_upper_z = np.asarray(profile["mode0_void_upper_z"], dtype=float)
        void_draw_r = _inset_unreachable_curve_r(void_r, reference_r=outer_r)
        void_polygon = _mode0_void_polygon(profile)
        void_floor_z = float(profile["mode0_void_floor_z"])

        ax.fill_betweenx(z, -outer_r, outer_r, color=display_color, alpha=0.65, linewidth=0)
        if SHOW_UNREACHABLE and void_r.size >= 4:
            if void_polygon is not None:
                polygon_r, polygon_z, _ = void_polygon
                polygon_draw_r = _inset_unreachable_curve_r(polygon_r, reference_r=outer_r)
                ax.fill(polygon_draw_r, polygon_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
                ax.fill(-polygon_draw_r, polygon_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
            else:
                ax.fill_between(void_draw_r, void_floor_z, void_upper_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
                ax.fill_between(-void_draw_r, void_floor_z, void_upper_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)

        ax.plot(outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(-outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
        if SHOW_UNREACHABLE and void_r.size >= 4:
            if "mode0_void_outline_r" in profile:
                void_outline_r = np.asarray(profile["mode0_void_outline_r"], dtype=float)
                void_outline_z = np.asarray(profile["mode0_void_outline_z"], dtype=float)
                void_outline_draw_r = _inset_unreachable_curve_r(void_outline_r, reference_r=outer_r)
                ax.plot(void_outline_draw_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
                ax.plot(-void_outline_draw_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
            else:
                ax.plot(void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
                ax.plot(-void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        if DISPLAY_IN_MM:
            mm_formatter = FuncFormatter(lambda value, _pos: f"{value * 1000:.0f}")
            ax.xaxis.set_major_formatter(mm_formatter)
            ax.yaxis.set_major_formatter(mm_formatter)
            ax.set_xlabel("R (mm)")
            ax.set_ylabel("Z (mm)")
        else:
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
        return

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        void_r, void_upper_z, void_floor_z = _build_mode1_void_curve(profile)
        void_tail_r, void_tail_z = _build_mode1_void_tail(profile)
        void_draw_r = _inset_unreachable_curve_r(void_r, reference_r=upper_r) if void_r is not None else None
        void_tail_draw_r = _inset_unreachable_curve_r(void_tail_r, reference_r=upper_r) if void_tail_r is not None else None
        lower_draw_r, lower_draw_z = _mode1_outer_lower_draw_curve(profile)
        pos_x = np.concatenate([upper_r, lower_r[::-1]])
        pos_y = np.concatenate([upper_z, lower_z[::-1]])
        neg_x = -pos_x

        ax.fill(pos_x, pos_y, color=display_color, alpha=0.65, linewidth=0)
        ax.fill(neg_x, pos_y, color=display_color, alpha=0.65, linewidth=0)
        if SHOW_UNREACHABLE and void_r is not None:
            void_polygon = _mode1_void_polygon(profile)
            if void_polygon is not None:
                polygon_r, polygon_z, _ = void_polygon
                polygon_draw_r = _inset_unreachable_curve_r(polygon_r, reference_r=upper_r)
                ax.fill(polygon_draw_r, polygon_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
                ax.fill(-polygon_draw_r, polygon_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
            else:
                pos_void_x = np.concatenate([void_draw_r, void_draw_r[::-1]])
                pos_void_y = np.concatenate([void_upper_z, np.full_like(void_r, void_floor_z)[::-1]])
                neg_void_x = -pos_void_x
                ax.fill(pos_void_x, pos_void_y, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
                ax.fill(neg_void_x, pos_void_y, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
        ax.plot(upper_r, upper_z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(-upper_r, upper_z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(lower_draw_r, lower_draw_z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(-lower_draw_r, lower_draw_z, color=display_color, linewidth=1.0, alpha=0.9)
        if SHOW_UNREACHABLE and void_r is not None:
            ax.plot(void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
            ax.plot(-void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
            if void_tail_r is not None and void_tail_z is not None and void_tail_r.size >= 2:
                ax.plot(
                    [float(void_tail_draw_r[0]), float(void_tail_draw_r[0])],
                    [float(void_upper_z[-1]), float(void_tail_z[0])],
                    color=UNREACHABLE_COLOR,
                    linewidth=1.0,
                    alpha=0.95,
                )
                ax.plot(
                    [-float(void_tail_draw_r[0]), -float(void_tail_draw_r[0])],
                    [float(void_upper_z[-1]), float(void_tail_z[0])],
                    color=UNREACHABLE_COLOR,
                    linewidth=1.0,
                    alpha=0.95,
                )
                ax.plot(void_tail_draw_r, void_tail_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
                ax.plot(-void_tail_draw_r, void_tail_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
                ax.plot(
                    [0.0, float(void_tail_draw_r[-1])],
                    [void_floor_z, void_floor_z],
                    color=UNREACHABLE_COLOR,
                    linewidth=1.0,
                    alpha=0.65,
                )
                ax.plot(
                    [0.0, -float(void_tail_draw_r[-1])],
                    [void_floor_z, void_floor_z],
                    color=UNREACHABLE_COLOR,
                    linewidth=1.0,
                    alpha=0.65,
                )
            else:
                ax.plot(void_draw_r, np.full_like(void_r, void_floor_z), color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.65)
                ax.plot(-void_draw_r, np.full_like(void_r, void_floor_z), color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.65)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        if DISPLAY_IN_MM:
            mm_formatter = FuncFormatter(lambda value, _pos: f"{value * 1000:.0f}")
            ax.xaxis.set_major_formatter(mm_formatter)
            ax.yaxis.set_major_formatter(mm_formatter)
            ax.set_xlabel("Y (mm)")
            ax.set_ylabel("Z (mm)")
        else:
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
        return

    z, outer_r, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    ax.fill_betweenx(z, -outer_r, outer_r, color=display_color, alpha=0.65, linewidth=0)

    if SHOW_UNREACHABLE and inner_z_plot is not None and inner_z_plot.size >= 4:
        inner_draw_r = _inset_unreachable_curve_r(inner_r_plot, reference_r=outer_r)
        ax.fill_betweenx(
            inner_z_plot,
            -inner_draw_r,
            inner_draw_r,
            color=UNREACHABLE_COLOR,
            alpha=0.80,
            linewidth=0,
        )

    ax.plot(outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
    ax.plot(-outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
    if inner_z_plot is not None and inner_z_plot.size >= 4:
        ax.plot(inner_draw_r, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)
        ax.plot(-inner_draw_r, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_LINEWIDTH, alpha=0.95)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    if DISPLAY_IN_MM:
        mm_formatter = FuncFormatter(lambda value, _pos: f"{value * 1000:.0f}")
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        ax.set_xlabel("R (mm)")
        ax.set_ylabel("Z (mm)")
    else:
        ax.set_xlabel("R")
        ax.set_ylabel("Z")


def draw_mode_side_scatter(ax, side_points, mode):
    if side_points is None or side_points.size == 0:
        return

    r = np.asarray(side_points[:, 0], dtype=float)
    z = np.asarray(side_points[:, 1], dtype=float)
    ax.scatter(r, z, s=2.5, color=_display_color_for_mode(mode), alpha=0.14, edgecolors="none")
    ax.scatter(-r, z, s=2.5, color=_display_color_for_mode(mode), alpha=0.14, edgecolors="none")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    if DISPLAY_IN_MM:
        mm_formatter = FuncFormatter(lambda value, _pos: f"{value * 1000:.0f}")
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        ax.set_xlabel("R (mm)")
        ax.set_ylabel("Z (mm)")
    else:
        ax.set_xlabel("R")
        ax.set_ylabel("Z")


def _save_raw_scatter_debug_figure(sampled_points, profiles, manipulator_states):
    if not SHOW_SAMPLE_SCATTER:
        return

    scatter_modes = []
    for mode in PLOT_MODES:
        if mode == 0:
            scatter_modes.extend([source_mode for source_mode in COMBINED_SOURCE_MODES if source_mode in sampled_points])
        elif mode in sampled_points:
            scatter_modes.append(mode)
    scatter_modes = list(dict.fromkeys(scatter_modes))
    if not scatter_modes:
        return

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for scatter_mode in scatter_modes:
        draw_mode_side_scatter(ax, sampled_points.get(scatter_mode), scatter_mode)

    all_profiles = list(profiles.values()) if profiles else []
    scatter_input_points = {mode: sampled_points[mode] for mode in scatter_modes if mode in sampled_points}
    configure_side_view_axes(ax, all_profiles, manipulator_states, scatter_input_points)
    ax.set_title("Raw Sample Scatter")
    RAW_SCATTER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RAW_SCATTER_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved raw scatter debug figure to: {RAW_SCATTER_OUTPUT_PATH.resolve()}")


def overlay_profile_curves(ax, profile, mode):
    display_color = _display_color_for_mode(mode)
    if mode == 0 and "mode0_void_r" in profile:
        z, outer_r, _, _ = _build_plot_curves(profile, mode)
        void_r = np.asarray(profile["mode0_void_r"], dtype=float)
        void_upper_z = np.asarray(profile["mode0_void_upper_z"], dtype=float)
        void_draw_r = _inset_unreachable_curve_r(void_r, reference_r=outer_r)
        ax.plot(outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(-outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)
        if void_r.size >= 4:
            if "mode0_void_outline_r" in profile:
                void_outline_r = np.asarray(profile["mode0_void_outline_r"], dtype=float)
                void_outline_z = np.asarray(profile["mode0_void_outline_z"], dtype=float)
                void_outline_draw_r = _inset_unreachable_curve_r(void_outline_r, reference_r=outer_r)
                ax.plot(void_outline_draw_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
                ax.plot(-void_outline_draw_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
            else:
                ax.plot(void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
                ax.plot(-void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
        return

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        void_r, void_upper_z, _void_floor_z = _build_mode1_void_curve(profile)
        void_tail_r, void_tail_z = _build_mode1_void_tail(profile)
        void_draw_r = _inset_unreachable_curve_r(void_r, reference_r=upper_r) if void_r is not None else None
        void_tail_draw_r = _inset_unreachable_curve_r(void_tail_r, reference_r=upper_r) if void_tail_r is not None else None
        lower_draw_r, lower_draw_z = _mode1_outer_lower_draw_curve(profile)
        ax.plot(upper_r, upper_z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(-upper_r, upper_z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(lower_draw_r, lower_draw_z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(-lower_draw_r, lower_draw_z, color=display_color, linewidth=2.0, alpha=0.95)
        if SHOW_UNREACHABLE and void_r is not None:
            ax.plot(void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
            ax.plot(-void_draw_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
            if void_tail_r is not None and void_tail_z is not None and void_tail_r.size >= 2:
                ax.plot(
                    [float(void_tail_draw_r[0]), float(void_tail_draw_r[0])],
                    [float(void_upper_z[-1]), float(void_tail_z[0])],
                    color=UNREACHABLE_COLOR,
                    linewidth=2.0,
                    alpha=0.95,
                )
                ax.plot(
                    [-float(void_tail_draw_r[0]), -float(void_tail_draw_r[0])],
                    [float(void_upper_z[-1]), float(void_tail_z[0])],
                    color=UNREACHABLE_COLOR,
                    linewidth=2.0,
                    alpha=0.95,
                )
                ax.plot(void_tail_draw_r, void_tail_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
                ax.plot(-void_tail_draw_r, void_tail_z, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
        return

    z, outer_r, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    ax.plot(outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)
    ax.plot(-outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)

    if inner_z_plot is not None and inner_z_plot.size >= 4:
        inner_draw_r = _inset_unreachable_curve_r(inner_r_plot, reference_r=outer_r)
        ax.plot(inner_draw_r, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)
        ax.plot(-inner_draw_r, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=UNREACHABLE_SCATTER_LINEWIDTH, alpha=0.95)


def _collect_pseudo_main_bounds(all_profiles, manipulator_states_by_mode=None, proj_matrix=None):
    if proj_matrix is None:
        return None
    all_points = []
    theta_vals = np.linspace(0.0, 2.0 * np.pi, max(24, PSEUDO_MERIDIAN_SAMPLES // 2), endpoint=False)
    for mode, profile in zip(PLOT_MODES, all_profiles):
        for theta in theta_vals:
            outer_poly = _mode_outer_section_points(profile, mode, theta)
            if outer_poly.shape[0] >= 3:
                all_points.append(outer_poly)
            if SHOW_UNREACHABLE:
                unreachable_poly = _mode_unreachable_section_points(profile, mode, theta)
                if unreachable_poly.shape[0] >= 3:
                    all_points.append(unreachable_poly)

    if SHOW_MANIPULATOR and manipulator_states_by_mode:
        overlay_csm = CSM.from_config(CONFIG_PATH)
        for state_list in manipulator_states_by_mode.values():
            for state in state_list:
                _apply_mode_state(overlay_csm, state)
                geometry = overlay_csm.get_visualization_segments()
                for segment in geometry.get("segments", []):
                    points = np.asarray(segment.get("points"), dtype=float)
                    if points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0:
                        all_points.append(points)
                tool = geometry.get("tool", {})
                for key in ("start", "end"):
                    point = np.asarray(tool.get(key), dtype=float) if key in tool else None
                    if point is not None and point.shape == (3,):
                        all_points.append(point[np.newaxis, :])

    if not all_points:
        return None

    projected_sets = []
    for points in all_points:
        projected, _depth = _project_points_to_pseudo_view(points, proj_matrix)
        if projected.shape[0] > 0:
            projected_sets.append(projected)
    if not projected_sets:
        return None

    stacked = np.vstack(projected_sets)
    x_min = float(np.min(stacked[:, 0]))
    x_max = float(np.max(stacked[:, 0]))
    y_min = float(np.min(stacked[:, 1]))
    y_max = float(np.max(stacked[:, 1]))
    return x_min, x_max, y_min, y_max


def configure_pseudo_main_axes(ax, all_profiles, manipulator_states_by_mode=None, proj_matrix=None):
    bounds = _collect_pseudo_main_bounds(all_profiles, manipulator_states_by_mode, proj_matrix=proj_matrix)
    if bounds is None:
        return

    x_min, x_max, y_min, y_max = bounds
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    pad_x = 0.08 * x_span
    pad_y = 0.08 * y_span
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)

    if DISPLAY_IN_MM:
        mm_formatter = FuncFormatter(lambda value, _pos: f"{value * 1000:.0f}")
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        ax.set_xlabel("View X (mm)")
        ax.set_ylabel("View Z (mm)")
    else:
        ax.set_xlabel("View X")
        ax.set_ylabel("View Z")


def configure_axes(ax, all_profiles, style="real_3d", role="main"):
    outer_max = 0.0
    z_min = np.inf
    z_max = -np.inf
    for profile in all_profiles:
        if len(profile["z"]) == 0:
            continue
        if "mode1_r" in profile:
            upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
            outer_max = max(outer_max, float(np.max([np.max(upper_r), np.max(lower_r)])))
            z_min = min(z_min, float(np.min([np.min(upper_z), np.min(lower_z)])))
            z_max = max(z_max, float(np.max([np.max(upper_z), np.max(lower_z)])))
        else:
            outer_max = max(outer_max, float(np.max(profile["outer_r"])))
            z_min = min(z_min, float(np.min(profile["z"])))
            z_max = max(z_max, float(np.max(profile["z"])))

    if outer_max <= 0.0 or not np.isfinite(z_min) or not np.isfinite(z_max):
        return

    x_center = 0.0
    y_center = 0.0
    z_center = 0.5 * (z_min + z_max)
    half_range = 1.08 * max(outer_max, 0.5 * (z_max - z_min))

    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(z_center - half_range, z_center + half_range)
    ax.set_box_aspect([1, 1, 1])
    if role == "side" and style == "real_3d":
        ax.view_init(elev=SIDE_REAL_VIEW_ELEV, azim=SIDE_REAL_VIEW_AZIM)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
    elif style == "pseudo_3d":
        ax.view_init(elev=PSEUDO_VIEW_ELEV, azim=PSEUDO_VIEW_AZIM)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
    else:
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    if DISPLAY_IN_MM:
        mm_formatter = FuncFormatter(lambda value, _pos: f"{value * 1000:.0f}")
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        ax.zaxis.set_major_formatter(mm_formatter)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

    if not SHOW_AXES:
        ax.set_axis_off()


def _collect_side_view_manipulator_bounds(manipulator_states_by_mode):
    if not SHOW_MANIPULATOR or not manipulator_states_by_mode:
        return None

    overlay_csm = CSM.from_config(CONFIG_PATH)
    y_min = np.inf
    y_max = -np.inf
    z_min = np.inf
    z_max = -np.inf

    for state_list in manipulator_states_by_mode.values():
        for state in state_list:
            _apply_mode_state(overlay_csm, state)
            geometry = overlay_csm.get_visualization_segments()
            geometry_points = []

            for segment in geometry.get("segments", []):
                points = np.asarray(segment.get("points"), dtype=float)
                if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
                    continue
                geometry_points.append(points)

            tool = geometry.get("tool", {})
            for key in ("start", "end"):
                point = np.asarray(tool.get(key), dtype=float) if key in tool else None
                if point is None or point.shape != (3,):
                    continue
                geometry_points.append(point[np.newaxis, :])

            if not geometry_points:
                continue

            stacked_points = np.vstack(geometry_points)
            side_points = _map_points_to_side_view(stacked_points)
            if side_points.shape[0] == 0:
                continue
            y_min = min(y_min, float(np.min(side_points[:, 0])))
            y_max = max(y_max, float(np.max(side_points[:, 0])))
            z_min = min(z_min, float(np.min(side_points[:, 1])))
            z_max = max(z_max, float(np.max(side_points[:, 1])))

    if not np.isfinite(y_min) or not np.isfinite(y_max) or not np.isfinite(z_min) or not np.isfinite(z_max):
        return None
    return y_min, y_max, z_min, z_max


def _collect_side_points_bounds(sampled_points_by_mode):
    if not sampled_points_by_mode:
        return None

    r_max = 0.0
    z_min = np.inf
    z_max = -np.inf
    for side_points in sampled_points_by_mode.values():
        if side_points is None:
            continue
        side_points = np.asarray(side_points, dtype=float)
        if side_points.ndim != 2 or side_points.shape[0] == 0 or side_points.shape[1] != 2:
            continue
        r_max = max(r_max, float(np.max(np.abs(side_points[:, 0]))))
        z_min = min(z_min, float(np.min(side_points[:, 1])))
        z_max = max(z_max, float(np.max(side_points[:, 1])))

    if r_max <= 0.0 or not np.isfinite(z_min) or not np.isfinite(z_max):
        return None
    return r_max, z_min, z_max


def configure_side_view_axes(ax, all_profiles, manipulator_states_by_mode=None, sampled_points_by_mode=None):
    outer_max = 0.0
    z_min = np.inf
    z_max = -np.inf
    for profile in all_profiles:
        if len(profile["z"]) == 0:
            continue
        if "mode1_r" in profile:
            upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
            outer_max = max(outer_max, float(np.max([np.max(upper_r), np.max(lower_r)])))
            z_min = min(z_min, float(np.min([np.min(upper_z), np.min(lower_z)])))
            z_max = max(z_max, float(np.max([np.max(upper_z), np.max(lower_z)])))
        else:
            outer_max = max(outer_max, float(np.max(profile["outer_r"])))
            z_min = min(z_min, float(np.min(profile["z"])))
            z_max = max(z_max, float(np.max(profile["z"])))

    manip_bounds = _collect_side_view_manipulator_bounds(manipulator_states_by_mode)
    if manip_bounds is not None:
        y_min_manip, y_max_manip, z_min_manip, z_max_manip = manip_bounds
        outer_max = max(outer_max, abs(y_min_manip), abs(y_max_manip))
        z_min = min(z_min, z_min_manip)
        z_max = max(z_max, z_max_manip)

    points_bounds = _collect_side_points_bounds(sampled_points_by_mode)
    if points_bounds is not None:
        points_r_max, z_min_pts, z_max_pts = points_bounds
        outer_max = max(outer_max, points_r_max)
        z_min = min(z_min, z_min_pts)
        z_max = max(z_max, z_max_pts)

    if outer_max <= 0.0 or not np.isfinite(z_min) or not np.isfinite(z_max):
        return

    pad_r = 0.08 * outer_max
    z_span = max(z_max - z_min, 1e-9)
    pad_z = 0.08 * z_span
    ax.set_xlim(-(outer_max + pad_r), outer_max + pad_r)
    ax.set_ylim(z_min - pad_z, z_max + pad_z)


def _load_or_sample_mode_points(csm, mode):
    side_points = _load_profile_cache(mode)
    if side_points is not None:
        print(f"Mode {mode}: loaded cached sampled points")
        return side_points

    side_points = sample_mode_side_points(csm, mode, MODE_SAMPLE_RES[mode])
    _save_profile_cache(mode, side_points)
    return side_points


def _sample_random_mode_state(csm, mode, rng):
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    L1 = 0.0
    L2 = float(csm.L_20)
    Lr = 0.0
    Ls = 0.0
    theta_1 = 0.0
    theta_2 = 0.0
    delta_1 = 0.0
    delta_2 = 0.0

    if mode == 1:
        L2 = float(rng.uniform(0.0, csm.L_20))
        theta_2 = float(rng.uniform(0.0, _segment_theta_upper_bound(csm, 2, L2))) if L2 > 0.0 else 0.0
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    elif mode == 2:
        Lr = float(rng.uniform(0.0, csm.L_r0))
        theta_2 = float(rng.uniform(0.0, _segment_theta_upper_bound(csm, 2, csm.L_20)))
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    elif mode == 3:
        L1 = float(rng.uniform(0.0, csm.L_10))
        Lr = float(csm.L_r0)
        theta_1 = float(rng.uniform(0.0, _segment_theta_upper_bound(csm, 1, L1))) if L1 > 0.0 else 0.0
        theta_2 = float(rng.uniform(0.0, _segment_theta_upper_bound(csm, 2, csm.L_20)))
        delta_1 = float(rng.uniform(0.0, 2.0 * np.pi))
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    elif mode == 4:
        L1 = float(csm.L_10)
        L2 = float(csm.L_20)
        Lr = float(csm.L_r0)
        Ls = float(rng.uniform(0.0, csm.L_s0))
        theta_1 = float(rng.uniform(0.0, _segment_theta_upper_bound(csm, 1, csm.L_10)))
        theta_2 = float(rng.uniform(0.0, _segment_theta_upper_bound(csm, 2, csm.L_20)))
        delta_1 = float(rng.uniform(0.0, 2.0 * np.pi))
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    state = {
        "mode": int(mode),
        "phi": phi,
        "L1": L1,
        "L2": L2,
        "Lr": Lr,
        "Ls": Ls,
        "theta_1": theta_1,
        "theta_2": theta_2,
        "delta_1": delta_1,
        "delta_2": delta_2,
    }
    csm.set_state(**state)
    return state


def _apply_mode_state(csm, state):
    csm.set_state(
        mode=int(state["mode"]),
        phi=float(state["phi"]),
        L1=float(state["L1"]),
        L2=float(state["L2"]),
        Lr=float(state["Lr"]),
        Ls=float(state["Ls"]),
        theta_1=float(state["theta_1"]),
        theta_2=float(state["theta_2"]),
        delta_1=float(state["delta_1"]),
        delta_2=float(state["delta_2"]),
    )


def _mode_has_distinct_configuration(csm, mode, tol=1e-9):
    if mode == 1:
        return float(csm.L_20) > tol
    if mode == 2:
        return float(csm.L_r0) > tol and float(csm.L_20) > tol
    if mode == 3:
        return float(csm.L_10) > tol and float(csm.L_r0) > tol and float(csm.L_20) > tol
    if mode == 4:
        return (
            float(csm.L_s0) > tol
            and float(csm.L_10) > tol
            and float(csm.L_r0) > tol
            and float(csm.L_20) > tol
        )
    return False


def _sample_manipulator_states():
    if not SHOW_MANIPULATOR:
        return {}

    rng = np.random.default_rng(MANIPULATOR_RNG_SEED)
    csm = CSM.from_config(CONFIG_PATH)
    sampled_states = {}
    for mode in PLOT_MODES:
        if mode == 0:
            state_list = []
            for source_mode in COMBINED_SOURCE_MODES:
                if not _mode_has_distinct_configuration(csm, source_mode):
                    continue
                state_list.append(_sample_random_mode_state(csm, source_mode, rng))
            sampled_states[0] = state_list
            continue
        sampled_states[mode] = [_sample_random_mode_state(csm, mode, rng)]
    return sampled_states


def _overlay_manipulators(ax, manipulator_states):
    if not SHOW_MANIPULATOR or ax is None:
        return

    overlay_csm = CSM.from_config(CONFIG_PATH)
    for state in manipulator_states:
        _apply_mode_state(overlay_csm, state)
        overlay_csm.target_pose = overlay_csm.pose.copy()
        overlay_csm.plot_manipulator(
            ax,
            render_mode=MANIPULATOR_RENDER_MODE,
            clear_ax=False,
            draw_target=False,
            configure_axes=False,
            title=None,
        )
        for artist in list(ax.lines) + list(ax.collections):
            try:
                artist.set_zorder(10)
            except Exception:
                pass


def _overlay_manipulators_main_pseudo(ax, manipulator_states, proj_matrix):
    if not SHOW_MANIPULATOR or ax is None:
        return

    overlay_csm = CSM.from_config(CONFIG_PATH)
    segment_colors = {
        "seg1": "#d97706",
        "seg2": "#1d4ed8",
        "rigid": "#4b5563",
        "base": "#4b5563",
    }

    for state in manipulator_states:
        _apply_mode_state(overlay_csm, state)
        geometry = overlay_csm.get_visualization_segments()

        merged_centerline = []
        for seg_idx, segment in enumerate(geometry.get("segments", [])):
            points = np.asarray(segment.get("points"), dtype=float)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
                continue
            if seg_idx > 0:
                points = points[1:]
            merged_centerline.append(points)

            projected, _depth = _project_points_to_pseudo_view(points, proj_matrix)
            if projected.shape[0] < 2:
                continue

            color = segment_colors.get(segment.get("label"), "#4b5563")
            ax.plot(
                projected[:, 0],
                projected[:, 1],
                color=color,
                linewidth=2.4,
                alpha=0.95,
                zorder=8,
            )

        if merged_centerline:
            merged = np.vstack(merged_centerline)
            projected, _depth = _project_points_to_pseudo_view(merged, proj_matrix)
            if projected.shape[0] >= 2:
                ax.plot(
                    projected[:, 0],
                    projected[:, 1],
                    color="#1f2937",
                    linewidth=1.1,
                    linestyle="--",
                    alpha=0.82,
                    zorder=9,
                )

        tool = geometry.get("tool", {})
        tool_start = np.asarray(tool.get("start"), dtype=float) if "start" in tool else None
        tool_end = np.asarray(tool.get("end"), dtype=float) if "end" in tool else None
        if tool_start is not None and tool_end is not None and tool_start.shape == (3,) and tool_end.shape == (3,):
            projected, _depth = _project_points_to_pseudo_view(np.vstack((tool_start, tool_end)), proj_matrix)
            if projected.shape[0] == 2:
                ax.plot(
                    projected[:, 0],
                    projected[:, 1],
                    color="#0f766e",
                    linewidth=2.0,
                    alpha=0.95,
                    zorder=10,
                )


def _overlay_manipulators_side_view(ax, manipulator_states):
    if not SHOW_MANIPULATOR or ax is None:
        return

    overlay_csm = CSM.from_config(CONFIG_PATH)
    segment_colors = {
        "seg1": "#d97706",
        "seg2": "#1d4ed8",
        "rigid": "#4b5563",
        "base": "#4b5563",
    }

    for state in manipulator_states:
        _apply_mode_state(overlay_csm, state)
        geometry = overlay_csm.get_visualization_segments()
        geometry_points = []
        for segment in geometry.get("segments", []):
            points = np.asarray(segment.get("points"), dtype=float)
            if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == 3:
                geometry_points.append(points)
        tool = geometry.get("tool", {})
        for key in ("start", "end"):
            point = np.asarray(tool.get(key), dtype=float) if key in tool else None
            if point is not None and point.shape == (3,):
                geometry_points.append(point[np.newaxis, :])

        reference_angle = _resolve_side_view_reference_angle(np.vstack(geometry_points)) if geometry_points else 0.0

        merged_centerline = []
        for seg_idx, segment in enumerate(geometry["segments"]):
            points = np.asarray(segment["points"], dtype=float)
            if points.size == 0:
                continue
            if seg_idx > 0:
                points = points[1:]
            merged_centerline.append(points)
            side_points = _map_points_to_side_view(points, reference_angle=reference_angle)
            if side_points.shape[0] < 2:
                continue

            color = segment_colors.get(segment["label"], "#4b5563")
            ax.plot(
                side_points[:, 0],
                side_points[:, 1],
                color=color,
                linewidth=2.4,
                alpha=0.92,
                zorder=8,
            )

        if merged_centerline:
            merged = np.vstack(merged_centerline)
            side_points = _map_points_to_side_view(merged, reference_angle=reference_angle)
            ax.plot(
                side_points[:, 0],
                side_points[:, 1],
                color="#1f2937",
                linewidth=1.1,
                linestyle="--",
                alpha=0.78,
                zorder=9,
            )

        tool = geometry.get("tool", {})
        tool_start = np.asarray(tool.get("start"), dtype=float) if "start" in tool else None
        tool_end = np.asarray(tool.get("end"), dtype=float) if "end" in tool else None
        if tool_start is not None and tool_end is not None and tool_start.size == 3 and tool_end.size == 3:
            side_points = _map_points_to_side_view(
                np.vstack((tool_start, tool_end)),
                reference_angle=reference_angle,
            )
            ax.plot(
                side_points[:, 0],
                side_points[:, 1],
                color="#0f766e",
                linewidth=2.0,
                alpha=0.92,
                zorder=10,
            )


def generate_profiles():
    csm = CSM.from_config(CONFIG_PATH)
    profiles = {}
    sampled_points = {}
    for mode in PLOT_MODES:
        if mode == 0:
            combined_points = []
            source_profiles = {}
            for source_mode in COMBINED_SOURCE_MODES:
                side_points = _load_or_sample_mode_points(csm, source_mode)
                sampled_points[source_mode] = side_points
                if side_points is not None and side_points.size > 0:
                    combined_points.append(np.asarray(side_points, dtype=float))
                    source_profile = build_profile_from_side_points(side_points, mode=source_mode, csm=csm)
                    if source_profile is not None:
                        source_profiles[source_mode] = source_profile

            if not combined_points:
                print("Mode 0: failed to gather source side points")
                continue

            merged_points = np.vstack(combined_points)
            merged_points = np.unique(np.round(merged_points, decimals=9), axis=0)
            sampled_points[0] = merged_points
            profile = _build_mode0_profile_from_sources(merged_points, source_profiles)
            if profile is None:
                print("Mode 0: failed to build combined profile")
                continue
            profiles[0] = profile
            continue

        side_points = _load_or_sample_mode_points(csm, mode)
        sampled_points[mode] = side_points
        profile = build_profile_from_side_points(side_points, mode=mode, csm=csm)
        if profile is None:
            print(f"Mode {mode}: failed to build profile")
            continue
        profiles[mode] = profile
    manipulator_states = _sample_manipulator_states()
    return profiles, sampled_points, manipulator_states


def main():
    profiles, sampled_points, manipulator_states = generate_profiles()
    if not profiles:
        raise RuntimeError("No valid workspace profiles were generated.")

    _save_raw_scatter_debug_figure(sampled_points, profiles, manipulator_states)

    if SEPARATE_PLOTS:
        fig = plt.figure(figsize=FIGSIZE)
        pseudo_proj_matrix = None
        if MAIN_VIEW_STYLE == "pseudo_3d":
            pseudo_proj_matrix = _pseudo_projection_matrix(list(profiles.values()), manipulator_states)
        axes = []
        for i in range(max(4, len(PLOT_MODES))):
            if MAIN_VIEW_STYLE == "pseudo_3d":
                axes.append(fig.add_subplot(2, 2, i + 1))
            else:
                axes.append(fig.add_subplot(2, 2, i + 1, projection="3d"))
        all_profiles = list(profiles.values())
        for mode, ax in zip(PLOT_MODES, axes):
            if mode not in profiles:
                ax.set_axis_off()
                continue
            if MAIN_VIEW_STYLE == "pseudo_3d":
                draw_mode_main_pseudo(ax, profiles[mode], mode, pseudo_proj_matrix)
                _overlay_manipulators_main_pseudo(ax, manipulator_states.get(mode, []), pseudo_proj_matrix)
                configure_pseudo_main_axes(ax, all_profiles, manipulator_states, proj_matrix=pseudo_proj_matrix)
            else:
                draw_mode_workspace(ax, profiles[mode], mode, style=MAIN_VIEW_STYLE)
                _overlay_manipulators(ax, manipulator_states.get(mode, []))
                configure_axes(ax, all_profiles, style=MAIN_VIEW_STYLE, role="main")
            ax.set_title(_display_label_for_mode(mode))
        for ax in axes[len(PLOT_MODES):]:
            ax.set_axis_off()
        plt.tight_layout()
    else:
        if SHOW_SIDE_VIEW or SHOW_SAMPLE_SCATTER:
            fig = plt.figure(figsize=(11.5, 7.2))
            grid = GridSpec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], figure=fig)
            if MAIN_VIEW_STYLE == "pseudo_3d":
                ax = fig.add_subplot(grid[:, 0])
            else:
                ax = fig.add_subplot(grid[:, 0], projection="3d")
            side_ax = None
            if SHOW_SIDE_VIEW:
                side_ax = fig.add_subplot(grid[0, 1], projection="3d" if SIDE_VIEW_STYLE == "real_3d" else None)
            scatter_ax = fig.add_subplot(grid[1, 1]) if SHOW_SAMPLE_SCATTER else None
        else:
            fig = plt.figure(figsize=(7, 7))
            if MAIN_VIEW_STYLE == "pseudo_3d":
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111, projection="3d")
            side_ax = None
            scatter_ax = None
        all_profiles = list(profiles.values())
        scatter_modes = []
        for mode in PLOT_MODES:
            if mode == 0:
                scatter_modes.extend([source_mode for source_mode in COMBINED_SOURCE_MODES if source_mode in sampled_points])
            elif mode in sampled_points:
                scatter_modes.append(mode)
            scatter_modes = list(dict.fromkeys(scatter_modes))
        pseudo_proj_matrix = None
        if MAIN_VIEW_STYLE == "pseudo_3d":
            pseudo_proj_matrix = _pseudo_projection_matrix(all_profiles, manipulator_states)
        for mode in PLOT_MODES:
            if mode not in profiles:
                continue
            if MAIN_VIEW_STYLE == "pseudo_3d":
                draw_mode_main_pseudo(ax, profiles[mode], mode, pseudo_proj_matrix)
                _overlay_manipulators_main_pseudo(ax, manipulator_states.get(mode, []), pseudo_proj_matrix)
            else:
                draw_mode_workspace(ax, profiles[mode], mode, style=MAIN_VIEW_STYLE)
                _overlay_manipulators(ax, manipulator_states.get(mode, []))
            if side_ax is not None:
                if SIDE_VIEW_STYLE == "pseudo_3d":
                    draw_mode_side_view(side_ax, profiles[mode], mode)
                    _overlay_manipulators_side_view(side_ax, manipulator_states.get(mode, []))
                else:
                    draw_mode_workspace(side_ax, profiles[mode], mode, style=SIDE_VIEW_STYLE)
                    _overlay_manipulators(side_ax, manipulator_states.get(mode, []))
        if scatter_ax is not None:
            for scatter_mode in scatter_modes:
                draw_mode_side_scatter(scatter_ax, sampled_points[scatter_mode], scatter_mode)
            if SHOW_SCATTER_PROFILE_CURVES:
                for mode in PLOT_MODES:
                    if mode in profiles:
                        overlay_profile_curves(scatter_ax, profiles[mode], mode)
        if MAIN_VIEW_STYLE == "pseudo_3d":
            configure_pseudo_main_axes(ax, all_profiles, manipulator_states, proj_matrix=pseudo_proj_matrix)
        else:
            configure_axes(ax, all_profiles, style=MAIN_VIEW_STYLE, role="main")
        if side_ax is not None:
            if SIDE_VIEW_STYLE == "pseudo_3d":
                configure_side_view_axes(side_ax, all_profiles, manipulator_states, sampled_points)
            else:
                configure_axes(side_ax, all_profiles, style=SIDE_VIEW_STYLE, role="side")
            side_ax.set_title("Side View")
        if scatter_ax is not None:
            scatter_input_points = {mode: sampled_points[mode] for mode in scatter_modes if mode in sampled_points}
            configure_side_view_axes(scatter_ax, all_profiles, manipulator_states, scatter_input_points)
            scatter_ax.set_title("Sample Scatter + Fit" if SHOW_SCATTER_PROFILE_CURVES else "Sample Scatter")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=False)
        plt.tight_layout()

    output_path = OUTPUT_PATH
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_path.resolve()}")

    if _has_interactive_display():
        plt.show()
    else:
        if output_path is None:
            FALLBACK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(FALLBACK_OUTPUT_PATH, dpi=300, bbox_inches="tight")
            print(f"No interactive display detected, saved figure to: {FALLBACK_OUTPUT_PATH.resolve()}")
        else:
            print("No interactive display detected, skipped plt.show().")


if __name__ == "__main__":
    main()
