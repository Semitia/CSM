"""
Module: plot_workspace_2.py
Description: Directly sample a side-view workspace profile from the robot model,
smooth the reachable/unreachable contours, and revolve them into a paper-like 3D plot.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

from csm.model import CSM

# ===== Data source =====
CONFIG_NAME = "csm_cfg_0.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
# PLOT_MODES = [1, 2, 3, 4]
PLOT_MODES = [1]
CACHE_DIR = Path("./data/profile_cache")
USE_CACHE = True
FORCE_REBUILD_CACHE = False

# ===== Direct sampling =====
# We sample only the side-view generating variables.
# phi / delta_1 / delta_2 are fixed to 0 and are reintroduced by revolving the profile.
MODE_SAMPLE_RES = {
    1: {"length": 120, "theta1": 1, "theta2": 120},
    2: {"length": 110, "theta1": 1, "theta2": 120},
    3: {"length": 90, "theta1": 90, "theta2": 90},
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
# ===== Surface rendering =====
REVOLVE_SAMPLES = 40
CAP_RADIAL_SAMPLES = 18
SHOW_UNREACHABLE = True
MAIN_VIEW_STYLE = "real_3d"   # "real_3d" | "pseudo_3d"
SIDE_VIEW_STYLE = "pseudo_3d"   # "pseudo_3d" | "real_3d"

MODE_COLORS = {1: "#E9A3A7", 2: "#F0E0AA", 3: "#9FD4EA", 4: "#CFCFCF"}
MODE_LABELS = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
UNREACHABLE_COLOR = "#D79A6B"

REACH_ALPHA = 0.34
UNREACHABLE_ALPHA = 0.40

# ===== Figure =====
SEPARATE_PLOTS = False
OUTPUT_PATH = None
FIGSIZE = (10, 10)
VIEW_ELEV = 18
VIEW_AZIM = -40
SHOW_AXES = True
DISPLAY_IN_MM = True
SHOW_SIDE_VIEW = True
SHOW_SAMPLE_SCATTER = True
PSEUDO_VIEW_ELEV = 10
PSEUDO_VIEW_AZIM = -88
SIDE_REAL_VIEW_ELEV = 0
SIDE_REAL_VIEW_AZIM = -90


def _linspace_from_zero(stop, num):
    if num <= 1 or stop <= 0:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, float(stop), int(num))


def _cache_file_path(mode):
    return CACHE_DIR / f"{CONFIG_PATH.stem}_mode{mode}_profile_cache.npz"


def _current_cache_metadata(mode):
    return {
        "config_name": CONFIG_NAME,
        "config_path": str(CONFIG_PATH),
        "mode": int(mode),
        "sample_cfg": MODE_SAMPLE_RES[mode],
    }


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


def _count_mode_side_samples(csm, mode, sample_cfg):
    length_res = int(sample_cfg["length"])
    theta1_res = int(sample_cfg["theta1"])
    theta2_res = int(sample_cfg["theta2"])

    total = 0
    if mode == 1:
        for L2 in _linspace_from_zero(csm.L_20, length_res):
            total += len(_linspace_from_zero(csm.kappa_20 * L2, theta2_res))
    elif mode == 2:
        for _Lr in _linspace_from_zero(csm.L_r0, length_res):
            total += len(_linspace_from_zero(csm.kappa_20 * csm.L2, theta2_res))
    elif mode == 3:
        for L1 in _linspace_from_zero(csm.L_10, length_res):
            theta1_vals = _linspace_from_zero(csm.kappa_10 * L1, theta1_res)
            total += len(theta1_vals) * len(_linspace_from_zero(csm.kappa_20 * csm.L2, theta2_res))
    elif mode == 4:
        for _Ls in _linspace_from_zero(csm.L_s0, length_res):
            theta1_vals = _linspace_from_zero(csm.kappa_10 * csm.L1, theta1_res)
            total += len(theta1_vals) * len(_linspace_from_zero(csm.kappa_20 * csm.L2, theta2_res))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return total


def sample_mode_side_points(csm, mode, sample_cfg):
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
                theta2_vals = _linspace_from_zero(csm.kappa_20 * L2, theta2_res)
                for theta_2 in theta2_vals:
                    csm.theta_2 = theta_2
                    csm.update()
                    side_points.append([_side_radius(csm.pose[:3]), float(csm.pose[2])])
                    pbar.update(1)

        elif mode == 2:
            for Lr in _linspace_from_zero(csm.L_r0, length_res):
                csm.Lr = Lr
                theta2_vals = _linspace_from_zero(csm.kappa_20 * csm.L2, theta2_res)
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
                theta1_vals = _linspace_from_zero(csm.kappa_10 * L1, theta1_res)
                for theta_1 in theta1_vals:
                    csm.theta_1 = theta_1
                    theta2_vals = _linspace_from_zero(csm.kappa_20 * csm.L2, theta2_res)
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
                theta1_vals = _linspace_from_zero(csm.kappa_10 * csm.L1, theta1_res)
                for theta_1 in theta1_vals:
                    csm.theta_1 = theta_1
                    theta2_vals = _linspace_from_zero(csm.kappa_20 * csm.L2, theta2_res)
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


def _side_tip_radius_for_endpoint(r0, z0, slope, z_tip):
    dz = z0 - z_tip
    if dz <= 0 or slope is None or slope >= -1e-9:
        return None
    return float(r0 + dz / slope + dz * np.sqrt(1.0 + 1.0 / (slope * slope)))


def _solve_shared_side_tip(outer_z, outer_r, inner_z, inner_r, max_extend_bins=12):
    if outer_z.size < 3 or inner_z.size < 3:
        return None

    slope_outer = _estimate_dr_dz(outer_z, outer_r, at_end=False)
    slope_inner = _estimate_dr_dz(inner_z, inner_r, at_end=False)
    if slope_outer is None or slope_inner is None:
        return None
    if slope_outer >= -1e-9 or slope_inner >= -1e-9:
        return None

    z_outer_0 = float(outer_z[0])
    z_inner_0 = float(inner_z[0])
    r_outer_0 = float(outer_r[0])
    r_inner_0 = float(inner_r[0])
    dz_outer = np.median(np.diff(outer_z)) if outer_z.size >= 2 else 0.0
    dz_inner = np.median(np.diff(inner_z)) if inner_z.size >= 2 else 0.0
    max_extend_dz = max_extend_bins * max(dz_outer, dz_inner)
    if max_extend_dz <= 0:
        return None

    k_outer = 1.0 / slope_outer + np.sqrt(1.0 + 1.0 / (slope_outer * slope_outer))
    k_inner = 1.0 / slope_inner + np.sqrt(1.0 + 1.0 / (slope_inner * slope_inner))
    denom = k_inner - k_outer
    if abs(denom) < 1e-12:
        return None

    z_tip = (r_outer_0 - r_inner_0 + k_outer * z_outer_0 - k_inner * z_inner_0) / denom
    z_high = min(z_outer_0, z_inner_0) - 1e-9
    z_low = z_high - max_extend_dz
    if not np.isfinite(z_tip) or z_tip >= z_high or z_tip < z_low:
        return None

    r_tip = _side_tip_radius_for_endpoint(r_outer_0, z_outer_0, slope_outer, z_tip)
    if r_tip is None or not np.isfinite(r_tip):
        return None
    return float(z_tip), float(r_tip)


def _prepend_side_arc(z_vals, r_vals, z_tip, r_tip, arc_samples=24):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 3:
        return z_vals, r_vals

    slope = _estimate_dr_dz(z_vals, r_vals, at_end=False)
    if slope is None or slope >= -1e-9:
        return z_vals, r_vals

    z_start = float(z_vals[0])
    r_start = float(r_vals[0])
    if z_tip >= z_start or r_tip <= r_start:
        return z_vals, r_vals

    center_r = r_start + (z_start - z_tip) / slope
    radius = r_tip - center_r
    if radius <= 0 or not np.isfinite(radius):
        return z_vals, r_vals

    theta_start = float(np.arctan2(z_start - z_tip, r_start - center_r))
    theta_arc = np.linspace(0.0, theta_start, arc_samples)
    arc_r = center_r + radius * np.cos(theta_arc)
    arc_z = z_tip + radius * np.sin(theta_arc)
    return (
        np.concatenate([arc_z[:-1], z_vals]),
        np.concatenate([arc_r[:-1], r_vals]),
    )


def _build_plot_curves(profile, mode):
    z_outer = np.asarray(profile["z"], dtype=float)
    r_outer = np.asarray(profile["outer_r"], dtype=float)

    inner_mask = np.asarray(profile["inner_r"], dtype=float) > 0.0
    if np.count_nonzero(inner_mask) < 3:
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

    side_tip = _solve_shared_side_tip(
        np.asarray(profile["z"], dtype=float),
        np.asarray(profile["outer_r"], dtype=float),
        z_inner,
        r_inner,
        max_extend_bins=PROFILE_INNER_APEX_MAX_EXTEND_BINS,
    )
    if side_tip is not None:
        z_tip, r_tip = side_tip
        z_outer, r_outer = _prepend_side_arc(z_outer, r_outer, z_tip, r_tip)
        z_inner, r_inner = _prepend_side_arc(z_inner, r_inner, z_tip, r_tip)

    return z_outer, r_outer, z_inner, r_inner


def build_profile_from_side_points(side_points, mode=None):
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

    # Extend the outer curve to the true sampled z-extrema instead of stopping at bin centers.
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
    inner_valid = np.clip(inner_valid, 0.0, outer_valid * 0.98)
    inner_valid[inner_valid < 0.5 * radial_res] = 0.0

    inner_keep = _longest_true_run(inner_valid > 0.0, inner_valid ** 2)
    inner_valid = np.where(inner_keep, inner_valid, 0.0)

    return {
        "z": z_valid,
        "outer_r": outer_valid,
        "inner_r": inner_valid,
    }


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


def draw_mode_workspace(ax, profile, mode, style="real_3d"):
    outer_z_plot, outer_r_plot, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    if style == "pseudo_3d":
        _plot_profile_wall(
            ax,
            outer_z_plot,
            outer_r_plot,
            color=MODE_COLORS[mode],
            alpha=0.78,
            label=MODE_LABELS[mode],
            y_plane=0.0,
        )
        if SHOW_UNREACHABLE:
            if inner_z_plot is not None and inner_z_plot.size >= 4:
                _plot_profile_wall(
                    ax,
                    inner_z_plot,
                    inner_r_plot,
                    color=UNREACHABLE_COLOR,
                    alpha=0.82,
                    label=None,
                    y_plane=0.0,
                )
        return

    plot_revolved_profile(
        ax,
        outer_z_plot,
        outer_r_plot,
        color=MODE_COLORS[mode],
        alpha=REACH_ALPHA,
        label=MODE_LABELS[mode],
        cap_ends=False,
    )

    if SHOW_UNREACHABLE:
        if inner_z_plot is not None and inner_z_plot.size >= 4:
            plot_revolved_profile(
                ax,
                inner_z_plot,
                inner_r_plot,
                color=UNREACHABLE_COLOR,
                alpha=UNREACHABLE_ALPHA,
                label=None,
                cap_ends=False,
            )


def draw_mode_side_view(ax, profile, mode):
    z, outer_r, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    ax.fill_betweenx(z, -outer_r, outer_r, color=MODE_COLORS[mode], alpha=0.65, linewidth=0)

    if SHOW_UNREACHABLE and inner_z_plot is not None and inner_z_plot.size >= 4:
        ax.fill_betweenx(
            inner_z_plot,
            -inner_r_plot,
            inner_r_plot,
            color=UNREACHABLE_COLOR,
            alpha=0.80,
            linewidth=0,
        )

    ax.plot(outer_r, z, color=MODE_COLORS[mode], linewidth=1.0, alpha=0.9)
    ax.plot(-outer_r, z, color=MODE_COLORS[mode], linewidth=1.0, alpha=0.9)
    if inner_z_plot is not None and inner_z_plot.size >= 4:
        ax.plot(inner_r_plot, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=1.0, alpha=0.95)
        ax.plot(-inner_r_plot, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=1.0, alpha=0.95)

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


def draw_mode_side_scatter(ax, side_points, mode):
    if side_points is None or side_points.size == 0:
        return

    r = np.asarray(side_points[:, 0], dtype=float)
    z = np.asarray(side_points[:, 1], dtype=float)
    ax.scatter(r, z, s=2.5, color=MODE_COLORS[mode], alpha=0.14, edgecolors="none")
    ax.scatter(-r, z, s=2.5, color=MODE_COLORS[mode], alpha=0.14, edgecolors="none")

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


def overlay_profile_curves(ax, profile, mode):
    z, outer_r, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    ax.plot(outer_r, z, color=MODE_COLORS[mode], linewidth=2.0, alpha=0.95)
    ax.plot(-outer_r, z, color=MODE_COLORS[mode], linewidth=2.0, alpha=0.95)

    if inner_z_plot is not None and inner_z_plot.size >= 4:
        ax.plot(inner_r_plot, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=2.0, alpha=0.95)
        ax.plot(-inner_r_plot, inner_z_plot, color=UNREACHABLE_COLOR, linewidth=2.0, alpha=0.95)


def configure_axes(ax, all_profiles, style="real_3d", role="main"):
    outer_max = 0.0
    z_min = np.inf
    z_max = -np.inf
    for profile in all_profiles:
        if len(profile["z"]) == 0:
            continue
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


def configure_side_view_axes(ax, all_profiles):
    outer_max = 0.0
    z_min = np.inf
    z_max = -np.inf
    for profile in all_profiles:
        if len(profile["z"]) == 0:
            continue
        outer_max = max(outer_max, float(np.max(profile["outer_r"])))
        z_min = min(z_min, float(np.min(profile["z"])))
        z_max = max(z_max, float(np.max(profile["z"])))

    if outer_max <= 0.0 or not np.isfinite(z_min) or not np.isfinite(z_max):
        return

    pad_r = 0.08 * outer_max
    pad_z = 0.08 * (z_max - z_min)
    ax.set_xlim(-(outer_max + pad_r), outer_max + pad_r)
    ax.set_ylim(z_min - pad_z, z_max + pad_z)


def generate_profiles():
    csm = CSM.from_config(CONFIG_PATH)
    profiles = {}
    sampled_points = {}
    for mode in PLOT_MODES:
        side_points = _load_profile_cache(mode)
        if side_points is not None:
            print(f"Mode {mode}: loaded cached sampled points")
        else:
            side_points = sample_mode_side_points(csm, mode, MODE_SAMPLE_RES[mode])
            _save_profile_cache(mode, side_points)

        sampled_points[mode] = side_points
        profile = build_profile_from_side_points(side_points, mode=mode)
        if profile is None:
            print(f"Mode {mode}: failed to build profile")
            continue
        profiles[mode] = profile
    return profiles, sampled_points


def main():
    profiles, sampled_points = generate_profiles()
    if not profiles:
        raise RuntimeError("No valid workspace profiles were generated.")

    if SEPARATE_PLOTS:
        fig = plt.figure(figsize=FIGSIZE)
        axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
        all_profiles = list(profiles.values())
        for mode, ax in zip((1, 2, 3, 4), axes):
            if mode not in profiles:
                ax.set_axis_off()
                continue
            draw_mode_workspace(ax, profiles[mode], mode, style=MAIN_VIEW_STYLE)
            ax.set_title(MODE_LABELS[mode])
            configure_axes(ax, all_profiles, style=MAIN_VIEW_STYLE, role="main")
        plt.tight_layout()
    else:
        if SHOW_SIDE_VIEW or SHOW_SAMPLE_SCATTER:
            fig = plt.figure(figsize=(11.5, 7.2))
            grid = GridSpec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], figure=fig)
            ax = fig.add_subplot(grid[:, 0], projection="3d")
            side_ax = None
            if SHOW_SIDE_VIEW:
                side_ax = fig.add_subplot(grid[0, 1], projection="3d" if SIDE_VIEW_STYLE == "real_3d" else None)
            scatter_ax = fig.add_subplot(grid[1, 1]) if SHOW_SAMPLE_SCATTER else None
        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            side_ax = None
            scatter_ax = None
        all_profiles = list(profiles.values())
        for mode in PLOT_MODES:
            if mode not in profiles:
                continue
            draw_mode_workspace(ax, profiles[mode], mode, style=MAIN_VIEW_STYLE)
            if side_ax is not None:
                if SIDE_VIEW_STYLE == "pseudo_3d":
                    draw_mode_side_view(side_ax, profiles[mode], mode)
                else:
                    draw_mode_workspace(side_ax, profiles[mode], mode, style=SIDE_VIEW_STYLE)
            if scatter_ax is not None and mode in sampled_points:
                draw_mode_side_scatter(scatter_ax, sampled_points[mode], mode)
                overlay_profile_curves(scatter_ax, profiles[mode], mode)
        configure_axes(ax, all_profiles, style=MAIN_VIEW_STYLE, role="main")
        if side_ax is not None:
            if SIDE_VIEW_STYLE == "pseudo_3d":
                configure_side_view_axes(side_ax, all_profiles)
            else:
                configure_axes(side_ax, all_profiles, style=SIDE_VIEW_STYLE, role="side")
            side_ax.set_title("Side View")
        if scatter_ax is not None:
            configure_side_view_axes(scatter_ax, all_profiles)
            scatter_ax.set_title("Sample Scatter")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=False)
        plt.tight_layout()

    if OUTPUT_PATH is not None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {OUTPUT_PATH.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
