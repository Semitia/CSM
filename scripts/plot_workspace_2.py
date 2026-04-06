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
CONFIG_NAME = "csm_cfg_3mm.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
# PLOT_MODES = [1, 2, 3, 4]
PLOT_MODES = [1]
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

MODE_COLORS = {1: "#E9A3A7", 2: "#F0E0AA", 3: "#9FD4EA", 4: "#CFCFCF"}
MODE_LABELS = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
UNREACHABLE_COLOR = "#D79A6B"
COMBINED_REACHABLE_COLOR = "#86BFA3"

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
SHOW_MANIPULATOR = True
MANIPULATOR_RENDER_MODE = "detailed"
MANIPULATOR_RNG_SEED = 20260406


def _linspace_from_zero(stop, num):
    if num <= 1 or stop <= 0:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, float(stop), int(num))


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

    # Drawing helpers expect r as a function of z for regular modes; mode1 uses upper/lower z(r).
    return {
        "z": np.concatenate((lower_plot, upper_plot)),
        "outer_r": np.concatenate((r_plot, r_plot)),
        "inner_r": np.zeros_like(np.concatenate((r_plot, r_plot))),
        "mode1_r": r_plot,
        "mode1_upper_z": upper_plot,
        "mode1_lower_z": lower_plot,
    }


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
    inner_valid = np.clip(inner_valid, 0.0, outer_valid * 0.98)
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
    built = _build_mode0_void_outline_from_source_profiles(
        source_profiles,
        mode0_outer_curve=mode0_outer_curve,
    )
    if built is None:
        return profile
    outline, outer_override = built

    profile["mode0_void_outline_r"] = outline[:, 0]
    profile["mode0_void_outline_z"] = outline[:, 1]
    profile["mode0_void_floor_z"] = float(np.min(outline[:, 1]))
    profile["mode0_outer_r"] = outer_override[:, 0]
    profile["mode0_outer_z"] = outer_override[:, 1]
    return profile


def _build_mode1_plot_curves(profile):
    upper_r = np.asarray(profile["mode1_r"], dtype=float)
    lower_r = np.asarray(profile["mode1_r"], dtype=float)
    upper_z = np.asarray(profile["mode1_upper_z"], dtype=float)
    lower_z = np.asarray(profile["mode1_lower_z"], dtype=float)

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


def build_profile_from_side_points(side_points, mode=None):
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
        void_lower_z = np.full_like(void_r, void_floor_z)

        if style == "pseudo_3d":
            _plot_profile_wall(
                ax,
                outer_z_plot,
                outer_r_plot,
                color=display_color,
                alpha=0.78,
                label=display_label,
                y_plane=0.0,
            )
            if SHOW_UNREACHABLE and void_r.size >= 4:
                if "mode0_void_outline_r" in profile:
                    _plot_mode0_void_wall(
                        ax,
                        profile,
                        color=UNREACHABLE_COLOR,
                        alpha=0.82,
                        label=None,
                        y_plane=0.0,
                    )
                else:
                    _plot_symmetric_band_wall(
                        ax,
                        void_r,
                        void_lower_z,
                        void_upper_z,
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
            color=display_color,
            alpha=REACH_ALPHA,
            label=display_label,
            cap_ends=False,
        )
        if SHOW_UNREACHABLE and void_r.size >= 4:
            plot_revolved_profile(
                ax,
                void_outline_z,
                void_outline_r,
                color=UNREACHABLE_COLOR,
                alpha=UNREACHABLE_ALPHA,
                label=None,
                cap_ends=False,
            )
        return

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)

        if style == "pseudo_3d":
            x_poly = np.concatenate([upper_r, lower_r[::-1], -lower_r, -upper_r[::-1]])
            z_poly = np.concatenate([upper_z, lower_z[::-1], lower_z, upper_z[::-1]])
            y_poly = np.zeros_like(x_poly)
            verts = np.column_stack((x_poly, y_poly, z_poly))
            poly = ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=np.array([[0, i, i + 1] for i in range(1, len(verts) - 1)], dtype=int),
                color=display_color,
                alpha=0.78,
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
            ax.plot([], [], [], color=display_color, alpha=0.78, label=display_label)
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
            lower_z,
            lower_r,
            color=display_color,
            alpha=REACH_ALPHA,
            label=None,
            cap_ends=False,
        )
        return

    outer_z_plot, outer_r_plot, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    if style == "pseudo_3d":
        _plot_profile_wall(
            ax,
            outer_z_plot,
            outer_r_plot,
            color=display_color,
            alpha=0.78,
            label=display_label,
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
                inner_r_plot,
                color=UNREACHABLE_COLOR,
                alpha=UNREACHABLE_ALPHA,
                label=None,
                cap_ends=False,
            )


def draw_mode_side_view(ax, profile, mode):
    display_color = _display_color_for_mode(mode)
    if mode == 0 and "mode0_void_r" in profile:
        z, outer_r, _, _ = _build_plot_curves(profile, mode)
        void_r = np.asarray(profile["mode0_void_r"], dtype=float)
        void_upper_z = np.asarray(profile["mode0_void_upper_z"], dtype=float)
        void_polygon = _mode0_void_polygon(profile)
        void_floor_z = float(profile["mode0_void_floor_z"])

        ax.fill_betweenx(z, -outer_r, outer_r, color=display_color, alpha=0.65, linewidth=0)
        if SHOW_UNREACHABLE and void_r.size >= 4:
            if void_polygon is not None:
                polygon_r, polygon_z, _ = void_polygon
                ax.fill(polygon_r, polygon_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
                ax.fill(-polygon_r, polygon_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
            else:
                ax.fill_between(void_r, void_floor_z, void_upper_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)
                ax.fill_between(-void_r, void_floor_z, void_upper_z, color=UNREACHABLE_COLOR, alpha=0.80, linewidth=0)

        ax.plot(outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(-outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
        if SHOW_UNREACHABLE and void_r.size >= 4:
            if "mode0_void_outline_r" in profile:
                void_outline_r = np.asarray(profile["mode0_void_outline_r"], dtype=float)
                void_outline_z = np.asarray(profile["mode0_void_outline_z"], dtype=float)
                ax.plot(void_outline_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=1.0, alpha=0.95)
                ax.plot(-void_outline_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=1.0, alpha=0.95)
            else:
                ax.plot(void_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=1.0, alpha=0.95)
                ax.plot(-void_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=1.0, alpha=0.95)

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

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        pos_x = np.concatenate([upper_r, lower_r[::-1]])
        pos_y = np.concatenate([upper_z, lower_z[::-1]])
        neg_x = -pos_x

        ax.fill(pos_x, pos_y, color=display_color, alpha=0.65, linewidth=0)
        ax.fill(neg_x, pos_y, color=display_color, alpha=0.65, linewidth=0)
        ax.plot(upper_r, upper_z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(-upper_r, upper_z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(lower_r, lower_z, color=display_color, linewidth=1.0, alpha=0.9)
        ax.plot(-lower_r, lower_z, color=display_color, linewidth=1.0, alpha=0.9)

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
        ax.fill_betweenx(
            inner_z_plot,
            -inner_r_plot,
            inner_r_plot,
            color=UNREACHABLE_COLOR,
            alpha=0.80,
            linewidth=0,
        )

    ax.plot(outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
    ax.plot(-outer_r, z, color=display_color, linewidth=1.0, alpha=0.9)
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
    ax.scatter(r, z, s=2.5, color=_display_color_for_mode(mode), alpha=0.14, edgecolors="none")
    ax.scatter(-r, z, s=2.5, color=_display_color_for_mode(mode), alpha=0.14, edgecolors="none")

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
    display_color = _display_color_for_mode(mode)
    if mode == 0 and "mode0_void_r" in profile:
        z, outer_r, _, _ = _build_plot_curves(profile, mode)
        void_r = np.asarray(profile["mode0_void_r"], dtype=float)
        void_upper_z = np.asarray(profile["mode0_void_upper_z"], dtype=float)
        ax.plot(outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(-outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)
        if void_r.size >= 4:
            if "mode0_void_outline_r" in profile:
                void_outline_r = np.asarray(profile["mode0_void_outline_r"], dtype=float)
                void_outline_z = np.asarray(profile["mode0_void_outline_z"], dtype=float)
                ax.plot(void_outline_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=2.0, alpha=0.95)
                ax.plot(-void_outline_r, void_outline_z, color=UNREACHABLE_COLOR, linewidth=2.0, alpha=0.95)
            else:
                ax.plot(void_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=2.0, alpha=0.95)
                ax.plot(-void_r, void_upper_z, color=UNREACHABLE_COLOR, linewidth=2.0, alpha=0.95)
        return

    if mode == 1 and "mode1_r" in profile:
        upper_r, upper_z, lower_r, lower_z = _build_mode1_plot_curves(profile)
        ax.plot(upper_r, upper_z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(-upper_r, upper_z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(lower_r, lower_z, color=display_color, linewidth=2.0, alpha=0.95)
        ax.plot(-lower_r, lower_z, color=display_color, linewidth=2.0, alpha=0.95)
        return

    z, outer_r, inner_z_plot, inner_r_plot = _build_plot_curves(profile, mode)

    ax.plot(outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)
    ax.plot(-outer_r, z, color=display_color, linewidth=2.0, alpha=0.95)

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


def configure_side_view_axes(ax, all_profiles):
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

    pad_r = 0.08 * outer_max
    pad_z = 0.08 * max(z_max, z_max - z_min)
    ax.set_xlim(-(outer_max + pad_r), outer_max + pad_r)
    ax.set_ylim(0.0, z_max + pad_z)


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
        theta_2 = float(rng.uniform(0.0, csm.kappa_20 * L2)) if L2 > 0.0 else 0.0
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    elif mode == 2:
        Lr = float(rng.uniform(0.0, csm.L_r0))
        theta_2 = float(rng.uniform(0.0, csm.kappa_20 * csm.L_20))
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    elif mode == 3:
        L1 = float(rng.uniform(0.0, csm.L_10))
        Lr = float(csm.L_r0)
        theta_1 = float(rng.uniform(0.0, csm.kappa_10 * L1)) if L1 > 0.0 else 0.0
        theta_2 = float(rng.uniform(0.0, csm.kappa_20 * csm.L_20))
        delta_1 = float(rng.uniform(0.0, 2.0 * np.pi))
        delta_2 = float(rng.uniform(0.0, 2.0 * np.pi))
    elif mode == 4:
        L1 = float(csm.L_10)
        L2 = float(csm.L_20)
        Lr = float(csm.L_r0)
        Ls = float(rng.uniform(0.0, csm.L_s0))
        theta_1 = float(rng.uniform(0.0, csm.kappa_10 * csm.L_10))
        theta_2 = float(rng.uniform(0.0, csm.kappa_20 * csm.L_20))
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
                    source_profile = build_profile_from_side_points(side_points, mode=source_mode)
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
        profile = build_profile_from_side_points(side_points, mode=mode)
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

    if SEPARATE_PLOTS:
        fig = plt.figure(figsize=FIGSIZE)
        axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(max(4, len(PLOT_MODES)))]
        all_profiles = list(profiles.values())
        for mode, ax in zip(PLOT_MODES, axes):
            if mode not in profiles:
                ax.set_axis_off()
                continue
            draw_mode_workspace(ax, profiles[mode], mode, style=MAIN_VIEW_STYLE)
            _overlay_manipulators(ax, manipulator_states.get(mode, []))
            ax.set_title(_display_label_for_mode(mode))
            configure_axes(ax, all_profiles, style=MAIN_VIEW_STYLE, role="main")
        for ax in axes[len(PLOT_MODES):]:
            ax.set_axis_off()
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
        scatter_modes = []
        for mode in PLOT_MODES:
            if mode == 0:
                scatter_modes.extend([source_mode for source_mode in COMBINED_SOURCE_MODES if source_mode in sampled_points])
            elif mode in sampled_points:
                scatter_modes.append(mode)
        scatter_modes = list(dict.fromkeys(scatter_modes))

        for mode in PLOT_MODES:
            if mode not in profiles:
                continue
            draw_mode_workspace(ax, profiles[mode], mode, style=MAIN_VIEW_STYLE)
            _overlay_manipulators(ax, manipulator_states.get(mode, []))
            if side_ax is not None:
                if SIDE_VIEW_STYLE == "pseudo_3d":
                    draw_mode_side_view(side_ax, profiles[mode], mode)
                else:
                    draw_mode_workspace(side_ax, profiles[mode], mode, style=SIDE_VIEW_STYLE)
        if scatter_ax is not None:
            for scatter_mode in scatter_modes:
                draw_mode_side_scatter(scatter_ax, sampled_points[scatter_mode], scatter_mode)
            for mode in PLOT_MODES:
                if mode in profiles:
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
