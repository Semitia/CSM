"""
Raw sample scatter plotter for the 3 mm configuration, mode 3.

This script only keeps:
1. Config loading
2. Sample-cache file interaction
3. Sampling when cache is missing
4. Raw side-view scatter plotting
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import binary_closing, gaussian_filter, gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from csm.model import CSM
from plot_workspace_2 import (
    FIGSIZE,
    _has_interactive_display,
    configure_side_view_axes,
    draw_mode_side_scatter,
    draw_mode_side_view,
    overlay_profile_curves,
    sample_mode_side_points,
)


CONFIG_NAME = "csm_cfg_3mm_2.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
PLOT_MODE = 3
SAMPLE_CFG = {"length": 90, "theta1": 90, "theta2": 90}
OUTPUT_PATH = Path("./data/plot_workspace_2_hollow_fit.png")
CACHE_DIR = Path("./data/profile_cache")
GRID_Z_BINS = 280
GRID_R_BINS = 260
GRID_SIGMA_Z = 1.35
GRID_SIGMA_R = 1.1
GLOBAL_OCC_THRESHOLD = 0.055
ROW_OCC_THRESHOLD = 0.17
MIN_ROW_RUN_BINS = 3
OUTER_MIN_ROWS = 12
INNER_MIN_ROWS = 18
OUTER_ROW_QUANTILE = 0.995
CURVE_SMOOTH_SIGMA_OUTER = 3.2
CURVE_SMOOTH_SIGMA_INNER = 1.6
AXIS_NOISE_R = 0.0045
INNER_BOTTOM_FLAT_MAX_BINS = 20
TOP_ARC_MAX_EXTEND_BINS = 24


def _find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _keep_longest_run(mask: np.ndarray, weights: np.ndarray, min_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    keep = np.zeros_like(mask, dtype=bool)
    best = None
    best_score = -np.inf
    for start, end in _find_true_runs(mask):
        run_len = end - start + 1
        if run_len < min_len:
            continue
        score = float(np.sum(weights[start:end + 1]))
        if score > best_score:
            best_score = score
            best = (start, end)
    if best is not None:
        keep[best[0]:best[1] + 1] = True
    return keep


def _smooth_valid_curve(z_vals: np.ndarray, r_vals: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 4:
        return z_vals, r_vals
    dense_count = max(240, int(z_vals.size * 4))
    z_dense = np.linspace(float(z_vals[0]), float(z_vals[-1]), dense_count)
    r_dense = np.interp(z_dense, z_vals, r_vals)
    if sigma > 0.0:
        r_dense = gaussian_filter1d(r_dense, sigma=sigma, mode="nearest")
    return z_dense, np.maximum(r_dense, 0.0)


def _estimate_dr_dz(z_vals: np.ndarray, r_vals: np.ndarray, at_end: bool = True):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 3 or r_vals.size != z_vals.size:
        return None

    sample_count = min(7, z_vals.size)
    if at_end:
        z_fit = z_vals[-sample_count:]
        r_fit = r_vals[-sample_count:]
    else:
        z_fit = z_vals[:sample_count]
        r_fit = r_vals[:sample_count]

    dz = np.ptp(z_fit)
    if dz <= 1e-12:
        return None
    return float(np.polyfit(z_fit, r_fit, deg=1)[0])


def _append_axis_arc(z_vals: np.ndarray, r_vals: np.ndarray, max_extend_bins: int = TOP_ARC_MAX_EXTEND_BINS, arc_samples: int = 64):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 3 or r_vals.size != z_vals.size:
        return z_vals, r_vals

    slope = _estimate_dr_dz(z_vals, r_vals, at_end=True)
    if slope is None or slope >= -1e-9:
        return z_vals, r_vals

    r_end = float(r_vals[-1])
    z_end = float(z_vals[-1])
    z_center = z_end + slope * r_end
    radius = float(np.hypot(r_end, z_end - z_center))
    if not np.isfinite(radius) or radius <= 0.0:
        return z_vals, r_vals

    z_axis = z_center + radius
    if not np.isfinite(z_axis) or z_axis <= z_end:
        return z_vals, r_vals

    dz_med = float(np.median(np.diff(z_vals))) if z_vals.size >= 2 else 0.0
    if dz_med > 0.0 and z_axis - z_end > max_extend_bins * dz_med:
        return z_vals, r_vals

    theta_end = float(np.arctan2(z_end - z_center, r_end))
    theta_arc = np.linspace(theta_end, 0.5 * np.pi, arc_samples)
    arc_r = radius * np.cos(theta_arc)
    arc_z = z_center + radius * np.sin(theta_arc)
    return np.concatenate([z_vals, arc_z[1:]]), np.concatenate([r_vals, arc_r[1:]])


def _fit_upper_axis_arc(z_vals: np.ndarray, r_vals: np.ndarray, fit_samples: int = 9, arc_samples: int = 120, max_extra_ratio: float = 0.30):
    z_vals = np.asarray(z_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    if z_vals.size < 4 or r_vals.size != z_vals.size:
        return z_vals, r_vals

    sample_count = min(fit_samples, z_vals.size)
    z_fit = z_vals[-sample_count:]
    r_fit = r_vals[-sample_count:]
    dz = np.ptp(z_fit)
    if dz <= 1e-12:
        return z_vals, r_vals

    slope = float(np.polyfit(z_fit, r_fit, deg=1)[0])
    if slope >= -1e-9:
        return z_vals, r_vals

    r_end = float(r_vals[-1])
    z_end = float(z_vals[-1])
    z_center = z_end + slope * r_end
    radius = float(np.hypot(r_end, z_end - z_center))
    if not np.isfinite(radius) or radius <= 0.0:
        return z_vals, r_vals

    z_axis = z_center + radius
    z_span = float(np.max(z_vals) - np.min(z_vals))
    if not np.isfinite(z_axis) or z_axis <= z_end or z_axis - z_end > max_extra_ratio * max(z_span, 1e-9):
        return z_vals, r_vals

    theta_end = float(np.arctan2(z_end - z_center, r_end))
    theta_arc = np.linspace(theta_end, 0.5 * np.pi, arc_samples)
    arc_r = radius * np.cos(theta_arc)
    arc_z = z_center + radius * np.sin(theta_arc)
    return np.concatenate([z_vals, arc_z[1:]]), np.concatenate([r_vals, arc_r[1:]])


def _build_mode3_special_profile(side_points: np.ndarray):
    side_points = np.asarray(side_points, dtype=float)
    if side_points.ndim != 2 or side_points.shape[1] != 2 or side_points.shape[0] < 32:
        return None

    r = side_points[:, 0]
    z = side_points[:, 1]
    r_max = float(np.max(r))
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if not np.isfinite(r_max) or r_max <= 0.0 or not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return None

    r_edges = np.linspace(0.0, r_max, GRID_R_BINS + 1)
    z_edges = np.linspace(z_min, z_max, GRID_Z_BINS + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_bin_idx = np.clip(np.digitize(z, z_edges) - 1, 0, GRID_Z_BINS - 1)

    row_point_outer = np.full(z_centers.shape, np.nan, dtype=float)
    for row_idx in range(GRID_Z_BINS):
        row_points = r[z_bin_idx == row_idx]
        if row_points.size > 0:
            row_point_outer[row_idx] = float(np.quantile(row_points, OUTER_ROW_QUANTILE))

    hist, _, _ = np.histogram2d(z, r, bins=(z_edges, r_edges))
    if not np.any(hist > 0.0):
        return None

    occ = gaussian_filter(hist.astype(float), sigma=(GRID_SIGMA_Z, GRID_SIGMA_R), mode="nearest")
    global_max = float(np.max(occ))
    if global_max <= 0.0:
        return None

    outer_r = np.full(z_centers.shape, np.nan, dtype=float)
    inner_r = np.full(z_centers.shape, np.nan, dtype=float)
    gap_strength = np.zeros(z_centers.shape, dtype=float)

    for row_idx, row in enumerate(occ):
        row_max = float(np.max(row))
        if row_max <= 0.0:
            continue

        threshold = max(GLOBAL_OCC_THRESHOLD * global_max, ROW_OCC_THRESHOLD * row_max)
        mask = row >= threshold
        if not np.any(mask):
            continue

        mask = binary_closing(mask[np.newaxis, :], structure=np.ones((1, 3), dtype=bool))[0]
        runs = [(start, end) for start, end in _find_true_runs(mask) if end - start + 1 >= MIN_ROW_RUN_BINS]
        if not runs:
            continue

        shell_start, shell_end = runs[-1]
        outer_here = float(r_edges[shell_end + 1])
        if np.isfinite(row_point_outer[row_idx]):
            outer_here = max(outer_here, float(row_point_outer[row_idx]))
        inner_here = float(r_edges[shell_start])
        outer_r[row_idx] = outer_here

        has_inner_gap = len(runs) >= 2 or inner_here > AXIS_NOISE_R
        if has_inner_gap and inner_here < outer_here:
            inner_r[row_idx] = inner_here
            if len(runs) >= 2:
                prev_end = runs[-2][1]
                gap_strength[row_idx] = float(r_edges[shell_start] - r_edges[prev_end + 1])
            else:
                gap_strength[row_idx] = inner_here

    valid_outer = np.isfinite(outer_r)
    if np.count_nonzero(valid_outer) < OUTER_MIN_ROWS:
        return None

    keep_outer = _keep_longest_run(valid_outer, np.nan_to_num(outer_r, nan=0.0), min_len=OUTER_MIN_ROWS)
    valid_outer &= keep_outer
    outer_first = int(np.flatnonzero(valid_outer)[0])
    outer_last = int(np.flatnonzero(valid_outer)[-1])
    outer_z_raw = z_centers[outer_first:outer_last + 1]
    outer_raw = np.interp(
        outer_z_raw,
        z_centers[valid_outer],
        outer_r[valid_outer],
    )
    outer_z_raw = np.concatenate(([z_min], outer_z_raw))
    outer_raw = np.concatenate(([outer_raw[0]], outer_raw))
    outer_z, outer_vals = _smooth_valid_curve(outer_z_raw, outer_raw, CURVE_SMOOTH_SIGMA_OUTER)
    outer_z, outer_vals = _fit_upper_axis_arc(outer_z, outer_vals)
    outer_z, outer_vals = _append_axis_arc(outer_z, outer_vals)

    valid_inner = np.isfinite(inner_r)
    if np.count_nonzero(valid_inner) >= INNER_MIN_ROWS:
        valid_inner = binary_closing(valid_inner, structure=np.ones(9, dtype=bool))
        weights = np.nan_to_num(inner_r, nan=0.0) + 6.0 * gap_strength
        keep_inner = _keep_longest_run(valid_inner, weights, min_len=INNER_MIN_ROWS)
        valid_inner &= keep_inner
    else:
        valid_inner[:] = False

    inner_curve = None
    if np.count_nonzero(valid_inner) >= INNER_MIN_ROWS:
        inner_idx = np.flatnonzero(valid_inner)
        src_inner = np.isfinite(inner_r)
        inner_z = z_centers[inner_idx[0]:inner_idx[-1] + 1]
        inner_vals = np.interp(
            inner_z,
            z_centers[src_inner],
            inner_r[src_inner],
        )
        inner_z, inner_vals = _smooth_valid_curve(inner_z, inner_vals, CURVE_SMOOTH_SIGMA_INNER)
        outer_on_inner = np.interp(inner_z, outer_z, outer_vals)
        radial_margin = max(r_max / max(GRID_R_BINS, 1), 1e-6)
        inner_vals = np.minimum(inner_vals, np.maximum(outer_on_inner - radial_margin, 0.0))
        if inner_z.size >= 2:
            dz_med = float(np.median(np.diff(inner_z)))
            if inner_z[0] - z_min <= INNER_BOTTOM_FLAT_MAX_BINS * max(dz_med, 1e-9):
                inner_z = np.concatenate(([z_min], inner_z))
                inner_vals = np.concatenate(([inner_vals[0]], inner_vals))
        keep = inner_vals > AXIS_NOISE_R
        if np.count_nonzero(keep) >= INNER_MIN_ROWS:
            inner_curve = (inner_z[keep], inner_vals[keep])

    profile = {
        "z": outer_z,
        "outer_r": outer_vals,
        "inner_r": np.zeros_like(outer_z),
    }
    if inner_curve is not None:
        inner_z, inner_vals = inner_curve
        profile["inner_r"] = np.interp(outer_z, inner_z, inner_vals, left=0.0, right=0.0)
        profile["inner_r"] = np.minimum(profile["inner_r"], np.maximum(profile["outer_r"] - r_max / max(GRID_R_BINS, 1), 0.0))
        profile["inner_r"][profile["inner_r"] <= AXIS_NOISE_R] = 0.0
        nz = np.flatnonzero(profile["inner_r"] > 0.0)
        if nz.size > 0:
            first_nz = int(nz[0])
            dz_med = float(np.median(np.diff(profile["z"]))) if profile["z"].size >= 2 else 0.0
            if first_nz > 0 and profile["z"][first_nz] - profile["z"][0] <= INNER_BOTTOM_FLAT_MAX_BINS * max(dz_med, 1e-9):
                profile["inner_r"][:first_nz + 1] = profile["inner_r"][first_nz]

    return profile


def _cache_file_path(mode: int) -> Path:
    return CACHE_DIR / f"{Path(__file__).stem}_{CONFIG_PATH.stem}_mode{mode}_samples.npz"


def _current_cache_metadata(mode: int) -> dict:
    config_sha256 = None
    if CONFIG_PATH.exists():
        config_sha256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    return {
        "config_name": CONFIG_NAME,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha256,
        "mode": int(mode),
        "sample_cfg": SAMPLE_CFG,
    }


def _load_profile_cache(mode: int):
    cache_path = _cache_file_path(mode)
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as payload:
        if "metadata" not in payload or "side_points" not in payload:
            return None

        cached_meta = json.loads(str(payload["metadata"].item()))
        current_meta = _current_cache_metadata(mode)
        for key, value in current_meta.items():
            if cached_meta.get(key) != value:
                return None

        return np.asarray(payload["side_points"], dtype=float)


def _save_profile_cache(mode: int, side_points):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_path(mode)
    np.savez_compressed(
        cache_path,
        metadata=np.asarray(json.dumps(_current_cache_metadata(mode))),
        side_points=np.asarray(side_points, dtype=float),
    )
    print(f"Mode {mode}: cached sampled points -> {cache_path.resolve()}")


def main():
    print(f"Loading CSM config: {CONFIG_PATH}")
    csm = CSM.from_config(CONFIG_PATH)

    side_points = _load_profile_cache(PLOT_MODE)
    if side_points is not None:
        print(f"Mode {PLOT_MODE}: loaded cached sampled points")
    else:
        print(f"Mode {PLOT_MODE}: sample cache not found, sampling now...")
        side_points = sample_mode_side_points(csm, PLOT_MODE, SAMPLE_CFG)
        _save_profile_cache(PLOT_MODE, side_points)

    profile = _build_mode3_special_profile(side_points)
    if profile is None:
        raise RuntimeError("Failed to build specialized mode-3 profile from sampled points.")

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.20)

    ax_scatter = fig.add_subplot(grid[0, 0])
    draw_mode_side_scatter(ax_scatter, side_points, PLOT_MODE)
    overlay_profile_curves(ax_scatter, profile, PLOT_MODE)
    configure_side_view_axes(
        ax_scatter,
        all_profiles=[profile],
        sampled_points_by_mode={PLOT_MODE: side_points},
    )
    ax_scatter.set_title(f"Raw Scatter + Specialized Fit ({CONFIG_NAME}, mode {PLOT_MODE})")

    ax_fit = fig.add_subplot(grid[0, 1])
    draw_mode_side_view(ax_fit, profile, PLOT_MODE)
    configure_side_view_axes(
        ax_fit,
        all_profiles=[profile],
        sampled_points_by_mode={PLOT_MODE: side_points},
    )
    ax_fit.set_title("Specialized Boundary Fit")

    if OUTPUT_PATH is not None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
        print(f"Saved figure to: {OUTPUT_PATH.resolve()}")

    if _has_interactive_display():
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
