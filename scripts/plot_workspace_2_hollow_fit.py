"""
Sampling + edge-fitting workspace plotter for large-bending hollow shells.

Compared with ``plot_workspace_2.py``, this variant still starts from directly
sampled side-view points, but the profile extraction is changed:

1. Rasterize sampled points into an R-Z occupancy grid
2. Smooth and threshold the occupancy field
3. For each Z row, detect contiguous radial occupied runs
4. Use the outermost run to define the shell:
   - outer contour = right edge of the outermost run
   - inner contour = left edge of the outermost run when that run does not
     touch the axis, or when there are multiple radial runs in the row

This is intended for cases where the manipulator bends past the centerline and
the side-view workspace should be interpreted as a hollow shell instead of a
simple solid body with a single ``r_min / r_max`` envelope.
"""
from __future__ import annotations

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

from plot_workspace_2 import (
    CONFIG_NAME,
    CONFIG_PATH,
    FIGSIZE,
    MODE_COLORS,
    MODE_LABELS,
    PROFILE_CURVE_SMOOTH_SIGMA,
    PROFILE_R_BINS,
    PROFILE_Z_BINS,
    REVOLVE_SAMPLES,
    SHOW_AXES,
    SIDE_REAL_VIEW_AZIM,
    SIDE_REAL_VIEW_ELEV,
    UNREACHABLE_ALPHA,
    UNREACHABLE_COLOR,
    _has_interactive_display,
    plot_revolved_profile,
    sample_mode_side_points,
)
from csm.model import CSM


PLOT_MODE = 3
OUTPUT_PATH = Path("./data/plot_workspace_2_hollow_fit.png")

# Sampling resolution follows the original script's philosophy.
SAMPLE_CFG = {"length": 80, "theta1": 80, "theta2": 120}

# Occupancy-grid extraction tuned for large-bending hollow shells.
GRID_Z_BINS = 260
GRID_R_BINS = 220
GRID_SIGMA_Z = 1.25
GRID_SIGMA_R = 1.0
GLOBAL_OCC_THRESHOLD = 0.06
ROW_OCC_THRESHOLD = 0.18
MIN_ROW_RUN_BINS = 3
MIN_INNER_VALID_ROWS = 8
CURVE_UPSAMPLE_FACTOR = 5


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


def _keep_longest_run(mask: np.ndarray, weights: np.ndarray | None = None, min_len: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    runs = _find_true_runs(mask)
    if not runs:
        return np.zeros_like(mask, dtype=bool)

    best = None
    best_score = -np.inf
    for start, end in runs:
        run_len = end - start + 1
        if run_len < min_len:
            continue
        if weights is None:
            score = float(run_len)
        else:
            score = float(np.sum(weights[start:end + 1]))
        if score > best_score:
            best_score = score
            best = (start, end)

    keep = np.zeros_like(mask, dtype=bool)
    if best is not None:
        keep[best[0]:best[1] + 1] = True
    return keep


def _smooth_curve(z_vals: np.ndarray, r_vals: np.ndarray, sigma: float = 1.2, upsample_factor: int = 4):
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


def _build_hollow_profile_from_side_points(side_points: np.ndarray):
    side_points = np.asarray(side_points, dtype=float)
    if side_points.ndim != 2 or side_points.shape[0] < 16 or side_points.shape[1] != 2:
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
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    hist, _, _ = np.histogram2d(z, r, bins=(z_edges, r_edges))
    if not np.any(hist > 0.0):
        return None

    occ = gaussian_filter(hist.astype(float), sigma=(GRID_SIGMA_Z, GRID_SIGMA_R), mode="nearest")
    global_max = float(np.max(occ))
    if global_max <= 0.0:
        return None

    outer_r = np.full(z_centers.shape, np.nan, dtype=float)
    inner_r = np.full(z_centers.shape, np.nan, dtype=float)
    run_count = np.zeros(z_centers.shape, dtype=int)

    for row_idx in range(occ.shape[0]):
        row = occ[row_idx]
        row_max = float(np.max(row))
        if row_max <= 0.0:
            continue

        row_threshold = max(GLOBAL_OCC_THRESHOLD * global_max, ROW_OCC_THRESHOLD * row_max)
        mask = row >= row_threshold
        if not np.any(mask):
            continue

        mask = binary_closing(mask[np.newaxis, :], structure=np.ones((1, 3), dtype=bool))[0]
        runs = [(start, end) for start, end in _find_true_runs(mask) if end - start + 1 >= MIN_ROW_RUN_BINS]
        if not runs:
            continue

        run_count[row_idx] = len(runs)
        outer_start, outer_end = runs[-1]
        outer_r[row_idx] = float(r_edges[outer_end + 1])

        touches_axis = outer_start <= 1
        if len(runs) >= 2 or not touches_axis:
            inner_r[row_idx] = float(r_edges[outer_start])

    valid_outer = np.isfinite(outer_r)
    if np.count_nonzero(valid_outer) < 6:
        return None

    outer_keep = _keep_longest_run(valid_outer, weights=np.nan_to_num(outer_r, nan=0.0), min_len=6)
    valid_outer &= outer_keep
    outer_z = z_centers[valid_outer]
    outer_r_vals = outer_r[valid_outer]
    outer_z, outer_r_vals = _smooth_curve(
        outer_z,
        outer_r_vals,
        sigma=max(1.0, 0.75 * PROFILE_CURVE_SMOOTH_SIGMA),
        upsample_factor=CURVE_UPSAMPLE_FACTOR,
    )

    valid_inner = np.isfinite(inner_r)
    if np.count_nonzero(valid_inner) >= MIN_INNER_VALID_ROWS:
        inner_weights = np.nan_to_num(inner_r, nan=0.0) * np.maximum(run_count, 1)
        inner_keep = _keep_longest_run(valid_inner, weights=inner_weights, min_len=MIN_INNER_VALID_ROWS)
        valid_inner &= inner_keep
    else:
        valid_inner[:] = False

    inner_z = None
    inner_r_vals = None
    if np.count_nonzero(valid_inner) >= MIN_INNER_VALID_ROWS:
        inner_z = z_centers[valid_inner]
        inner_r_vals = inner_r[valid_inner]
        inner_z, inner_r_vals = _smooth_curve(
            inner_z,
            inner_r_vals,
            sigma=max(0.8, 0.65 * PROFILE_CURVE_SMOOTH_SIGMA),
            upsample_factor=CURVE_UPSAMPLE_FACTOR,
        )
        inner_r_vals = np.clip(inner_r_vals, 0.0, None)
        outer_on_inner = np.interp(inner_z, outer_z, outer_r_vals)
        inner_r_vals = np.minimum(inner_r_vals, np.maximum(outer_on_inner - 1e-6, 0.0))
        inner_r_vals[inner_r_vals < (r_max / max(GRID_R_BINS, 1))] = 0.0
        keep = inner_r_vals > 0.0
        if np.count_nonzero(keep) >= MIN_INNER_VALID_ROWS:
            inner_z = inner_z[keep]
            inner_r_vals = inner_r_vals[keep]
        else:
            inner_z = None
            inner_r_vals = None

    debug = {
        "z_centers": z_centers,
        "r_centers": r_centers,
        "hist": hist,
        "occ": occ,
        "row_run_count": run_count,
        "row_outer_r": outer_r,
        "row_inner_r": inner_r,
    }
    return {
        "outer_z": outer_z,
        "outer_r": outer_r_vals,
        "inner_z": inner_z,
        "inner_r": inner_r_vals,
        "debug": debug,
    }


def _draw_side_view(ax, profile: dict, color: str, label: str):
    outer_z = np.asarray(profile["outer_z"], dtype=float) * 1000.0
    outer_r = np.asarray(profile["outer_r"], dtype=float) * 1000.0
    ax.fill_betweenx(outer_z, -outer_r, outer_r, color=color, alpha=0.65, linewidth=0.0, label=label)
    ax.plot(outer_r, outer_z, color=color, linewidth=1.6, alpha=0.95)
    ax.plot(-outer_r, outer_z, color=color, linewidth=1.6, alpha=0.95)

    inner_z = profile.get("inner_z")
    inner_r = profile.get("inner_r")
    if inner_z is not None and inner_r is not None:
        inner_z_mm = np.asarray(inner_z, dtype=float) * 1000.0
        inner_r_mm = np.asarray(inner_r, dtype=float) * 1000.0
        ax.fill_betweenx(
            inner_z_mm,
            -inner_r_mm,
            inner_r_mm,
            color="white",
            alpha=1.0,
            linewidth=0.0,
            zorder=3,
        )
        ax.plot(inner_r_mm, inner_z_mm, color=UNREACHABLE_COLOR, linewidth=1.2, alpha=0.95, zorder=4)
        ax.plot(-inner_r_mm, inner_z_mm, color=UNREACHABLE_COLOR, linewidth=1.2, alpha=0.95, zorder=4)

    ax.set_xlabel("R (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _configure_3d_axes(ax, profile: dict):
    outer_r = np.asarray(profile["outer_r"], dtype=float) * 1000.0
    outer_z = np.asarray(profile["outer_z"], dtype=float) * 1000.0
    curves = [(outer_r, outer_z)]

    inner_z = profile.get("inner_z")
    inner_r = profile.get("inner_r")
    if inner_z is not None and inner_r is not None:
        curves.append((np.asarray(inner_r, dtype=float) * 1000.0, np.asarray(inner_z, dtype=float) * 1000.0))

    all_r = np.concatenate([curve_r for curve_r, _ in curves])
    all_z = np.concatenate([curve_z for _, curve_z in curves])
    r_max = float(np.max(all_r))
    z_min = float(np.min(all_z))
    z_max = float(np.max(all_z))
    margin_r = max(2.0, 0.08 * r_max)
    margin_z = max(2.0, 0.08 * max(z_max - z_min, 1.0))

    ax.set_xlim(-(r_max + margin_r), r_max + margin_r)
    ax.set_ylim(-(r_max + margin_r), r_max + margin_r)
    ax.set_zlim(z_min - margin_z, z_max + margin_z)
    ax.set_box_aspect((1.0, 1.0, max(z_max - z_min, 1e-6) / max(2.0 * r_max, 1e-6)))
    ax.view_init(elev=SIDE_REAL_VIEW_ELEV + 18, azim=SIDE_REAL_VIEW_AZIM + 48)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.grid(True, alpha=0.25)
    if not SHOW_AXES:
        ax.set_axis_off()


def _draw_debug_rows(ax, profile: dict):
    debug = profile["debug"]
    z = np.asarray(debug["z_centers"], dtype=float) * 1000.0
    outer_r = np.asarray(debug["row_outer_r"], dtype=float) * 1000.0
    inner_r = np.asarray(debug["row_inner_r"], dtype=float) * 1000.0
    row_runs = np.asarray(debug["row_run_count"], dtype=int)

    valid_outer = np.isfinite(outer_r)
    valid_inner = np.isfinite(inner_r)
    if np.any(valid_outer):
        ax.scatter(outer_r[valid_outer], z[valid_outer], s=10, color="#1F5D78", alpha=0.55, label="row outer edge")
        ax.scatter(-outer_r[valid_outer], z[valid_outer], s=10, color="#1F5D78", alpha=0.55)
    if np.any(valid_inner):
        colors = np.where(row_runs[valid_inner] >= 2, "#8C5A2F", "#D79A6B")
        ax.scatter(inner_r[valid_inner], z[valid_inner], s=10, c=colors, alpha=0.65, label="row inner edge")
        ax.scatter(-inner_r[valid_inner], z[valid_inner], s=10, c=colors, alpha=0.65)


def main():
    print(f"Loading CSM config: {CONFIG_PATH}")
    csm = CSM.from_config(CONFIG_PATH)

    print(f"Sampling side-view points for mode {PLOT_MODE} using direct sampling...")
    side_points = sample_mode_side_points(csm, PLOT_MODE, SAMPLE_CFG)
    profile = _build_hollow_profile_from_side_points(side_points)
    if profile is None:
        raise RuntimeError("Failed to build hollow-shell profile from sampled side-view points.")

    color = MODE_COLORS.get(PLOT_MODE, "#9FD4EA")
    label = MODE_LABELS.get(PLOT_MODE, f"Mode {PLOT_MODE}")

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.22)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    outer_z_mm = np.asarray(profile["outer_z"], dtype=float) * 1000.0
    outer_r_mm = np.asarray(profile["outer_r"], dtype=float) * 1000.0
    plot_revolved_profile(ax3d, outer_z_mm, outer_r_mm, color=color, alpha=0.34, label=label)

    inner_z = profile.get("inner_z")
    inner_r = profile.get("inner_r")
    if inner_z is not None and inner_r is not None:
        plot_revolved_profile(
            ax3d,
            np.asarray(inner_z, dtype=float) * 1000.0,
            np.asarray(inner_r, dtype=float) * 1000.0,
            color=UNREACHABLE_COLOR,
            alpha=UNREACHABLE_ALPHA,
            label="Inner shell",
        )
    _configure_3d_axes(ax3d, profile)
    ax3d.set_title("Hollow Shell Workspace")
    ax3d.legend(loc="upper right")

    ax_side = fig.add_subplot(gs[0, 1])
    _draw_side_view(ax_side, profile, color=color, label=label)
    ax_side.set_title("Side View")

    ax_debug = fig.add_subplot(gs[1, 1])
    _draw_side_view(ax_debug, profile, color=color, label=label)
    _draw_debug_rows(ax_debug, profile)
    ax_debug.scatter(side_points[:, 0] * 1000.0, side_points[:, 1] * 1000.0, s=4, alpha=0.10, color=color)
    ax_debug.scatter(-side_points[:, 0] * 1000.0, side_points[:, 1] * 1000.0, s=4, alpha=0.10, color=color)
    ax_debug.set_title("Sample Scatter + Fitted Edges")

    fig.suptitle(f"Sampling + Hollow-Shell Fit ({CONFIG_NAME}, mode {PLOT_MODE})", fontsize=14)

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
