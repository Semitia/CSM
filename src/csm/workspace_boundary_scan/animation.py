"""
Animation helpers for workspace boundary scan profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from .core import WorkspaceAnimationData, WorkspaceAnimationCurve
from ..model import CSM
from .plotting import has_interactive_display


@dataclass
class BoundaryScanAnimationOptions:
    enabled: bool = False
    output_path: Path | None = None
    fps: int = 12
    dpi: int = 180
    show_figure: bool = False
    save_debug_frames: bool = False
    figsize: tuple[float, float] | None = None


def _curve_prefix(points_rz, progress):
    points_rz = np.asarray(points_rz, dtype=float)
    if points_rz.ndim != 2 or points_rz.shape[0] == 0:
        return points_rz.copy()
    if progress <= 0.0:
        return points_rz[:1].copy()
    if progress >= 1.0 or points_rz.shape[0] == 1:
        return points_rz.copy()

    scaled = progress * (points_rz.shape[0] - 1)
    idx = int(np.floor(scaled))
    frac = scaled - idx
    prefix = points_rz[: idx + 1].copy()
    if idx < points_rz.shape[0] - 1:
        point = points_rz[idx] + frac * (points_rz[idx + 1] - points_rz[idx])
        if prefix.shape[0] == 0:
            prefix = point[np.newaxis, :]
        elif not np.allclose(prefix[-1], point, atol=1e-9):
            prefix = np.vstack((prefix, point))
        else:
            prefix[-1] = point
    return prefix


def _plot_animation_curve(ax, curve: WorkspaceAnimationCurve, progress: float):
    points_rz = curve.points_rz if not curve.progressive else _curve_prefix(curve.points_rz, progress)
    points_rz = np.asarray(points_rz, dtype=float)
    if points_rz.ndim != 2 or points_rz.shape[0] == 0:
        return

    points_mm = points_rz * 1000.0
    color = curve.color or "#1F2937"
    if curve.marker is not None:
        ax.scatter(
            points_mm[:, 0],
            points_mm[:, 1],
            s=curve.markersize ** 2,
            color=color,
            alpha=curve.alpha,
            label=curve.label,
        )
        if curve.mirror:
            ax.scatter(
                -points_mm[:, 0],
                points_mm[:, 1],
                s=curve.markersize ** 2,
                color=color,
                alpha=curve.alpha,
            )
        return

    ax.plot(
        points_mm[:, 0],
        points_mm[:, 1],
        color=color,
        linestyle=curve.linestyle,
        linewidth=curve.linewidth,
        alpha=curve.alpha,
        label=curve.label,
    )
    if curve.mirror:
        ax.plot(
            -points_mm[:, 0],
            points_mm[:, 1],
            color=color,
            linestyle=curve.linestyle,
            linewidth=curve.linewidth,
            alpha=curve.alpha,
        )


def _clone_csm_from_spec(csm_spec):
    if csm_spec is None:
        return None
    return CSM(**csm_spec)


def _state_for_progress(state_sequence, progress):
    if not state_sequence:
        return None
    if len(state_sequence) == 1:
        return dict(state_sequence[0])
    idx = int(np.clip(round(progress * (len(state_sequence) - 1)), 0, len(state_sequence) - 1))
    return dict(state_sequence[idx])


def _plot_arm_projection(ax, csm: CSM, color="#111111"):
    geometry = csm.get_visualization_segments()
    drawn = False
    for segment in geometry["segments"]:
        points = np.asarray(segment["points"], dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            continue
        radius_mm = np.hypot(points[:, 0], points[:, 1]) * 1000.0
        z_mm = points[:, 2] * 1000.0
        ax.plot(radius_mm, z_mm, color=color, linewidth=2.0, alpha=0.95, zorder=7)
        drawn = True

    tool = geometry["tool"]
    tool_points = np.vstack((tool["start"], tool["end"]))
    tool_radius_mm = np.hypot(tool_points[:, 0], tool_points[:, 1]) * 1000.0
    tool_z_mm = tool_points[:, 2] * 1000.0
    ax.plot(tool_radius_mm, tool_z_mm, color="#2563EB", linewidth=2.2, alpha=0.95, zorder=8)
    ax.scatter([tool_radius_mm[-1]], [tool_z_mm[-1]], color="#2563EB", s=20, zorder=9)
    return drawn


def _format_state_text(csm: CSM):
    tip_r = float(np.hypot(csm.pose[0], csm.pose[1]) * 1000.0)
    tip_z = float(csm.pose[2] * 1000.0)
    return (
        f"mode={csm.mode}  tip=({tip_r:.1f}, {tip_z:.1f}) mm\n"
        f"L1={csm.L1*1000.0:.1f}  L2={csm.L2*1000.0:.1f}  "
        f"Lr={csm.Lr*1000.0:.1f}  Ls={csm.Ls*1000.0:.1f} mm\n"
        f"theta1={csm.theta_1:.3f}  theta2={csm.theta_2:.3f} rad"
    )


def _collect_profile_side_extents(profile):
    curves = [np.asarray(profile.outer_open_curve_rz, dtype=float)]
    curves.extend(np.asarray(curve, dtype=float) for curve in profile.unreachable_closed_profiles_rz)
    curves = [curve for curve in curves if curve.ndim == 2 and curve.shape[0] > 0]
    if not curves:
        return 1.0, 0.0, 1.0
    all_r = np.concatenate([curve[:, 0] for curve in curves]) * 1000.0
    all_z = np.concatenate([curve[:, 1] for curve in curves]) * 1000.0
    return float(np.max(all_r)), float(np.min(all_z)), float(np.max(all_z))


def _collect_arm_side_extents(csm: CSM, state_sequence):
    if csm is None or not state_sequence:
        return None
    all_r = []
    all_z = []
    for state in state_sequence:
        csm.set_state(**state)
        geometry = csm.get_visualization_segments()
        for segment in geometry["segments"]:
            points = np.asarray(segment["points"], dtype=float)
            if points.ndim != 2 or points.shape[0] == 0:
                continue
            all_r.append(np.hypot(points[:, 0], points[:, 1]) * 1000.0)
            all_z.append(points[:, 2] * 1000.0)
        tool_points = np.vstack((geometry["tool"]["start"], geometry["tool"]["end"]))
        all_r.append(np.hypot(tool_points[:, 0], tool_points[:, 1]) * 1000.0)
        all_z.append(tool_points[:, 2] * 1000.0)
    if not all_r:
        return None
    return (
        float(np.max(np.concatenate(all_r))),
        float(np.min(np.concatenate(all_z))),
        float(np.max(np.concatenate(all_z))),
    )


def _compute_side_limits(profile, mode_animation, csm):
    r_max, z_min, z_max = _collect_profile_side_extents(profile)
    for stage in mode_animation.stages:
        arm_extents = _collect_arm_side_extents(csm, stage.state_sequence)
        if arm_extents is None:
            continue
        arm_r_max, arm_z_min, arm_z_max = arm_extents
        r_max = max(r_max, arm_r_max)
        z_min = min(z_min, arm_z_min)
        z_max = max(z_max, arm_z_max)

    margin_r = max(2.0, 0.08 * r_max)
    span_z = z_max - z_min if z_max > z_min else 1.0
    margin_z = max(2.0, 0.08 * span_z)
    return (
        -(r_max + margin_r),
        r_max + margin_r,
        z_min - margin_z,
        z_max + margin_z,
    )


def _apply_side_limits(ax, limits):
    x_min, x_max, y_min, y_max = limits
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _mode_total_frames(animation_data: WorkspaceAnimationData):
    return max(1, sum(max(1, stage.frames) for stage in animation_data.stages))


def _locate_stage(animation_data: WorkspaceAnimationData, frame_idx: int):
    if not animation_data.stages:
        return None, 1.0
    cursor = 0
    for stage in animation_data.stages:
        stage_frames = max(1, stage.frames)
        end = cursor + stage_frames
        if frame_idx < end:
            local = frame_idx - cursor
            progress = 1.0 if stage_frames == 1 else local / float(stage_frames - 1)
            return stage, progress
        cursor = end
    return animation_data.stages[-1], 1.0


def _debug_frame_dir(options: BoundaryScanAnimationOptions):
    if options.output_path is None:
        return Path("./data/workspace_boundary_scan_animation_frames")
    output_path = Path(options.output_path)
    return output_path.parent / f"{output_path.stem}_frames"


def _require_pillow_writer():
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError(
            "GIF export requires Pillow. Install it with `pip install pillow` or add it to the project environment."
        )


def render_workspace_scan_animation(
    profiles,
    animation_data,
    options: BoundaryScanAnimationOptions | None = None,
):
    options = options or BoundaryScanAnimationOptions()
    if not options.enabled:
        return None
    if not options.show_figure and options.output_path is None:
        raise ValueError("Animation rendering needs `output_path` when `show_figure` is False.")
    if len(profiles) != len(animation_data):
        raise ValueError("`profiles` and `animation_data` must have the same length.")

    num_modes = max(len(profiles), 1)
    figsize = options.figsize if options.figsize is not None else (8.6, 4.4 * num_modes)
    fig, axes = plt.subplots(num_modes, 1, figsize=figsize, squeeze=False)
    axes = axes[:, 0].tolist()
    csm_instances = [_clone_csm_from_spec(item.csm_spec) for item in animation_data]
    side_limits = [
        _compute_side_limits(profile, mode_animation, csm)
        for profile, mode_animation, csm in zip(profiles, animation_data, csm_instances)
    ]

    total_frames = max(_mode_total_frames(item) for item in animation_data) if animation_data else 1
    debug_dir = _debug_frame_dir(options) if options.save_debug_frames else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    def _draw_frame(frame_idx):
        for ax, profile, mode_animation, csm, limits in zip(axes, profiles, animation_data, csm_instances, side_limits):
            ax.clear()
            _apply_side_limits(ax, limits)
            stage, progress = _locate_stage(mode_animation, min(frame_idx, _mode_total_frames(mode_animation) - 1))
            if stage is None:
                ax.set_title(f"Mode {profile.mode} Boundary Animation")
                continue
            for curve in stage.curves:
                _plot_animation_curve(ax, curve, progress)
            current_state = _state_for_progress(stage.state_sequence, progress)
            if current_state is not None and csm is not None:
                csm.set_state(**current_state)
                _plot_arm_projection(ax, csm)
                ax.text(
                    0.99,
                    0.98,
                    _format_state_text(csm),
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=9,
                    family="monospace",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#D1D5DB"},
                )
            ax.set_title(stage.title)
            if stage.annotation:
                ax.text(
                    0.01,
                    0.98,
                    stage.annotation,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "#D1D5DB"},
                )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels, handles))
                ax.legend(
                    unique.values(),
                    unique.keys(),
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=False,
                )

        fig.tight_layout()
        if debug_dir is not None:
            fig.savefig(debug_dir / f"frame_{frame_idx:04d}.png", dpi=options.dpi, bbox_inches="tight")
        return []

    animation = FuncAnimation(
        fig,
        _draw_frame,
        frames=total_frames,
        interval=1000.0 / max(options.fps, 1),
        blit=False,
        repeat=False,
    )

    if options.output_path is not None:
        _require_pillow_writer()
        output_path = Path(options.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=options.fps)
        animation.save(output_path, writer=writer, dpi=options.dpi)
    if options.show_figure and has_interactive_display():
        plt.show()
    plt.close(fig)
    return animation
