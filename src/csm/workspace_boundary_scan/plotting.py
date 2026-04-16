"""
Plotting helpers for workspace boundary scan profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import symmetric_fill_polygon


DEFAULT_MODE_COLORS = {
    1: "#9FD4EA",
    2: "#F0E0AA",
    3: "#E9A3A7",
    4: "#CFCFCF",
    0: "#9FD4EA",
}


@dataclass
class BoundaryScanPlotOptions:
    figsize: tuple[float, float] | None = None
    revolve_samples: int = 30
    revolve_max_axial_samples: int = 96
    render_3d_mode: str = "fast_surface"
    show_figure: bool = True
    save_debug_figures: bool = True
    output_path: Path | None = None
    debug_output_dir: Path | None = None
    reach_alpha: float = 0.34
    unreachable_alpha: float = 0.40
    mode_colors: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_MODE_COLORS))
    unreachable_color: str = "#D79A6B"
    outer_contour_color: str = "#2D5F73"


def has_interactive_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _build_revolve_faces(num_axial, num_theta):
    faces = []
    for i in range(num_axial - 1):
        row = i * num_theta
        next_row = (i + 1) * num_theta
        for j in range(num_theta):
            jn = (j + 1) % num_theta
            a = row + j
            b = row + jn
            c = next_row + j
            d = next_row + jn
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(faces, dtype=int)


def _decimate_curve(curve_rz, max_samples):
    curve_rz = np.asarray(curve_rz, dtype=float)
    if curve_rz.shape[0] <= max_samples:
        return curve_rz
    sample_idx = np.linspace(0, curve_rz.shape[0] - 1, max_samples).astype(int)
    sample_idx = np.unique(sample_idx)
    return curve_rz[sample_idx]


def _profile_mesh(profile_rz, revolve_samples):
    phi = np.linspace(0.0, 2.0 * np.pi, revolve_samples)
    radii = profile_rz[:, 0][:, None] * 1000.0
    heights = profile_rz[:, 1][:, None] * 1000.0
    x = radii * np.cos(phi)[None, :]
    y = radii * np.sin(phi)[None, :]
    z = np.repeat(heights, phi.size, axis=1)
    return x, y, z


def draw_revolved_shell(ax, profile_rz, color, alpha, options: BoundaryScanPlotOptions):
    profile_rz = _decimate_curve(profile_rz, max_samples=options.revolve_max_axial_samples)
    if options.render_3d_mode == "fast_surface":
        x, y, z = _profile_mesh(profile_rz, options.revolve_samples)
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=alpha,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
        return

    theta = np.linspace(0.0, 2.0 * np.pi, options.revolve_samples, endpoint=False)
    theta_grid, z_grid = np.meshgrid(theta, profile_rz[:, 1] * 1000.0)
    r_grid = np.repeat((profile_rz[:, 0] * 1000.0)[:, None], theta.size, axis=1)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    verts = np.column_stack((x.ravel(), y.ravel(), z_grid.ravel()))
    faces = _build_revolve_faces(len(profile_rz), theta.size)
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


def draw_revolved_profile(ax, profile, color, options: BoundaryScanPlotOptions):
    draw_revolved_shell(ax, profile.outer_open_curve_rz, color=color, alpha=options.reach_alpha, options=options)
    for unreachable_profile in profile.unreachable_open_curves_rz:
        draw_revolved_shell(ax, unreachable_profile, color=options.unreachable_color, alpha=options.unreachable_alpha, options=options)


def draw_side_profile(ax, profile, color, label, options: BoundaryScanPlotOptions):
    symmetric_outline = symmetric_fill_polygon(profile.outer_open_curve_rz) * 1000.0
    ax.fill(
        symmetric_outline[:, 0],
        symmetric_outline[:, 1],
        color=color,
        alpha=0.30,
        linewidth=0.0,
        label=label,
    )
    for idx, unreachable_profile in enumerate(profile.unreachable_closed_profiles_rz):
        symmetric_void = symmetric_fill_polygon(unreachable_profile) * 1000.0
        ax.fill(
            symmetric_void[:, 0],
            symmetric_void[:, 1],
            color=options.unreachable_color,
            alpha=0.80,
            linewidth=0.0,
            label="Unreachable" if idx == 0 else None,
        )
    for idx, curve in enumerate(profile.inner_segments):
        curve_mm = np.asarray(curve, dtype=float) * 1000.0
        ax.plot(curve_mm[:, 0], curve_mm[:, 1], color=options.unreachable_color, linewidth=2.0, label="Inner contour" if idx == 0 else None)
        ax.plot(-curve_mm[:, 0], curve_mm[:, 1], color=options.unreachable_color, linewidth=2.0)
    for idx, curve in enumerate(profile.outer_segments):
        curve_mm = np.asarray(curve, dtype=float) * 1000.0
        ax.plot(curve_mm[:, 0], curve_mm[:, 1], color=options.outer_contour_color, linewidth=2.0, label="Outer contour" if idx == 0 else None)
        ax.plot(-curve_mm[:, 0], curve_mm[:, 1], color=options.outer_contour_color, linewidth=2.0)


def configure_3d_axes(ax, profiles):
    all_curves = []
    for profile in profiles:
        all_curves.append(profile.outer_open_curve_rz)
        all_curves.extend(profile.unreachable_open_curves_rz)
    all_r = np.concatenate([curve[:, 0] for curve in all_curves]) * 1000.0
    all_z = np.concatenate([curve[:, 1] for curve in all_curves]) * 1000.0
    r_max = float(np.max(all_r)) if all_r.size else 1.0
    z_min = float(np.min(all_z)) if all_z.size else 0.0
    z_max = float(np.max(all_z)) if all_z.size else 1.0
    margin_r = max(2.0, 0.08 * r_max)
    margin_z = max(2.0, 0.08 * (z_max - z_min if z_max > z_min else 1.0))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(-(r_max + margin_r), r_max + margin_r)
    ax.set_ylim(-(r_max + margin_r), r_max + margin_r)
    ax.set_zlim(z_min - margin_z, z_max + margin_z)
    ax.set_box_aspect((1.0, 1.0, max(z_max - z_min, 1e-6) / max(2.0 * r_max, 1e-6)))
    ax.view_init(elev=18, azim=-42)
    ax.grid(True, alpha=0.25)


def configure_side_axes(ax, profiles):
    all_curves = []
    for profile in profiles:
        all_curves.append(profile.outer_open_curve_rz)
        all_curves.extend(profile.unreachable_closed_profiles_rz)
    all_r = np.concatenate([curve[:, 0] for curve in all_curves]) * 1000.0
    all_z = np.concatenate([curve[:, 1] for curve in all_curves]) * 1000.0
    r_max = float(np.max(all_r)) if all_r.size else 1.0
    z_min = float(np.min(all_z)) if all_z.size else 0.0
    z_max = float(np.max(all_z)) if all_z.size else 1.0
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_xlim(-(r_max * 1.08 if r_max > 0 else 1.0), r_max * 1.08 if r_max > 0 else 1.0)
    ax.set_ylim(z_min - 1.0, z_max + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _plot_debug_curve(ax, curve_rz, label, color, linewidth=1.6, linestyle="-", alpha=1.0):
    curve_mm = np.asarray(curve_rz, dtype=float) * 1000.0
    ax.plot(curve_mm[:, 0], curve_mm[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
    ax.plot(-curve_mm[:, 0], curve_mm[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def save_profile_debug_figure(profile, options: BoundaryScanPlotOptions):
    if not options.save_debug_figures or profile.debug_data is None or options.debug_output_dir is None:
        return

    debug_output_dir = Path(options.debug_output_dir)
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    debug = profile.debug_data
    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    primitives = debug.get("primitives", {})
    color_map = {"tau0": "#C96868", "tau1": "#7C5CFC", "tau2": "#2E8B57"}
    for name, curve in primitives.items():
        _plot_debug_curve(ax, curve, label=name, color=color_map.get(name, "#444444"), linewidth=1.4, alpha=0.9)

    for entry in debug.get("all_curves", []):
        if len(entry) != 3:
            continue
        name, role, curve = entry
        _plot_debug_curve(ax, curve, label=name, color="#8A8A8A" if role == "outer" else "#C28C62", linewidth=1.2, linestyle="--", alpha=0.55)

    for entry in debug.get("candidate_segments", []):
        if len(entry) != 3:
            continue
        name, role, curve = entry
        _plot_debug_curve(ax, curve, label=name, color="#A0A0A0" if "outer" in role else "#B98B6A", linewidth=1.0, linestyle=":", alpha=0.4)

    for entry in debug.get("discarded_segments", []):
        if len(entry) != 3:
            continue
        name, role, curve = entry
        _plot_debug_curve(ax, curve, label=name, color="#666666" if "outer" in role else "#9A6A43", linewidth=2.0, linestyle="--", alpha=0.65)

    for idx, segment in enumerate(debug.get("outer_segments", [])):
        _plot_debug_curve(ax, segment, label=f"outer_seg_{idx+1}", color="#1F5D78", linewidth=2.2, alpha=0.95)
    for idx, segment in enumerate(debug.get("inner_segments", [])):
        _plot_debug_curve(ax, segment, label=f"inner_seg_{idx+1}", color=options.unreachable_color, linewidth=2.2, alpha=0.95)
    for idx, segment in enumerate(debug.get("trace_segments", [])):
        _plot_debug_curve(ax, segment, label=f"trace_seg_{idx+1}", color="#111111", linewidth=2.6, alpha=0.95)

    trace_path = debug.get("trace_path")
    if trace_path is not None:
        trace_mm = np.asarray(trace_path, dtype=float) * 1000.0
        ax.scatter(trace_mm[:, 0], trace_mm[:, 1], s=12, color="#111111", alpha=0.7, zorder=6, label="trace_path")
        ax.scatter(-trace_mm[:, 0], trace_mm[:, 1], s=12, color="#111111", alpha=0.7, zorder=6)

    for hit_key, color in (("hit_tau1", "#7C5CFC"), ("hit_tau2", "#2E8B57"), ("chosen_hit", "#111111")):
        hit = debug.get(hit_key)
        if hit is None:
            continue
        point = np.asarray(hit["point"], dtype=float) * 1000.0
        ax.scatter([point[0], -point[0]], [point[1], point[1]], s=50, color=color, zorder=5, label=hit_key)

    for hit_name, hit in debug.get("overlap_hits", {}).items():
        point = np.asarray(hit["point"], dtype=float) * 1000.0
        ax.scatter([point[0], -point[0]], [point[1], point[1]], s=40, color="#444444", zorder=5, label=hit_name)

    ax.set_title(f"Mode {profile.mode} Debug Primitives")
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Z [mm]")
    configure_side_axes(ax, [profile])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    debug_path = debug_output_dir / f"mode{profile.mode}_debug.png"
    fig.savefig(debug_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_workspace_profiles(profiles, options: BoundaryScanPlotOptions | None = None):
    options = options or BoundaryScanPlotOptions()
    n_modes = max(len(profiles), 1)
    figsize = options.figsize if options.figsize is not None else (11.5, 5.2 * n_modes)
    fig = plt.figure(figsize=figsize)

    for idx, profile in enumerate(profiles, start=1):
        ax3d = fig.add_subplot(n_modes, 2, 2 * idx - 1, projection="3d")
        ax2d = fig.add_subplot(n_modes, 2, 2 * idx)
        color = options.mode_colors.get(profile.mode, DEFAULT_MODE_COLORS.get(profile.mode, "#9FD4EA"))

        draw_revolved_profile(ax3d, profile, color, options)
        draw_side_profile(ax2d, profile, color, label=f"Mode {profile.mode}", options=options)

        configure_3d_axes(ax3d, [profile])
        configure_side_axes(ax2d, [profile])
        ax3d.set_title(f"Mode {profile.mode} Workspace")
        ax2d.set_title(f"Mode {profile.mode} Side View")

        handles, labels = ax2d.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax2d.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

        save_profile_debug_figure(profile, options)

    fig.tight_layout()
    if options.output_path is not None:
        output_path = Path(options.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, bbox_inches="tight")
    if options.show_figure and has_interactive_display():
        plt.show()
    return fig
