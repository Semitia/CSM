"""
Module: plot_workspace_boundary_scan.py
Description: Build workspace profiles by directly scanning boundary motions instead
of extracting contours from dense point clouds.

Current status:
- mode 1 implemented
- mode 2/3/4/0 reserved in the framework
"""
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from csm import CSM


CONFIG_NAME = "csm_cfg_0_tool.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
PLOT_MODES = [1]

FIGSIZE = (11.5, 6.5)
REVOLVE_SAMPLES = 30
REVOLVE_MAX_AXIAL_SAMPLES = 96
MODE_COLORS = {
    1: "#9FD4EA",
    2: "#F0E0AA",
    3: "#E9A3A7",
    4: "#CFCFCF",
}
OUTPUT_PATH = Path("./data/plot_workspace_boundary_scan.png")
SHOW_FIGURE = True


def _has_interactive_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@dataclass
class WorkspaceProfile:
    mode: int
    inner_segments: list
    outer_segments: list
    closed_profile_rz: np.ndarray


def _tool_point_from_state(csm, *, mode, L1, L2, Lr, Ls, theta_1, theta_2):
    csm.set_state(
        mode=mode,
        phi=0.0,
        L1=float(L1),
        L2=float(L2),
        Lr=float(Lr),
        Ls=float(Ls),
        theta_1=float(theta_1),
        theta_2=float(theta_2),
        delta_1=0.0,
        delta_2=0.0,
    )
    return np.asarray(csm.pose[:3], dtype=float)


def _sample_state_curve(csm, states):
    points = np.asarray([_tool_point_from_state(csm, **state) for state in states], dtype=float)
    radii = np.hypot(points[:, 0], points[:, 1])
    return np.column_stack((radii, points[:, 2]))


def _build_axis_closure(start_rz, end_rz, samples=48):
    radii = np.linspace(float(start_rz[0]), float(end_rz[0]), samples)
    heights = np.linspace(float(start_rz[1]), float(end_rz[1]), samples)
    return np.column_stack((radii, heights))


def build_mode1_profile(csm, length_samples=240, angle_samples=240):
    inner_states = []
    for L2 in np.linspace(0.0, csm.L_20, length_samples):
        inner_states.append(
            {
                "mode": 1,
                "L1": 0.0,
                "L2": L2,
                "Lr": 0.0,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": min(csm.kappa_20 * L2, csm.theta2_limit),
            }
        )

    outer_states = []
    for theta_2 in np.linspace(csm.theta2_limit, 0.0, angle_samples):
        outer_states.append(
            {
                "mode": 1,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": 0.0,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        )

    inner_curve = _sample_state_curve(csm, inner_states)
    outer_curve = _sample_state_curve(csm, outer_states)
    axis_closure = _build_axis_closure(outer_curve[-1], inner_curve[0])
    closed_profile = np.vstack((inner_curve, outer_curve[1:], axis_closure[1:]))

    return WorkspaceProfile(
        mode=1,
        inner_segments=[inner_curve],
        outer_segments=[outer_curve],
        closed_profile_rz=closed_profile,
    )


def build_workspace_profile(csm, mode):
    if mode == 1:
        return build_mode1_profile(csm)
    raise NotImplementedError(f"Mode {mode} is not implemented yet in boundary-scan plotting.")


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


def _decimate_curve(curve_rz, max_samples=REVOLVE_MAX_AXIAL_SAMPLES):
    curve_rz = np.asarray(curve_rz, dtype=float)
    if curve_rz.shape[0] <= max_samples:
        return curve_rz
    sample_idx = np.linspace(0, curve_rz.shape[0] - 1, max_samples).astype(int)
    sample_idx = np.unique(sample_idx)
    return curve_rz[sample_idx]


def draw_revolved_profile(ax, profile, color):
    profile_rz = _decimate_curve(profile.closed_profile_rz)
    theta = np.linspace(0.0, 2.0 * np.pi, REVOLVE_SAMPLES, endpoint=False)
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
        alpha=0.38,
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

def draw_side_profile(ax, profile, color, label):
    rz = profile.closed_profile_rz * 1000.0
    mirrored_rz = np.column_stack((-rz[:, 0], rz[:, 1]))
    symmetric_outline = np.vstack((mirrored_rz[::-1], rz[1:]))
    ax.fill(
        symmetric_outline[:, 0],
        symmetric_outline[:, 1],
        color=color,
        alpha=0.30,
        linewidth=0.0,
        label=label,
    )

    for idx, curve in enumerate(profile.inner_segments):
        curve_mm = curve * 1000.0
        ax.plot(
            curve_mm[:, 0],
            curve_mm[:, 1],
            color="#C96868",
            linewidth=2.0,
            label="Inner contour" if idx == 0 else None,
        )
        ax.plot(
            -curve_mm[:, 0],
            curve_mm[:, 1],
            color="#C96868",
            linewidth=2.0,
        )

    for idx, curve in enumerate(profile.outer_segments):
        curve_mm = curve * 1000.0
        ax.plot(
            curve_mm[:, 0],
            curve_mm[:, 1],
            color="#2D5F73",
            linewidth=2.0,
            label="Outer contour" if idx == 0 else None,
        )
        ax.plot(
            -curve_mm[:, 0],
            curve_mm[:, 1],
            color="#2D5F73",
            linewidth=2.0,
        )


def _configure_3d_axes(ax, profiles):
    all_r = np.concatenate([profile.closed_profile_rz[:, 0] for profile in profiles]) * 1000.0
    all_z = np.concatenate([profile.closed_profile_rz[:, 1] for profile in profiles]) * 1000.0
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


def _configure_side_axes(ax, profiles):
    all_r = np.concatenate([profile.closed_profile_rz[:, 0] for profile in profiles]) * 1000.0
    all_z = np.concatenate([profile.closed_profile_rz[:, 1] for profile in profiles]) * 1000.0
    r_max = float(np.max(all_r)) if all_r.size else 1.0
    z_min = float(np.min(all_z)) if all_z.size else 0.0
    z_max = float(np.max(all_z)) if all_z.size else 1.0

    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_xlim(-(r_max * 1.08 if r_max > 0 else 1.0), r_max * 1.08 if r_max > 0 else 1.0)
    ax.set_ylim(z_min - 1.0, z_max + 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def main():
    csm = CSM.from_config(CONFIG_PATH)
    profiles = [build_workspace_profile(csm, mode) for mode in PLOT_MODES]

    fig = plt.figure(figsize=FIGSIZE)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    for profile in profiles:
        color = MODE_COLORS.get(profile.mode, "#9FD4EA")
        draw_revolved_profile(ax3d, profile, color)
        draw_side_profile(ax2d, profile, color, label=f"Mode {profile.mode}")

    _configure_3d_axes(ax3d, profiles)
    _configure_side_axes(ax2d, profiles)
    ax3d.set_title("Workspace Boundary Scan")
    ax2d.set_title("Side Profile")

    handles, labels = ax2d.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax2d.legend(unique.values(), unique.keys(), loc="best", frameon=False)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH.resolve()}")

    if SHOW_FIGURE and _has_interactive_display():
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
