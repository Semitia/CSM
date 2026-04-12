"""
Render two mode3 dexterous-workspace panels for one box center and one box vertex.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from csm import (
    CSM,
    AnalyticDexterousWorkspace,
    DexterousMode3State,
    DexterousParameters,
    DexterousPlotOptions,
    OperationBox,
    analytic_fk_mode3,
    build_dexterous_probe,
    make_mode3_display_csm,
    draw_operation_box,
    fit_largest_centered_operation_box_in_mode3,
    operation_box_vertices,
    plot_dexterous_probe,
    scan_directions,
)
from csm.workspace_boundary_scan import BoundaryScanOptions, build_workspace_profiles


DEFAULT_OUTPUT = Path("data/example_mode3_dexterous_figure.png")
DEFAULT_BOX_SIZE_MM = np.array([50.0, 50.0, 40.0], dtype=float)
DEFAULT_TOP_MARGIN_MM = 4.0
SPHERE_DIAMETER_TO_BOX_MEAN_RATIO = 0.8


@dataclass(frozen=True)
class PanelSolution:
    point_xyz: np.ndarray
    direction_xyz: np.ndarray
    state: object
    method: str
    seed: int


def _parse_box_size_mm(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.shape != (3,):
        raise ValueError("Expected --box-size-mm as 'sx,sy,sz'.")
    return values


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _rotation_from_tool_axis(axis_z: np.ndarray) -> np.ndarray:
    axis_z = _normalize(axis_z)
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(helper, axis_z))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    axis_x = np.cross(helper, axis_z)
    axis_x = _normalize(axis_x)
    axis_y = np.cross(axis_z, axis_x)
    axis_y = _normalize(axis_y)
    return np.column_stack((axis_x, axis_y, axis_z))


def _sample_candidate_directions(seed: int, count: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, 3))
    dirs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    anchors = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return np.vstack((dirs, anchors))


def _validate_state(params: DexterousParameters, state, point_xyz: np.ndarray, direction_xyz: np.ndarray) -> bool:
    fk_point, fk_axis = analytic_fk_mode3(params, state)
    pos_err = float(np.linalg.norm(fk_point - point_xyz))
    ori_err = float(np.arccos(np.clip(float(np.dot(_normalize(fk_axis), _normalize(direction_xyz))), -1.0, 1.0)))
    return pos_err <= 2.0e-4 and ori_err <= np.deg2rad(2.0)


def _solve_panel_state(csm: CSM, point_xyz: np.ndarray, seed: int) -> PanelSolution:
    params = DexterousParameters.from_csm(csm)
    analytic = AnalyticDexterousWorkspace(params)
    theta2_grid = np.linspace(0.0, params.theta2_plus, 31)
    directions = _sample_candidate_directions(seed)

    for direction in directions:
        R_target = _rotation_from_tool_axis(direction)
        for theta2 in theta2_grid:
            state, success = analytic.solve_mode3_from_theta2(theta2, point_xyz, R_target)
            if not success or state is None:
                continue
            if _validate_state(params, state, point_xyz, direction):
                return PanelSolution(
                    point_xyz=np.asarray(point_xyz, dtype=float),
                    direction_xyz=_normalize(direction),
                    state=state,
                    method="analytic",
                    seed=seed,
                )

    fallback = scan_directions(csm, point_xyz, n_directions=180)
    reachable_idx = np.flatnonzero(fallback.reachable_mask)
    if reachable_idx.size == 0:
        raise RuntimeError("No reachable dexterous direction found for the selected point.")

    target_direction = directions[0]
    best_idx = int(reachable_idx[np.argmax(fallback.directions_world[reachable_idx] @ target_direction)])
    state = fallback.solved_states[best_idx]
    if state is None:
        state = fallback.q_seed
    if state is None:
        raise RuntimeError("Fallback scan found a reachable direction but did not produce a state.")
    return PanelSolution(
        point_xyz=np.asarray(point_xyz, dtype=float),
        direction_xyz=_normalize(fallback.directions_world[best_idx]),
        state=state,
        method="fallback",
        seed=seed,
    )


def _seed_display_states(csm: CSM, point_xyz: np.ndarray) -> list[DexterousMode3State]:
    target = np.asarray(point_xyz, dtype=float)
    radial_xy = float(np.linalg.norm(target[:2]))
    if radial_xy > 1e-9:
        phi0 = float(np.arctan2(target[1], target[0]))
        phi_values = [phi0, phi0 + 0.25 * np.pi, phi0 - 0.25 * np.pi, phi0 + np.pi, 0.0]
    else:
        phi_values = [0.0, 0.5 * np.pi, -0.5 * np.pi]
    theta1_values = [0.12, 0.22, 0.40, 0.58, min(float(csm.theta1_limit) * 0.82, 0.82)]
    theta2_values = [0.12, 0.28, 0.48, 0.70, min(float(csm.theta2_limit) * 0.82, 1.05)]
    L1_values = [
        max(0.12 * float(csm.L_10), 1e-5),
        0.28 * float(csm.L_10),
        0.48 * float(csm.L_10),
        0.68 * float(csm.L_10),
        0.88 * float(csm.L_10),
    ]

    states: list[DexterousMode3State] = []
    seen: set[tuple[float, float, float, float]] = set()
    for phi in phi_values:
        phi_wrapped = float(np.arctan2(np.sin(phi), np.cos(phi)))
        for theta1 in theta1_values:
            for theta2 in theta2_values:
                for L1 in L1_values:
                    theta1_eff = min(theta1, max(1e-6, L1 / max(float(csm.ri_min or 1.0), 1e-8)))
                    key = (
                        round(phi_wrapped, 6),
                        round(float(theta1_eff), 6),
                        round(float(theta2), 6),
                        round(float(L1), 6),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    states.append(
                        DexterousMode3State(
                            phi=phi_wrapped,
                            theta1=float(theta1_eff),
                            L1=float(L1),
                            delta1=0.0,
                            theta2=float(theta2),
                            delta2=0.0,
                        )
                    )
    return states


def _preferred_probe_direction(probe) -> np.ndarray:
    if probe.feasible_directions_world.size:
        mean_dir = np.mean(probe.feasible_directions_world, axis=0)
        if np.linalg.norm(mean_dir) > 1e-9:
            return _normalize(mean_dir)
        return _normalize(probe.feasible_directions_world[0])
    if probe.type1_boundary_world.size:
        mean_dir = np.mean(probe.type1_boundary_world, axis=0)
        if np.linalg.norm(mean_dir) > 1e-9:
            return _normalize(mean_dir)
        return _normalize(probe.type1_boundary_world[0])
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _approximate_panel_state(csm: CSM, point_xyz: np.ndarray, seed: int, probe) -> PanelSolution:
    preferred_direction = _preferred_probe_direction(probe)
    best_score = float("inf")
    best_state: DexterousMode3State | None = None
    best_axis = preferred_direction
    for state in _seed_display_states(csm, point_xyz):
        display_csm = make_mode3_display_csm(csm, state)
        pos = np.asarray(display_csm.pose[:3], dtype=float)
        axis = _normalize(np.asarray(display_csm.pose[3:], dtype=float))
        pos_err = float(np.linalg.norm(pos - point_xyz))
        axis_err = 1.0 - float(np.clip(np.dot(axis, preferred_direction), -1.0, 1.0))
        score = pos_err + 0.008 * axis_err
        if score < best_score:
            best_score = score
            best_state = state
            best_axis = axis

    if best_state is None:
        raise RuntimeError("Unable to choose an approximate display state for the selected point.")
    return PanelSolution(
        point_xyz=np.asarray(point_xyz, dtype=float),
        direction_xyz=best_axis,
        state=best_state,
        method="approx",
        seed=seed,
    )


def _resolve_panel_state(csm: CSM, point_xyz: np.ndarray, seed: int, probe) -> PanelSolution:
    try:
        return _solve_panel_state(csm, point_xyz, seed)
    except RuntimeError:
        return _approximate_panel_state(csm, point_xyz, seed, probe)


def _panel_points(csm: CSM, box: OperationBox) -> list[tuple[str, np.ndarray, int]]:
    center = np.asarray(box.center_xyz, dtype=float)
    half = np.asarray(box.half_size_xyz, dtype=float)
    points = [("Center", center, 20260410)]

    secondary_seed = 20260411
    fallback_error: RuntimeError | None = None
    candidate_offsets: list[tuple[str, np.ndarray]] = [
        ("Offset (-0.9, -0.9, +0.0)", np.array([-0.9, -0.9, 0.0], dtype=float)),
        ("Offset (-0.9, -0.9, +0.4)", np.array([-0.9, -0.9, 0.4], dtype=float)),
        ("Offset (-0.9, -0.9, -0.4)", np.array([-0.9, -0.9, -0.4], dtype=float)),
        ("Offset (-0.7, -0.7, +0.0)", np.array([-0.7, -0.7, 0.0], dtype=float)),
        ("Offset (+0.9, +0.9, +0.0)", np.array([0.9, 0.9, 0.0], dtype=float)),
        ("Offset (+0.7, +0.7, +0.0)", np.array([0.7, 0.7, 0.0], dtype=float)),
    ]
    xy_levels = (0.9, 0.75, 0.6, 0.45, 0.3)
    z_levels = (0.0, 0.35, -0.35, 0.65, -0.65)
    xy_signs = ((-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0))
    seen_offsets: set[tuple[float, float, float]] = {
        tuple(np.round(offset_xyz, 6).tolist()) for _, offset_xyz in candidate_offsets
    }
    for level in xy_levels:
        for sx, sy in xy_signs:
            for z in z_levels:
                offset_xyz = np.array([sx * level, sy * level, z], dtype=float)
                key = tuple(np.round(offset_xyz, 6).tolist())
                if key in seen_offsets:
                    continue
                seen_offsets.add(key)
                candidate_offsets.append(
                    (
                        f"Offset ({offset_xyz[0]:+.2f}, {offset_xyz[1]:+.2f}, {offset_xyz[2]:+.2f})",
                        offset_xyz,
                    )
                )
    for title, offset_scale_xyz in candidate_offsets:
        point_xyz = center + offset_scale_xyz * half
        try:
            probe = build_dexterous_probe(
                point_xyz,
                csm=csm,
                config="3mm",
                method="analytic",
                validate_with_fallback=True,
                label=title,
            )
            has_region = bool(
                probe.feasible_directions_world.size
                or probe.type1_boundary_world.size
                or probe.type2_boundary_world.size
            )
            if not has_region:
                raise RuntimeError("No dexterous region available for the selected point.")
        except RuntimeError as exc:
            fallback_error = exc
            continue
        points.append((title, point_xyz, secondary_seed))
        return points

    if fallback_error is not None:
        raise fallback_error
    raise RuntimeError("Unable to choose a reachable secondary point inside the operation box.")


def _overlay_point_and_arrow(ax, point_xyz: np.ndarray, direction_xyz: np.ndarray) -> None:
    ax.scatter(
        [point_xyz[0]],
        [point_xyz[1]],
        [point_xyz[2]],
        s=32,
        color="#111827",
        depthshade=False,
        zorder=6,
    )
    arrow_length = 0.018
    ax.quiver(
        point_xyz[0],
        point_xyz[1],
        point_xyz[2],
        direction_xyz[0],
        direction_xyz[1],
        direction_xyz[2],
        length=arrow_length,
        normalize=True,
        color="#D97706",
        linewidth=2.2,
        arrow_length_ratio=0.18,
    )


def _collect_box_and_direction_points(box: OperationBox, panel: PanelSolution) -> np.ndarray:
    verts = np.stack(list(operation_box_vertices(box).values()), axis=0)
    arrow_tip = panel.point_xyz + 0.018 * panel.direction_xyz
    return np.vstack((verts, panel.point_xyz[None, :], arrow_tip[None, :]))


def _collect_robot_points(csm: CSM, panel: PanelSolution) -> np.ndarray:
    if panel.state is None:
        return np.zeros((0, 3), dtype=float)
    display_csm = make_mode3_display_csm(csm, panel.state)
    vis = display_csm.get_visualization_segments(arc_points=28, straight_points=6)
    points = []
    for seg in vis["segments"]:
        points.append(np.asarray(seg["points"], dtype=float))
    points.append(np.asarray([vis["tool"]["start"], vis["tool"]["end"]], dtype=float))
    return np.vstack(points)


def _style_paper_axes(ax) -> None:
    ax.set_proj_type("ortho")
    pane_face = (1.0, 1.0, 1.0, 0.03)
    pane_edge = (0.35, 0.35, 0.35, 0.20)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor(pane_face)
        axis.pane.set_edgecolor(pane_edge)
    ax.grid(True, alpha=0.12, linestyle=":", linewidth=0.7)


def _compute_shared_axes_limits(
    *,
    csm: CSM,
    box: OperationBox,
    panels: list[PanelSolution],
    sphere_radius_m: float,
    xy_pad: float = 0.0015,
    z_pad_top: float = 0.006,
) -> tuple[float, float]:
    content: list[np.ndarray] = []
    for panel in panels:
        content.append(_collect_box_and_direction_points(box, panel))
        content.append(_collect_robot_points(csm, panel))
    points = np.vstack([pts for pts in content if pts.size]) if any(pts.size for pts in content) else np.zeros((0, 3), dtype=float)
    if points.size == 0:
        points = np.zeros((1, 3), dtype=float)
    xy_half = max(
        float(np.max(np.abs(points[:, 0]))) if points.size else 0.0,
        float(np.max(np.abs(points[:, 1]))) if points.size else 0.0,
        float(np.max(np.abs(box.bounds_max_xyz[:2]))),
        sphere_radius_m,
    ) + xy_pad
    z_top = max(
        float(np.max(points[:, 2])) if points.size else 0.0,
        float(box.bounds_max_xyz[2]),
        max(float(panel.point_xyz[2] + sphere_radius_m) for panel in panels),
    ) + z_pad_top
    return xy_half, z_top


def _set_axes_limits(ax, *, xy_half: float, z_top: float) -> None:
    ax.set_xlim(-xy_half, xy_half)
    ax.set_ylim(-xy_half, xy_half)
    ax.set_zlim(0.0, z_top)
    # Keep world units isotropic: x/y/z must remain on a strict 1:1:1 physical scale.
    ax.set_box_aspect((2.0 * xy_half, 2.0 * xy_half, z_top))
    _style_paper_axes(ax)


def build_figure(
    *,
    csm: CSM,
    box_size_mm: np.ndarray,
    top_margin_mm: float,
    output_path: Path | None,
    show_figure: bool,
) -> tuple[OperationBox, list[PanelSolution], float]:
    profile = build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=220, angle_samples=220),
    )[0]
    box, box_scale = fit_largest_centered_operation_box_in_mode3(
        profile,
        size_xyz_m=box_size_mm / 1000.0,
        top_margin_m=top_margin_mm / 1000.0,
    )
    sphere_radius_m = 0.5 * SPHERE_DIAMETER_TO_BOX_MEAN_RATIO * float(np.mean(box.size_xyz))

    solutions: list[PanelSolution] = []
    probes = []
    for title, point_xyz, seed in _panel_points(csm, box):
        probe = build_dexterous_probe(
            point_xyz,
            csm=csm,
            config="3mm",
            method="analytic",
            validate_with_fallback=False,
            sphere_radius_m=sphere_radius_m,
            label=title,
        )
        solution = _resolve_panel_state(csm, point_xyz, seed, probe)
        solutions.append(solution)
        probes.append(probe)

    fig = plt.figure(figsize=(13.4, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    plot_options = DexterousPlotOptions(
        show_figure=False,
        display_frame="world",
        show_robot=True,
        elev=18.0,
        azim=-40.0,
        save_debug_figure=False,
    )

    shared_xy_half, shared_z_top = _compute_shared_axes_limits(
        csm=csm,
        box=box,
        panels=solutions,
        sphere_radius_m=sphere_radius_m,
    )

    for idx, (probe, solution) in enumerate(zip(probes, solutions), start=1):
        probe.display_state = solution.state
        ax = fig.add_subplot(gs[0, idx - 1], projection="3d")
        plot_dexterous_probe(ax, probe, csm=csm, options=plot_options)
        draw_operation_box(
            ax,
            box,
            face_color="#7EB9DF",
            edge_color="#35566E",
            alpha=0.20,
            linewidth=1.0,
        )
        _overlay_point_and_arrow(ax, solution.point_xyz, solution.direction_xyz)
        _set_axes_limits(ax, xy_half=shared_xy_half, z_top=shared_z_top)
        ax.view_init(elev=14.0, azim=-66.0)
        ax.set_title(f"{probe.label}\n{solution.method} seed={solution.seed}", pad=10.0)

    fig.suptitle("Mode3 Dexterous Workspace at Two Operation-Box Points", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    return box, solutions, box_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a mode3 dexterous-workspace illustration.")
    parser.add_argument("--config", default="config/csm_cfg_3mm.yaml", help="CSM config path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output figure path.")
    parser.add_argument("--box-size-mm", default="50,50,40", help="Operation box size in millimeters: sx,sy,sz.")
    parser.add_argument("--top-margin-mm", default=DEFAULT_TOP_MARGIN_MM, type=float, help="Top clearance to workspace roof.")
    parser.add_argument("--hide", action="store_true", help="Render without opening a window.")
    args = parser.parse_args()

    csm = CSM.from_config(Path(args.config))
    requested_box_size_mm = _parse_box_size_mm(args.box_size_mm)
    box, solutions, box_scale = build_figure(
        csm=csm,
        box_size_mm=requested_box_size_mm,
        top_margin_mm=float(args.top_margin_mm),
        output_path=Path(args.output),
        show_figure=not args.hide,
    )
    print(f"Saved dexterous figure to: {Path(args.output).resolve()}")
    if box_scale < 0.999:
        print(
            "Requested box was uniformly scaled to fit mode3:",
            f"scale={box_scale:.3f}",
            f"requested_mm={requested_box_size_mm.round(2).tolist()}",
        )
    print(f"Operation box center [mm]: {(1000.0 * box.center_xyz).round(2).tolist()}")
    print(f"Operation box size   [mm]: {(1000.0 * box.size_xyz).round(2).tolist()}")
    for solution in solutions:
        print(
            "Panel:",
            f"point_mm={(1000.0 * solution.point_xyz).round(2).tolist()}",
            f"direction={np.round(solution.direction_xyz, 4).tolist()}",
            f"method={solution.method}",
            f"seed={solution.seed}",
        )


if __name__ == "__main__":
    main()
