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
    DexterousParameters,
    DexterousPlotOptions,
    OperationBox,
    analytic_fk_mode3,
    build_dexterous_probe,
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


def _panel_points(csm: CSM, box: OperationBox) -> list[tuple[str, np.ndarray, int]]:
    center = np.asarray(box.center_xyz, dtype=float)
    half = np.asarray(box.half_size_xyz, dtype=float)
    points = [("Center", center, 20260410)]

    secondary_seed = 20260411
    fallback_error: RuntimeError | None = None
    candidate_offsets = [
        ("Top Front Right", np.array([1.0, 1.0, 1.0], dtype=float)),
        ("Upper Front Right", np.array([0.85, 0.85, 0.50], dtype=float)),
        ("Mid Front Right", np.array([0.70, 0.70, 0.10], dtype=float)),
        ("Lower Front Right", np.array([0.55, 0.55, -0.25], dtype=float)),
        ("Interior Front Right", np.array([0.40, 0.40, -0.45], dtype=float)),
        ("Right Interior", np.array([0.60, 0.10, -0.10], dtype=float)),
        ("Front Interior", np.array([0.10, 0.60, -0.10], dtype=float)),
    ]
    for title, offset_scale_xyz in candidate_offsets:
        point_xyz = center + offset_scale_xyz * half
        try:
            _solve_panel_state(csm, point_xyz, secondary_seed)
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


def _set_equal_3d_axes(ax, points: np.ndarray, pad: float = 0.01) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * max(float(np.max(maxs - mins)), 1e-3) + pad
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _expand_axes_to_include(ax, extra_points: np.ndarray, pad: float = 0.006) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    current = np.array(
        [
            [xlim[0], ylim[0], zlim[0]],
            [xlim[0], ylim[0], zlim[1]],
            [xlim[0], ylim[1], zlim[0]],
            [xlim[0], ylim[1], zlim[1]],
            [xlim[1], ylim[0], zlim[0]],
            [xlim[1], ylim[0], zlim[1]],
            [xlim[1], ylim[1], zlim[0]],
            [xlim[1], ylim[1], zlim[1]],
        ],
        dtype=float,
    )
    _set_equal_3d_axes(ax, np.vstack((current, np.asarray(extra_points, dtype=float))), pad=pad)


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

    solutions: list[PanelSolution] = []
    probes = []
    for title, point_xyz, seed in _panel_points(csm, box):
        solution = _solve_panel_state(csm, point_xyz, seed)
        solutions.append(solution)
        probes.append(
            build_dexterous_probe(
                point_xyz,
                csm=csm,
                config="3mm",
                method="analytic",
                validate_with_fallback=True,
                label=title,
            )
        )

    fig = plt.figure(figsize=(13.4, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    plot_options = DexterousPlotOptions(
        show_figure=False,
        display_frame="world",
        show_robot=True,
        sphere_alpha=0.12,
        patch_alpha=0.72,
        sphere_color="#70C98B",
        patch_color="#C92EC8",
        robot_colors=("#2563EB", "#22C55E", "#EF4444"),
        elev=18.0,
        azim=-40.0,
        save_debug_figure=False,
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
        _expand_axes_to_include(ax, _collect_box_and_direction_points(box, solution), pad=0.012)
        ax.view_init(elev=18.0, azim=-40.0)
        ax.set_title(f"{probe.label}\n{solution.method} seed={solution.seed}", pad=10.0)
        ax.grid(True, alpha=0.18)

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
