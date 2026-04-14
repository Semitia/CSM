"""
Render two mode3 dexterous-workspace panels for one box center and one box vertex.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

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
    bootstrap_position_reachable_state,
    mode3_state_from_csm,
    clone_csm,
    scan_directions,
)
from csm.utils import axis_angle_from_vectors, normalize_vector
from csm.workspace_boundary_scan import BoundaryScanOptions, build_workspace_profiles


DEFAULT_OUTPUT = Path("data/example_mode3_dexterous_figure.png")
DEFAULT_BOX_SIZE_MM = np.array([50.0, 50.0, 40.0], dtype=float)
DEFAULT_TOP_MARGIN_MM = 4.0
SPHERE_DIAMETER_TO_BOX_MEAN_RATIO = 0.8
PROFILE_SCAN_LENGTH_SAMPLES = 120
PROFILE_SCAN_ANGLE_SAMPLES = 120
ARROW_TO_SPHERE_RADIUS_RATIO = 0.82
PANEL_SELECTION_FALLBACK_DIRECTIONS = 36
PANEL_SOLVE_FALLBACK_DIRECTIONS = 120
PANEL_EXACT_FALLBACK_DIRECTIONS = 720
OUTPUT_DPI = 180
MM_PER_M = 1000.0
BOX_INFO_SCHEMA_VERSION = 1
POSITION_SEED_RANDOM_SAMPLES = 2500


@dataclass(frozen=True)
class FigureStyle:
    sphere_color: str = "#7A88B8"
    sphere_alpha: float = 0.3
    patch_color: str = "#DD2E9F"
    patch_alpha: float = 0.15
    robot_colors: tuple[str, str, str] = ("#1E40AF", "#22C55E", "#DC2626")
    box_face_color: str = "#807EDF"
    box_edge_color: str = "#532764"
    box_alpha: float = 0.15
    box_linewidth: float = 0.8
    point_color: str = "#111827"
    point_size: float = 32.0
    arrow_color: str = "#D97706"
    arrow_linewidth: float = 2.2
    outer_box_color: tuple[float, float, float, float] = (0.18, 0.18, 0.18, 0.42)
    outer_box_linewidth: float = 1.15
    grid_color: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 0.28)
    grid_linewidth: float = 0.7
    grid_linestyle: str = ":"
    xy_pad_m: float = 0.001
    z_pad_top_m: float = 0.008
    main_title: str = "Mode3 Dexterous Workspace at Two Operation-Box Points"


DEFAULT_STYLE = FigureStyle()


@dataclass(frozen=True)
class PanelSolution:
    point_xyz: np.ndarray
    direction_xyz: np.ndarray
    state: object
    method: str
    seed: int


@dataclass(frozen=True)
class PanelTarget:
    title: str
    point_xyz: np.ndarray
    seed: int


def _parse_box_size_mm(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.shape != (3,):
        raise ValueError("Expected --box-size-mm as 'sx,sy,sz'.")
    return values


def _csm_signature(csm: CSM) -> dict[str, float]:
    return {
        "L_10_m": float(csm.L_10),
        "L_20_m": float(csm.L_20),
        "L_r0_m": float(csm.L_r0),
        "L_s0_m": float(csm.L_s0),
        "L_tool_m": float(csm.L_tool),
        "theta1_limit_rad": float(csm.theta1_limit),
        "theta2_limit_rad": float(csm.theta2_limit),
        "ri_min_m": None if csm.r1_min is None else float(csm.r1_min),
    }


def _assert_matching_csm_signature(csm: CSM, saved_signature: dict[str, object]) -> None:
    current = _csm_signature(csm)
    for key, current_value in current.items():
        saved_value = saved_signature.get(key)
        if saved_value is None and current_value is None:
            continue
        if saved_value is None or current_value is None:
            raise ValueError(f"Box info mismatch for {key}: saved={saved_value}, current={current_value}")
        if abs(float(saved_value) - float(current_value)) > 1e-9:
            raise ValueError(f"Box info mismatch for {key}: saved={saved_value}, current={current_value}")


def _load_box_info(path: Path, *, csm: CSM) -> tuple[OperationBox, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    if schema_version != BOX_INFO_SCHEMA_VERSION:
        raise ValueError(f"Unsupported box info schema_version={schema_version}.")
    saved_signature = payload.get("csm_signature")
    if not isinstance(saved_signature, dict):
        raise ValueError("Box info file is missing csm_signature.")
    _assert_matching_csm_signature(csm, saved_signature)
    center_mm = np.asarray(payload["box_center_mm"], dtype=float)
    size_mm = np.asarray(payload["box_size_mm"], dtype=float)
    if center_mm.shape != (3,) or size_mm.shape != (3,):
        raise ValueError("Box info must contain 3D box_center_mm and box_size_mm.")
    box = OperationBox(center_xyz=center_mm / MM_PER_M, size_xyz=size_mm / MM_PER_M)
    box_scale = float(payload.get("box_scale", 1.0))
    return box, box_scale


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _log_timing(enabled: bool, label: str, start_time: float, **stats: object) -> None:
    if not enabled:
        return
    elapsed = time.perf_counter() - start_time
    suffix = ""
    if stats:
        parts = [f"{key}={value}" for key, value in stats.items()]
        suffix = " " + " ".join(parts)
    print(f"[timing] {label}: {elapsed:.3f}s{suffix}")


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


def _sample_random_mode3_state(rng: np.random.Generator, params: DexterousParameters) -> DexterousMode3State:
    theta1 = float(rng.uniform(0.05 * params.theta1_plus, 0.98 * params.theta1_plus))
    theta2 = float(rng.uniform(0.05 * params.theta2_plus, 0.98 * params.theta2_plus))
    min_L1 = max(params.r1_minus_m * theta1, 1e-5)
    L1 = float(rng.uniform(min_L1, params.L10_m))
    return DexterousMode3State(
        phi=float(rng.uniform(-np.pi, np.pi)),
        theta1=theta1,
        L1=L1,
        delta1=float(rng.uniform(-np.pi, np.pi)),
        theta2=theta2,
        delta2=float(rng.uniform(-np.pi, np.pi)),
    )


def _bootstrap_nearest_position_seed(csm: CSM, point_xyz: np.ndarray, seed: int) -> DexterousMode3State | None:
    point_xyz = np.asarray(point_xyz, dtype=float)
    params = DexterousParameters.from_csm(csm)
    best_state: DexterousMode3State | None = None
    best_dist = float("inf")

    for state in _seed_display_states(csm, point_xyz):
        fk_point, _ = analytic_fk_mode3(params, state)
        dist = float(np.linalg.norm(fk_point - point_xyz))
        if dist < best_dist:
            best_dist = dist
            best_state = state

    rng = np.random.default_rng(seed)
    for _ in range(POSITION_SEED_RANDOM_SAMPLES):
        state = _sample_random_mode3_state(rng, params)
        fk_point, _ = analytic_fk_mode3(params, state)
        dist = float(np.linalg.norm(fk_point - point_xyz))
        if dist < best_dist:
            best_dist = dist
            best_state = state

    return best_state


def _solve_target_pose(
    worker: CSM,
    target_pose: np.ndarray,
    *,
    require_orientation: bool = True,
    max_steps: int = 700,
    pos_tol: float = 2.0e-4,
    ori_tol: float = np.deg2rad(2.0),
    v_gain: float = 5.0,
    w_gain: float = 4.0,
) -> tuple[bool, float, float]:
    worker.target_pose = np.asarray(target_pose, dtype=float).copy()
    for _ in range(max_steps):
        pos_error_vec = worker.target_pose[:3] - worker.pose[:3]
        pos_error = float(np.linalg.norm(pos_error_vec))
        if require_orientation:
            _, ori_error = axis_angle_from_vectors(worker.pose[3:], worker.target_pose[3:])
        else:
            ori_error = 0.0
        if pos_error < pos_tol and (not require_orientation or ori_error < ori_tol) and int(worker.mode) == 3:
            return True, pos_error, ori_error
        linear_velocity = normalize_vector(pos_error_vec) * min(v_gain * pos_error, 0.06)
        if require_orientation:
            axis_hat, theta = axis_angle_from_vectors(worker.pose[3:], worker.target_pose[3:])
            angular_velocity = axis_hat * min(w_gain * theta, 4.0)
        else:
            angular_velocity = np.zeros(3, dtype=float)
        worker.get_dot_PHI(linear_velocity, angular_velocity)
        worker.step()
        worker.update()
        worker.update_jacobians()
        worker.check_transition()
        if int(worker.mode) != 3:
            return False, pos_error, ori_error
    return False, pos_error, ori_error


def _ordered_probe_directions(probe) -> np.ndarray:
    direction_sets = []
    if probe.feasible_directions_world.size:
        direction_sets.append(np.asarray(probe.feasible_directions_world, dtype=float))
    if probe.type1_boundary_world.size:
        direction_sets.append(np.asarray(probe.type1_boundary_world, dtype=float))
    if not direction_sets:
        return np.zeros((0, 3), dtype=float)
    directions = np.vstack(direction_sets)
    preferred = _preferred_probe_direction(probe)
    order = np.argsort(-(directions @ preferred))
    directions = directions[order]
    if directions.shape[0] > 180:
        sample_idx = np.linspace(0, directions.shape[0] - 1, 180).round().astype(int)
        directions = directions[np.unique(sample_idx)]
    return directions


def _solve_panel_state_from_probe_directions(
    csm: CSM,
    point_xyz: np.ndarray,
    seed: int,
    probe,
) -> PanelSolution:
    q_seed = probe.display_state
    if q_seed is None:
        q_seed = bootstrap_position_reachable_state(csm, np.asarray(point_xyz, dtype=float))
    if q_seed is None:
        q_seed = _bootstrap_nearest_position_seed(csm, np.asarray(point_xyz, dtype=float), seed)
    if q_seed is None:
        raise RuntimeError("Could not bootstrap a position-reachable seed state for this probe point.")
    worker = make_mode3_display_csm(csm, q_seed)
    target_pose = np.concatenate([np.asarray(point_xyz, dtype=float), np.asarray(worker.pose[3:], dtype=float)], dtype=float)
    ok, pos_err, _ = _solve_target_pose(worker, target_pose, require_orientation=False)
    if not ok:
        raise RuntimeError("Position-only Jacobian solve did not converge to the requested target point.")
    solved_state = mode3_state_from_csm(worker)
    fk_point, fk_axis = analytic_fk_mode3(DexterousParameters.from_csm(csm), solved_state)
    if float(np.linalg.norm(fk_point - point_xyz)) > 2.0e-4:
        raise RuntimeError("Position-only Jacobian solve ended outside the target-point tolerance.")
    return PanelSolution(
        point_xyz=np.asarray(point_xyz, dtype=float),
        direction_xyz=_normalize(fk_axis),
        state=solved_state,
        method="jacobian-pos",
        seed=seed,
    )


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

    fallback = scan_directions(csm, point_xyz, n_directions=PANEL_SOLVE_FALLBACK_DIRECTIONS)
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
                    theta1_eff = min(theta1, max(1e-6, L1 / max(float(csm.r1_min or csm.ri_min or 1.0), 1e-8)))
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


def _solve_panel_state_with_fallback_scan(
    csm: CSM,
    point_xyz: np.ndarray,
    seed: int,
    probe,
) -> PanelSolution:
    preferred_direction = _preferred_probe_direction(probe)
    q_seed = probe.display_state
    fallback = scan_directions(
        csm,
        point_xyz,
        q_seed=q_seed,
        n_directions=PANEL_EXACT_FALLBACK_DIRECTIONS,
    )
    reachable_idx = np.flatnonzero(fallback.reachable_mask)
    if reachable_idx.size == 0:
        raise RuntimeError("Fallback scan did not find a reachable exact pose for this probe point.")
    scores = fallback.directions_world[reachable_idx] @ preferred_direction
    best_idx = int(reachable_idx[int(np.argmax(scores))])
    state = fallback.solved_states[best_idx]
    if state is None:
        state = fallback.q_seed
    if state is None:
        raise RuntimeError("Fallback scan found a reachable direction but did not produce a solved state.")
    params = DexterousParameters.from_csm(csm)
    direction_xyz = _normalize(fallback.directions_world[best_idx])
    if not _validate_state(params, state, point_xyz, direction_xyz):
        raise RuntimeError("Fallback scan returned a state that does not accurately hit the requested target pose.")
    return PanelSolution(
        point_xyz=np.asarray(point_xyz, dtype=float),
        direction_xyz=direction_xyz,
        state=state,
        method="fallback-exact",
        seed=seed,
    )


def _resolve_panel_state(csm: CSM, point_xyz: np.ndarray, seed: int, probe) -> PanelSolution:
    return _solve_panel_state_from_probe_directions(csm, point_xyz, seed, probe)


def _fast_panel_solution_from_probe(
    csm: CSM,
    point_xyz: np.ndarray,
    seed: int,
    probe,
) -> PanelSolution:
    preferred_direction = _preferred_probe_direction(probe)
    if probe.display_state is not None:
        return PanelSolution(
            point_xyz=np.asarray(point_xyz, dtype=float),
            direction_xyz=preferred_direction,
            state=probe.display_state,
            method="probe-seed",
            seed=seed,
        )
    return _approximate_panel_state(csm, point_xyz, seed, probe)


def _panel_targets(box: OperationBox) -> list[PanelTarget]:
    center = np.asarray(box.center_xyz, dtype=float)
    half = np.asarray(box.half_size_xyz, dtype=float)
    return [
        PanelTarget("Vertex (+1, -1, +1)", center + np.array([-1.0, 1.0, 1.0], dtype=float) * half, 20260410),
        PanelTarget("Vertex (+1, -1, -1)", center + np.array([1.0, -1.0, -1.0], dtype=float) * half, 20260411),
        PanelTarget("Inner Offset (+0.18, -0.12, -0.10)", center + np.array([0.2, -0.15, -0.10], dtype=float) * half, 20260412),
    ]


def _panel_arrow_length(sphere_radius_m: float) -> float:
    return max(0.001, ARROW_TO_SPHERE_RADIUS_RATIO * float(sphere_radius_m))


def _overlay_point_and_arrow(
    ax,
    point_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    sphere_radius_m: float,
    style: FigureStyle,
) -> None:
    ax.scatter(
        [point_xyz[0]],
        [point_xyz[1]],
        [point_xyz[2]],
        s=style.point_size,
        color=style.point_color,
        depthshade=False,
        zorder=6,
    )
    arrow_length = _panel_arrow_length(sphere_radius_m)
    ax.quiver(
        point_xyz[0],
        point_xyz[1],
        point_xyz[2],
        direction_xyz[0],
        direction_xyz[1],
        direction_xyz[2],
        length=arrow_length,
        normalize=True,
        color=style.arrow_color,
        linewidth=style.arrow_linewidth,
        arrow_length_ratio=0.18,
    )


def _collect_box_and_direction_points(box: OperationBox, panel: PanelSolution, sphere_radius_m: float) -> np.ndarray:
    verts = np.stack(list(operation_box_vertices(box).values()), axis=0)
    arrow_tip = panel.point_xyz + _panel_arrow_length(sphere_radius_m) * panel.direction_xyz
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


def _style_paper_axes(ax, style: FigureStyle) -> None:
    ax.set_proj_type("ortho")
    pane_face = (1.0, 1.0, 1.0, 0.0)
    pane_edge = (0.35, 0.35, 0.35, 0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor(pane_face)
        axis.pane.set_edgecolor(pane_edge)
        axis._axinfo["grid"]["color"] = style.grid_color
        axis._axinfo["grid"]["linestyle"] = style.grid_linestyle
        axis._axinfo["grid"]["linewidth"] = style.grid_linewidth
        axis._axinfo["axisline"]["color"] = (0.0, 0.0, 0.0, 0.0)
    ax.grid(True)


def _cuboid_edges(xlim: tuple[float, float], ylim: tuple[float, float], zlim: tuple[float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    vertices = {
        "000": np.array([x0, y0, z0], dtype=float),
        "001": np.array([x0, y0, z1], dtype=float),
        "010": np.array([x0, y1, z0], dtype=float),
        "011": np.array([x0, y1, z1], dtype=float),
        "100": np.array([x1, y0, z0], dtype=float),
        "101": np.array([x1, y0, z1], dtype=float),
        "110": np.array([x1, y1, z0], dtype=float),
        "111": np.array([x1, y1, z1], dtype=float),
    }
    edge_keys = [
        ("000", "100"),
        ("000", "010"),
        ("000", "001"),
        ("001", "101"),
        ("001", "011"),
        ("010", "110"),
        ("010", "011"),
        ("100", "110"),
        ("100", "101"),
        ("011", "111"),
        ("101", "111"),
        ("110", "111"),
    ]
    return [(vertices[a], vertices[b]) for a, b in edge_keys]


def _draw_outer_axes_box(
    ax,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    style: FigureStyle,
) -> None:
    for start, end in _cuboid_edges(xlim, ylim, zlim):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=style.outer_box_color,
            linewidth=style.outer_box_linewidth,
            zorder=0,
        )


def _choose_tick_step_mm(span_mm: float, target_ticks: int) -> float:
    raw_step = max(span_mm / max(target_ticks, 1), 1.0)
    candidates = np.array([1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0], dtype=float)
    idx = int(np.searchsorted(candidates, raw_step, side="left"))
    return float(candidates[min(idx, len(candidates) - 1)])


def _build_ticks_mm(vmin_m: float, vmax_m: float, *, target_ticks: int, symmetric: bool) -> tuple[np.ndarray, list[str]]:
    scale = 1000.0
    vmin_mm = vmin_m * scale
    vmax_mm = vmax_m * scale
    if symmetric:
        span_mm = max(abs(vmin_mm), abs(vmax_mm))
        step_mm = _choose_tick_step_mm(2.0 * span_mm, target_ticks)
        tick_max_mm = step_mm * np.floor(span_mm / step_mm)
        ticks_mm = np.arange(-tick_max_mm, tick_max_mm + 0.5 * step_mm, step_mm)
    else:
        span_mm = max(vmax_mm - vmin_mm, 1.0)
        step_mm = _choose_tick_step_mm(span_mm, target_ticks)
        start_mm = step_mm * np.ceil(vmin_mm / step_mm)
        end_mm = step_mm * np.floor(vmax_mm / step_mm)
        ticks_mm = np.arange(start_mm, end_mm + 0.5 * step_mm, step_mm)
    if ticks_mm.size == 0:
        ticks_mm = np.array([0.0], dtype=float)
    ticks_m = ticks_mm / scale
    labels = [f"{int(round(tick))}" if abs(tick - round(tick)) < 1e-9 else f"{tick:g}" for tick in ticks_mm]
    return ticks_m, labels


def _apply_mm_ticks(ax, *, xy_half: float, z_top: float) -> None:
    xticks, xlabels = _build_ticks_mm(-xy_half, xy_half, target_ticks=5, symmetric=True)
    yticks, ylabels = _build_ticks_mm(-xy_half, xy_half, target_ticks=5, symmetric=True)
    zticks, zlabels = _build_ticks_mm(0.0, z_top, target_ticks=6, symmetric=False)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.zaxis.set_major_locator(FixedLocator(zticks))
    ax.xaxis.set_major_formatter(FixedFormatter(xlabels))
    ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
    ax.zaxis.set_major_formatter(FixedFormatter(zlabels))
    ax.set_xlabel("X [mm]", labelpad=10.0)
    ax.set_ylabel("Y [mm]", labelpad=10.0)
    ax.set_zlabel("Z [mm]", labelpad=6.0)
    ax.tick_params(axis="both", which="major", labelsize=9, pad=1)
    ax.tick_params(axis="z", which="major", labelsize=9, pad=2)


def _compute_shared_axes_limits(
    *,
    csm: CSM,
    box: OperationBox,
    panels: list[PanelSolution],
    sphere_radius_m: float,
    style: FigureStyle,
) -> tuple[float, float]:
    content: list[np.ndarray] = []
    for panel in panels:
        content.append(_collect_box_and_direction_points(box, panel, sphere_radius_m))
        content.append(_collect_robot_points(csm, panel))
    points = np.vstack([pts for pts in content if pts.size]) if any(pts.size for pts in content) else np.zeros((0, 3), dtype=float)
    if points.size == 0:
        points = np.zeros((1, 3), dtype=float)
    xy_half = max(
        float(np.max(np.abs(points[:, 0]))) if points.size else 0.0,
        float(np.max(np.abs(points[:, 1]))) if points.size else 0.0,
        float(np.max(np.abs(box.bounds_max_xyz[:2]))),
        sphere_radius_m,
    ) + style.xy_pad_m
    z_top = max(
        float(np.max(points[:, 2])) if points.size else 0.0,
        float(box.bounds_max_xyz[2]),
        max(float(panel.point_xyz[2] + sphere_radius_m) for panel in panels),
    ) + style.z_pad_top_m
    return xy_half, z_top


def _set_axes_limits(ax, *, xy_half: float, z_top: float, style: FigureStyle) -> None:
    ax.set_xlim(-xy_half, xy_half)
    ax.set_ylim(-xy_half, xy_half)
    ax.set_zlim(0.0, z_top)
    # Keep the displayed data scale strictly isotropic in x/y/z.
    ax.set_box_aspect((2.0 * xy_half, 2.0 * xy_half, max(z_top, 1e-6)))
    _style_paper_axes(ax, style)
    _apply_mm_ticks(ax, xy_half=xy_half, z_top=z_top)
    _draw_outer_axes_box(
        ax,
        xlim=(-xy_half, xy_half),
        ylim=(-xy_half, xy_half),
        zlim=(0.0, z_top),
        style=style,
    )


def build_figure(
    *,
    csm: CSM,
    box_size_mm: np.ndarray,
    top_margin_mm: float,
    output_path: Path | None,
    show_figure: bool,
    box_override: OperationBox | None = None,
    box_scale_override: float | None = None,
    style: FigureStyle = DEFAULT_STYLE,
    verbose_timing: bool = False,
) -> tuple[OperationBox, list[PanelSolution], float]:
    total_start = time.perf_counter()
    if box_override is None:
        stage_start = time.perf_counter()
        profile = build_workspace_profiles(
            csm,
            modes=[3],
            options=BoundaryScanOptions(
                length_samples=PROFILE_SCAN_LENGTH_SAMPLES,
                angle_samples=PROFILE_SCAN_ANGLE_SAMPLES,
            ),
        )[0]
        _log_timing(
            verbose_timing,
            "build workspace profile",
            stage_start,
            length_samples=PROFILE_SCAN_LENGTH_SAMPLES,
            angle_samples=PROFILE_SCAN_ANGLE_SAMPLES,
        )

        stage_start = time.perf_counter()
        box, box_scale = fit_largest_centered_operation_box_in_mode3(
            profile,
            size_xyz_m=box_size_mm / 1000.0,
            top_margin_m=top_margin_mm / 1000.0,
        )
        _log_timing(verbose_timing, "fit operation box", stage_start, box_scale=f"{box_scale:.3f}")
    else:
        box = box_override
        box_scale = 1.0 if box_scale_override is None else float(box_scale_override)
        _log_timing(verbose_timing, "load operation box", total_start, box_scale=f"{box_scale:.3f}")
    sphere_radius_m = 0.5 * SPHERE_DIAMETER_TO_BOX_MEAN_RATIO * float(np.mean(box.size_xyz))

    solutions: list[PanelSolution] = []
    probes = []
    targets = _panel_targets(box)
    for panel_idx, target in enumerate(targets, start=1):
        probe_start = time.perf_counter()
        probe = build_dexterous_probe(
            target.point_xyz,
            csm=csm,
            config="3mm",
            method="analytic",
            validate_with_fallback=False,
            sphere_radius_m=sphere_radius_m,
            label=target.title,
        )
        _log_timing(
            verbose_timing,
            f"panel[{panel_idx}] probe {target.title}",
            probe_start,
            feasible=int(probe.feasible_directions_world.shape[0]),
            type1=int(probe.type1_boundary_world.shape[0]),
            type2=int(probe.type2_boundary_world.shape[0]),
        )
        solve_start = time.perf_counter()
        solution = _resolve_panel_state(csm, target.point_xyz, target.seed, probe)
        _log_timing(
            verbose_timing,
            f"panel[{panel_idx}] solve {target.title}",
            solve_start,
            method=solution.method,
        )
        solutions.append(solution)
        probes.append(probe)

    stage_start = time.perf_counter()
    fig = plt.figure(figsize=(19.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, len(solutions))
    _log_timing(verbose_timing, "create figure", stage_start)

    plot_options = DexterousPlotOptions(
        show_figure=False,
        display_frame="world",
        show_robot=True,
        sphere_alpha=style.sphere_alpha,
        patch_alpha=style.patch_alpha,
        sphere_color=style.sphere_color,
        patch_color=style.patch_color,
        robot_colors=style.robot_colors,
        elev=18.0,
        azim=-40.0,
        save_debug_figure=False,
    )

    shared_xy_half, shared_z_top = _compute_shared_axes_limits(
        csm=csm,
        box=box,
        panels=solutions,
        sphere_radius_m=sphere_radius_m,
        style=style,
    )

    for idx, (probe, solution) in enumerate(zip(probes, solutions), start=1):
        panel_render_start = time.perf_counter()
        probe.display_state = solution.state
        ax = fig.add_subplot(gs[0, idx - 1], projection="3d")
        plot_dexterous_probe(ax, probe, csm=csm, options=plot_options)
        draw_operation_box(
            ax,
            box,
            face_color=style.box_face_color,
            edge_color=style.box_edge_color,
            alpha=style.box_alpha,
            linewidth=style.box_linewidth,
        )
        _overlay_point_and_arrow(ax, solution.point_xyz, solution.direction_xyz, sphere_radius_m, style)
        _set_axes_limits(
            ax,
            xy_half=shared_xy_half,
            z_top=shared_z_top,
            style=style,
        )
        ax.view_init(elev=14.0, azim=-66.0)
        ax.set_title(f"{probe.label}\n{solution.method} seed={solution.seed}", pad=10.0)
        _log_timing(
            verbose_timing,
            f"panel[{idx}] render {probe.label}",
            panel_render_start,
            feasible=int(probe.feasible_directions_world.shape[0]),
            type1=int(probe.type1_boundary_world.shape[0]),
            type2=int(probe.type2_boundary_world.shape[0]),
        )

    fig.suptitle(style.main_title, fontsize=14)

    if output_path is not None:
        save_start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        _log_timing(verbose_timing, "save figure", save_start, dpi=OUTPUT_DPI)
    if show_figure:
        show_start = time.perf_counter()
        plt.show()
        _log_timing(verbose_timing, "show figure", show_start)
    else:
        close_start = time.perf_counter()
        plt.close(fig)
        _log_timing(verbose_timing, "close figure", close_start)
    _log_timing(verbose_timing, "build figure total", total_start)
    return box, solutions, box_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a mode3 dexterous-workspace illustration.")
    parser.add_argument("--config", default="config/csm_cfg_3mm.yaml", help="CSM config path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output figure path.")
    parser.add_argument("--load-box-info", help="Optional JSON path exported by the translation script; reuses exactly the same box.")
    parser.add_argument("--box-size-mm", default="50,50,40", help="Operation box size in millimeters: sx,sy,sz.")
    parser.add_argument("--top-margin-mm", default=DEFAULT_TOP_MARGIN_MM, type=float, help="Top clearance to workspace roof.")
    parser.add_argument("--hide", action="store_true", help="Render without opening a window.")
    parser.add_argument("--timing", action="store_true", help="Print per-stage timing diagnostics.")
    args = parser.parse_args()

    main_start = time.perf_counter()
    csm = CSM.from_config(Path(args.config))
    _log_timing(args.timing, "load config", main_start, config=args.config)
    requested_box_size_mm = _parse_box_size_mm(args.box_size_mm)
    loaded_box = None
    loaded_box_scale = None
    if args.load_box_info:
        loaded_box, loaded_box_scale = _load_box_info(Path(args.load_box_info), csm=csm)
    box, solutions, box_scale = build_figure(
        csm=csm,
        box_size_mm=requested_box_size_mm,
        top_margin_mm=float(args.top_margin_mm),
        output_path=Path(args.output),
        show_figure=not args.hide,
        box_override=loaded_box,
        box_scale_override=loaded_box_scale,
        style=DEFAULT_STYLE,
        verbose_timing=args.timing,
    )
    _log_timing(args.timing, "main total", main_start)
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
