"""
Public orchestration layer for local dexterous-workspace probes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csm.model import CSM

from .analytic import AnalyticDexterousWorkspace
from .fallback import FallbackScanResult, bootstrap_position_reachable_state, scan_directions
from .kinematics import DexterousMode3State, DexterousParameters
from .plotting import DexterousPlotOptions, plot_dexterous_probe, render_dexterous_figure


@dataclass
class DexterousProbe:
    position_xyz: np.ndarray
    sphere_radius_m: float
    feasible_directions_world: np.ndarray
    feasible_directions_sym: np.ndarray
    type1_boundary_sym: np.ndarray
    type2_boundary_sym: np.ndarray
    type1_boundary_world: np.ndarray
    type2_boundary_world: np.ndarray
    gamma: float
    display_state: DexterousMode3State | None
    status: str
    method: str
    label: str | None = None
    cap_center_world: np.ndarray | None = None
    cap_angular_radius: float | None = None
    cap_fit_error: float | None = None
    fallback_scan: FallbackScanResult | None = None
    debug_data: dict | None = None
    boundary_families_sym: list | None = None


def _fit_spherical_cap(directions_world: np.ndarray) -> tuple[np.ndarray | None, float | None, float | None]:
    dirs = np.asarray(directions_world, dtype=float)
    if dirs.shape[0] < 3:
        return None, None, None
    center = np.mean(dirs, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm < 1e-10:
        return None, None, None
    center = center / center_norm
    angles = np.arccos(np.clip(dirs @ center, -1.0, 1.0))
    angular_radius = float(np.max(angles))
    fit_error = float(np.std(angles))
    return center, angular_radius, fit_error


def _build_debug_data(
    *,
    probe_position: np.ndarray,
    position_sym: np.ndarray | None,
    feasible_world: np.ndarray,
    feasible_sym: np.ndarray,
    type1_sym: np.ndarray,
    type2_sym: np.ndarray,
    type1_world: np.ndarray,
    type2_world: np.ndarray,
    gamma: float,
    display_state: DexterousMode3State | None,
    status: str,
    method: str,
    cap_center_world: np.ndarray | None,
    cap_angular_radius: float | None,
    cap_fit_error: float | None,
    render_use_cap: bool,
    fallback_scan_result: FallbackScanResult | None,
    boundary_families_sym,
    analytic_line_coefficients: tuple[float, float, float] | None,
) -> dict:
    p_sym = None if position_sym is None else np.asarray(position_sym, dtype=float).copy()
    is_boundary_only = bool(len(feasible_sym) == 0 and len(type1_sym) > 0 and len(type2_sym) == 0)
    axis_degenerate = bool(
        p_sym is not None
        and abs(float(p_sym[0])) <= 1e-8
        and is_boundary_only
    )
    debug_data = {
        "position_xyz": np.asarray(probe_position, dtype=float).copy(),
        "position_sym": p_sym,
        "feasible_count_world": int(len(feasible_world)),
        "feasible_count_sym": int(len(feasible_sym)),
        "type1_boundary_count_sym": int(len(type1_sym)),
        "type2_boundary_count_sym": int(len(type2_sym)),
        "type1_boundary_count_world": int(len(type1_world)),
        "type2_boundary_count_world": int(len(type2_world)),
        "gamma_rad": float(gamma),
        "gamma_deg": float(np.degrees(gamma)),
        "status": status,
        "method": method,
        "display_state": None if display_state is None else {
            "phi": float(display_state.phi),
            "theta1": float(display_state.theta1),
            "L1": float(display_state.L1),
            "delta1": float(display_state.delta1),
            "theta2": float(display_state.theta2),
            "delta2": float(display_state.delta2),
        },
        "cap_center_world": None if cap_center_world is None else np.asarray(cap_center_world, dtype=float).copy(),
        "cap_angular_radius_rad": None if cap_angular_radius is None else float(cap_angular_radius),
        "cap_angular_radius_deg": None if cap_angular_radius is None else float(np.degrees(cap_angular_radius)),
        "cap_fit_error_rad": None if cap_fit_error is None else float(cap_fit_error),
        "cap_fit_error_deg": None if cap_fit_error is None else float(np.degrees(cap_fit_error)),
        "fit_used_as_cap": bool(render_use_cap),
        "is_boundary_only": is_boundary_only,
        "axis_degenerate_case": axis_degenerate,
        "degeneracy_reason": (
            "symmetry-axis boundary-only case: feasible set collapsed from 2D region to 1D boundary"
            if axis_degenerate
            else None
        ),
        "analytic_line_coefficients": None if analytic_line_coefficients is None else {
            "A": float(analytic_line_coefficients[0]),
            "B": float(analytic_line_coefficients[1]),
            "C": float(analytic_line_coefficients[2]),
        },
        "fallback": None,
        "boundary_families": [
            {
                "family_id": fam.family_id,
                "primitive_type": fam.primitive_type,
                "fit_error": float(fam.fit_error),
                "point_count": int(len(fam.points_sym)),
                "scores": fam.fit_meta.get("scores", {}) if isinstance(fam.fit_meta, dict) else {},
            }
            for fam in (boundary_families_sym or [])
            if getattr(fam, "points_sym", np.zeros((0, 2))).size
        ],
    }
    if fallback_scan_result is not None:
        debug_data["fallback"] = {
            "direction_samples": int(len(fallback_scan_result.directions_world)),
            "reachable_count": int(np.count_nonzero(fallback_scan_result.reachable_mask)),
            "has_q_seed": fallback_scan_result.q_seed is not None,
        }
    return debug_data


def build_dexterous_probe(
    position_xyz,
    *,
    csm: CSM,
    config: str = "3mm",
    method: str = "analytic",
    validate_with_fallback: bool = False,
    fallback_direction_samples: int = 120,
    sphere_radius_m: float = 0.012,
    label: str | None = None,
) -> DexterousProbe:
    if config != "3mm":
        raise ValueError(f"Only the 3mm configuration is supported, got {config!r}.")
    params = DexterousParameters.from_csm(csm, config=config)
    probe_position = np.asarray(position_xyz, dtype=float).reshape(3)
    analytic = AnalyticDexterousWorkspace(params)

    display_state = bootstrap_position_reachable_state(csm, probe_position)
    feasible_world = np.zeros((0, 3), dtype=float)
    feasible_sym = np.zeros((0, 2), dtype=float)
    type1_world = np.zeros((0, 3), dtype=float)
    type2_world = np.zeros((0, 3), dtype=float)
    type1_sym = np.zeros((0, 2), dtype=float)
    type2_sym = np.zeros((0, 2), dtype=float)
    gamma = 0.0
    status = "unreachable"
    fallback_scan_result = None
    boundary_families_sym = []
    position_sym = None
    analytic_line_coefficients = None

    if method == "analytic":
        position_sym, _, _ = analytic.to_symmetry_frame(probe_position, np.array([0.0, 0.0, 1.0], dtype=float))
        region = analytic.build_region(probe_position)
        feasible_world = region.feasible_area_world
        feasible_sym = region.feasible_area_sym
        type1_sym = region.type1_sym
        type2_sym = region.type2_sym
        type1_world = region.type1_world
        type2_world = region.type2_world
        gamma = region.gamma
        boundary_families_sym = region.boundary_families_sym
        analytic_line_coefficients = analytic.ci1_line_coefficients(probe_position, params.theta2_plus)
        status = "analytic_ok" if feasible_world.size else "analytic_empty"
    elif method == "fallback":
        fallback_scan_result = scan_directions(csm, probe_position, n_directions=fallback_direction_samples)
        feasible_world = fallback_scan_result.directions_world[fallback_scan_result.reachable_mask]
        feasible_sym = np.zeros((0, 2), dtype=float)
        display_state = fallback_scan_result.q_seed
        status = "fallback_ok" if feasible_world.size else "fallback_empty"
    else:
        raise ValueError(f"Unsupported method {method!r}.")

    if validate_with_fallback or (method == "analytic" and feasible_world.size == 0):
        fallback_scan_result = scan_directions(
            csm,
            probe_position,
            q_seed=display_state,
            n_directions=fallback_direction_samples,
        )
        if method != "fallback" and feasible_world.size == 0 and np.any(fallback_scan_result.reachable_mask):
            feasible_world = fallback_scan_result.directions_world[fallback_scan_result.reachable_mask]
            status = "analytic_empty_fallback_ok"
            display_state = fallback_scan_result.q_seed

    cap_center_world, cap_angular_radius, cap_fit_error = _fit_spherical_cap(feasible_world)
    render_use_cap = bool(
        cap_center_world is not None
        and cap_angular_radius is not None
        and cap_fit_error is not None
        and np.degrees(cap_fit_error) <= 4.0
    )

    return DexterousProbe(
        position_xyz=probe_position,
        sphere_radius_m=float(sphere_radius_m),
        feasible_directions_world=np.asarray(feasible_world, dtype=float),
        feasible_directions_sym=np.asarray(feasible_sym, dtype=float),
        type1_boundary_sym=np.asarray(type1_sym, dtype=float),
        type2_boundary_sym=np.asarray(type2_sym, dtype=float),
        type1_boundary_world=np.asarray(type1_world, dtype=float),
        type2_boundary_world=np.asarray(type2_world, dtype=float),
        gamma=float(gamma),
        display_state=display_state,
        status=status,
        method=method,
        label=label,
        cap_center_world=cap_center_world,
        cap_angular_radius=cap_angular_radius,
        cap_fit_error=cap_fit_error,
        fallback_scan=fallback_scan_result,
        boundary_families_sym=boundary_families_sym,
        debug_data=_build_debug_data(
            probe_position=probe_position,
            position_sym=position_sym,
            feasible_world=feasible_world,
            feasible_sym=feasible_sym,
            type1_sym=type1_sym,
            type2_sym=type2_sym,
            type1_world=type1_world,
            type2_world=type2_world,
            gamma=gamma,
            display_state=display_state,
            status=status,
            method=method,
            cap_center_world=cap_center_world,
            cap_angular_radius=cap_angular_radius,
            cap_fit_error=cap_fit_error,
            render_use_cap=render_use_cap,
            fallback_scan_result=fallback_scan_result,
            boundary_families_sym=boundary_families_sym,
            analytic_line_coefficients=analytic_line_coefficients,
        ),
    )
