"""
Fallback Jacobian-based local probe builder for dexterous workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from csm.model import CSM
from csm.utils import axis_angle_from_vectors, normalize_vector

from .kinematics import DexterousMode3State, clone_csm, make_mode3_display_csm, mode3_state_from_csm


@dataclass
class FallbackScanResult:
    q_seed: DexterousMode3State | None
    directions_world: np.ndarray
    reachable_mask: np.ndarray
    solved_states: list[DexterousMode3State | None]


def fibonacci_sphere(samples: int) -> np.ndarray:
    if samples <= 0:
        return np.zeros((0, 3), dtype=float)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = np.zeros((samples, 3), dtype=float)
    for i in range(samples):
        y = 1.0 - (2.0 * i + 1.0) / samples
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        points[i] = [math.cos(theta) * radius, math.sin(theta) * radius, y]
    return points


def _seed_states(csm: CSM, position_xyz: np.ndarray | None = None) -> list[DexterousMode3State]:
    theta1_list = [0.15, 0.45, min(csm.theta1_limit * 0.85, 0.85)]
    theta2_list = [0.15, 0.55, min(csm.theta2_limit * 0.85, 1.1)]
    L1_list = [max(0.25 * csm.L_10, 1e-5), 0.6 * csm.L_10, 0.95 * csm.L_10]
    phi_list = [0.0]
    if position_xyz is not None:
        target = np.asarray(position_xyz, dtype=float).reshape(3)
        radial_xy = float(np.linalg.norm(target[:2]))
        if radial_xy > 1e-9:
            phi0 = float(math.atan2(target[1], target[0]))
            # Try the target azimuth first, then nearby yaw guesses, so the
            # position-only IK does not start from a seed facing the wrong
            # quadrant for off-axis box points.
            phi_list = [
                phi0,
                phi0 + 0.45 * math.pi,
                phi0 - 0.45 * math.pi,
                phi0 + 0.25 * math.pi,
                phi0 - 0.25 * math.pi,
                phi0 + math.pi,
                0.0,
            ]
            deduped: list[float] = []
            for phi in phi_list:
                wrapped = float(math.atan2(math.sin(phi), math.cos(phi)))
                if any(abs(wrapped - old) < 1e-9 for old in deduped):
                    continue
                deduped.append(wrapped)
            phi_list = deduped
    seeds: list[DexterousMode3State] = []
    for phi in phi_list:
        for theta1 in theta1_list:
            for theta2 in theta2_list:
                for L1 in L1_list:
                    theta1_eff = min(theta1, max(1e-6, L1 / max(csm.ri_min or 1.0, 1e-8)))
                    seeds.append(
                        DexterousMode3State(
                            phi=float(phi),
                            theta1=float(theta1_eff),
                            L1=float(L1),
                            delta1=0.0,
                            theta2=float(theta2),
                            delta2=0.0,
                        )
                    )
    if int(csm.mode) == 3:
        seeds.insert(0, mode3_state_from_csm(csm))
    return seeds


def _solve_target(
    csm: CSM,
    target_pose: np.ndarray,
    *,
    require_orientation: bool,
    max_steps: int = 450,
    pos_tol: float = 8e-4,
    ori_tol: float = 0.08,
    v_gain: float = 5.0,
    w_gain: float = 4.0,
) -> tuple[bool, float, float]:
    csm.target_pose = np.asarray(target_pose, dtype=float).copy()
    for _ in range(max_steps):
        pos_error_vec = csm.target_pose[:3] - csm.pose[:3]
        pos_error = float(np.linalg.norm(pos_error_vec))
        if require_orientation:
            _, ori_error = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
        else:
            ori_error = 0.0

        if pos_error < pos_tol and (not require_orientation or ori_error < ori_tol):
            return int(csm.mode) == 3, pos_error, ori_error

        linear_velocity = normalize_vector(pos_error_vec) * min(v_gain * pos_error, 0.06)
        if require_orientation:
            axis_hat, theta = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
            angular_velocity = axis_hat * min(w_gain * theta, 4.0)
        else:
            angular_velocity = np.zeros(3, dtype=float)

        csm.get_dot_PHI(linear_velocity, angular_velocity)
        csm.step()
        csm.check_transition()
        if int(csm.mode) != 3:
            return False, pos_error, ori_error
    return False, pos_error, ori_error


def bootstrap_position_reachable_state(csm: CSM, position_xyz: np.ndarray) -> DexterousMode3State | None:
    target_pos = np.asarray(position_xyz, dtype=float)
    for seed in _seed_states(csm, target_pos):
        worker = make_mode3_display_csm(csm, seed)
        target_pose = np.concatenate([target_pos, worker.pose[3:]], dtype=float)
        ok, pos_err, _ = _solve_target(worker, target_pose, require_orientation=False)
        if ok and pos_err < 1.2e-3:
            return mode3_state_from_csm(worker)
    return None


def scan_directions(
    csm: CSM,
    position_xyz: np.ndarray,
    *,
    q_seed: DexterousMode3State | None = None,
    n_directions: int = 120,
    pos_tol: float = 1.2e-3,
    ori_tol: float = 0.16,
) -> FallbackScanResult:
    directions = fibonacci_sphere(n_directions)
    if q_seed is None:
        q_seed = bootstrap_position_reachable_state(csm, position_xyz)
    reachable_mask = np.zeros(n_directions, dtype=bool)
    solved_states: list[DexterousMode3State | None] = [None] * n_directions
    if q_seed is None:
        return FallbackScanResult(q_seed=None, directions_world=directions, reachable_mask=reachable_mask, solved_states=solved_states)

    for idx, direction in enumerate(directions):
        worker = make_mode3_display_csm(csm, q_seed)
        target_pose = np.concatenate([np.asarray(position_xyz, dtype=float), direction], dtype=float)
        ok, pos_err, ori_err = _solve_target(worker, target_pose, require_orientation=True, pos_tol=pos_tol, ori_tol=ori_tol)
        if ok and pos_err < pos_tol and ori_err < ori_tol:
            reachable_mask[idx] = True
            solved_states[idx] = mode3_state_from_csm(worker)
    return FallbackScanResult(
        q_seed=q_seed,
        directions_world=directions,
        reachable_mask=reachable_mask,
        solved_states=solved_states,
    )
