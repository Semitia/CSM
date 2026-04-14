"""
Kinematics helpers for the local dexterous-workspace module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from csm.model import CSM


DEFAULT_R1_MINUS_BY_CONFIG_M = {
    "3mm": 0.0054038,
}


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _csm_segment_transform(theta_t: float, length_t: float, delta_t: float) -> np.ndarray:
    """
    Mirror the transform convention used by `CSM.get_trans_mat`.
    """
    R_b_1 = np.array(
        [
            [0.0, math.cos(delta_t), math.sin(delta_t)],
            [0.0, -math.sin(delta_t), math.cos(delta_t)],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    R_1_2 = np.array(
        [
            [math.cos(theta_t), -math.sin(theta_t), 0.0],
            [math.sin(theta_t), math.cos(theta_t), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    R_2_e = np.array(
        [
            [0.0, 0.0, 1.0],
            [math.cos(delta_t), -math.sin(delta_t), 0.0],
            [math.sin(delta_t), math.cos(delta_t), 0.0],
        ],
        dtype=float,
    )
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_b_1 @ R_1_2 @ R_2_e
    if abs(theta_t) < 1e-8:
        T[:3, 3] = [0.0, 0.0, length_t]
    else:
        T[:3, 3] = [
            length_t * math.cos(delta_t) * (1.0 - math.cos(theta_t)) / theta_t,
            length_t * math.sin(delta_t) * (math.cos(theta_t) - 1.0) / theta_t,
            length_t * math.sin(theta_t) / theta_t,
        ]
    return T


@dataclass(frozen=True)
class DexterousMode3State:
    phi: float
    theta1: float
    L1: float
    delta1: float
    theta2: float
    delta2: float


@dataclass(frozen=True)
class DexterousParameters:
    L10_m: float
    L20_m: float
    Lr_m: float
    Lg_m: float
    theta1_plus: float
    theta2_plus: float
    r1_minus_m: float

    @property
    def L10_mm(self) -> float:
        return 1000.0 * self.L10_m

    @property
    def L20_mm(self) -> float:
        return 1000.0 * self.L20_m

    @property
    def Lr_mm(self) -> float:
        return 1000.0 * self.Lr_m

    @property
    def Lg_mm(self) -> float:
        return 1000.0 * self.Lg_m

    @property
    def r1_minus_mm(self) -> float:
        return 1000.0 * self.r1_minus_m

    @classmethod
    def from_csm(cls, csm: CSM, config: str = "3mm") -> "DexterousParameters":
        if config != "3mm":
            raise ValueError(f"Only 3mm config is supported for now, got {config!r}.")
        if csm.L_s0 > 1e-12:
            raise ValueError("Dexterous mode only supports the 3mm configuration without base insertion.")
        r1_min = csm.r1_min
        if r1_min is None:
            r1_min = csm.ri_min
        if r1_min is None:
            r1_min = DEFAULT_R1_MINUS_BY_CONFIG_M.get(config)
        if r1_min is None:
            raise ValueError("r1_min (or ri_min) is required to build dexterous-workspace parameters.")
        return cls(
            L10_m=float(csm.L_10),
            L20_m=float(csm.L_20),
            Lr_m=float(csm.L_r0),
            Lg_m=float(csm.L_tool),
            theta1_plus=float(csm.theta1_limit),
            theta2_plus=float(csm.theta2_limit),
            r1_minus_m=float(r1_min),
        )


def clone_csm(csm: CSM) -> CSM:
    cloned = CSM(
        L_10=float(csm.L_10),
        L_20=float(csm.L_20),
        L_r0=float(csm.L_r0),
        L_s0=float(csm.L_s0),
        L_tool=float(csm.L_tool),
        theta1_max=float(csm.theta1_max),
        theta2_max=float(csm.theta2_max),
        delta_t=float(csm.delta_t),
        ri_min=None if csm.ri_min is None else float(csm.ri_min),
        r1_min=None if csm.r1_min is None else float(csm.r1_min),
        r2_min=None if csm.r2_min is None else float(csm.r2_min),
    )
    cloned.set_state(
        mode=int(csm.mode),
        phi=float(csm.phi),
        L1=float(csm.L1),
        L2=float(csm.L2),
        Lr=float(csm.Lr),
        Ls=float(csm.Ls),
        theta_1=float(csm.theta_1),
        theta_2=float(csm.theta_2),
        delta_1=float(csm.delta_1),
        delta_2=float(csm.delta_2),
    )
    return cloned


def mode3_state_from_csm(csm: CSM) -> DexterousMode3State:
    if int(csm.mode) != 3:
        raise ValueError(f"Dexterous workspace currently only supports CSM mode 3, got mode={csm.mode}.")
    return DexterousMode3State(
        phi=float(csm.phi),
        theta1=float(csm.theta_1),
        L1=float(csm.L1),
        delta1=float(csm.delta_1),
        theta2=float(csm.theta_2),
        delta2=float(csm.delta_2),
    )


def make_mode3_display_csm(csm: CSM, state: DexterousMode3State) -> CSM:
    display_csm = clone_csm(csm)
    display_csm.set_state(
        mode=3,
        phi=state.phi,
        L1=state.L1,
        L2=float(display_csm.L_20),
        Lr=float(display_csm.L_r0),
        Ls=0.0,
        theta_1=state.theta1,
        theta_2=state.theta2,
        delta_1=state.delta1,
        delta_2=state.delta2,
    )
    return display_csm


def _line_segment_length(length_m: float, theta: float) -> float:
    if abs(theta) < 1e-8:
        return 0.5 * length_m
    return float(length_m * math.tan(theta / 2.0) / theta)


def analytic_fk_mode3(params: DexterousParameters, state: DexterousMode3State) -> tuple[np.ndarray, np.ndarray]:
    """
    FK using the CI-1 style chain, but expressed in the project's meter-based
    mode-3 semantics.
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = rot_z(state.phi)

    T = T @ _csm_segment_transform(state.theta1, state.L1, state.delta1)
    T_r = np.eye(4, dtype=float)
    T_r[2, 3] = params.Lr_m
    T = T @ T_r
    T = T @ _csm_segment_transform(state.theta2, params.L20_m, state.delta2)
    tool_axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=float)
    tool_pos = T[:3, 3] + tool_axis * params.Lg_m
    return tool_pos.copy(), tool_axis.copy()


def segment_fk(length_m: float, theta: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(theta) < 1e-8:
        return np.array([0.0, 0.0, length_m], dtype=float), np.eye(3, dtype=float)
    p = (length_m / theta) * np.array(
        [
            math.cos(delta) * (1.0 - math.cos(theta)),
            math.sin(delta) * (1.0 - math.cos(theta)),
            math.sin(theta),
        ],
        dtype=float,
    )
    R = rot_z(delta) @ rot_y(theta) @ rot_z(-delta)
    return p, R


def state_to_target_pose(csm: CSM, state: DexterousMode3State) -> np.ndarray:
    display_csm = make_mode3_display_csm(csm, state)
    return np.asarray(display_csm.pose, dtype=float).copy()


def asdict_state(state: DexterousMode3State) -> dict[str, Any]:
    return {
        "phi": state.phi,
        "theta1": state.theta1,
        "L1": state.L1,
        "delta1": state.delta1,
        "theta2": state.theta2,
        "delta2": state.delta2,
    }
