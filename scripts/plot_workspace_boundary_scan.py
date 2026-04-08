"""
Module: plot_workspace_boundary_scan.py
Description: Build workspace profiles by directly scanning boundary motions instead
of extracting contours from dense point clouds.

Current status:
- mode 1/2/3/4 implemented
- mode 0 reserved in the framework
"""
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from csm import CSM


CONFIG_NAME = "csm_cfg_0_tool.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
PLOT_MODES = [3, 4]

FIGSIZE = None
REVOLVE_SAMPLES = 30
REVOLVE_MAX_AXIAL_SAMPLES = 96
RENDER_3D_MODE = "fast_surface"  # "fast_surface" | "trisurf"
SAVE_DEBUG_FIGURES = True
MODE_COLORS = {
    1: "#9FD4EA",
    2: "#F0E0AA",
    3: "#E9A3A7",
    4: "#CFCFCF",
}
UNREACHABLE_COLOR = "#D79A6B"
REACH_ALPHA = 0.34
UNREACHABLE_ALPHA = 0.40
OUTPUT_PATH = Path("./data/plot_workspace_boundary_scan.png")
DEBUG_OUTPUT_DIR = Path("./data/plot_workspace_boundary_scan_debug")
SHOW_FIGURE = True


def _has_interactive_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@dataclass
class WorkspaceProfile:
    mode: int
    inner_segments: list
    outer_segments: list
    outer_open_curve_rz: np.ndarray
    unreachable_open_curves_rz: list
    closed_profile_rz: np.ndarray
    unreachable_closed_profiles_rz: list
    debug_data: dict | None = None


@dataclass
class BoundaryPrimitive:
    name: str
    points_rz: np.ndarray


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


def _close_curve_to_axis(curve_rz):
    curve_rz = np.asarray(curve_rz, dtype=float)
    if curve_rz.ndim != 2 or curve_rz.shape[0] < 2:
        return curve_rz
    start = curve_rz[0]
    end = curve_rz[-1]
    closure = np.asarray(
        [
            [0.0, float(end[1])],
            [0.0, float(start[1])],
        ],
        dtype=float,
    )
    return np.vstack((curve_rz, closure))


def _symmetric_fill_polygon(open_curve_rz):
    open_curve_rz = np.asarray(open_curve_rz, dtype=float)
    mirrored = np.column_stack((-open_curve_rz[:, 0], open_curve_rz[:, 1]))
    return np.vstack((mirrored[::-1], open_curve_rz[1:]))

def _cross_2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(p0, p1, q0, q1, eps=1e-9):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    r = p1 - p0
    s = q1 - q0
    denom = _cross_2d(r, s)
    qmp = q0 - p0

    if abs(denom) < eps:
        return None

    t = _cross_2d(qmp, s) / denom
    u = _cross_2d(qmp, r) / denom
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        point = p0 + np.clip(t, 0.0, 1.0) * r
        return float(t), float(u), point
    return None


def _first_polyline_intersection(curve_a, curve_b):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    if curve_a.shape[0] < 2 or curve_b.shape[0] < 2:
        return None

    best = None
    best_progress = None
    for i in range(curve_a.shape[0] - 1):
        p0 = curve_a[i]
        p1 = curve_a[i + 1]
        for j in range(curve_b.shape[0] - 1):
            q0 = curve_b[j]
            q1 = curve_b[j + 1]
            result = _segment_intersection(p0, p1, q0, q1)
            if result is None:
                continue
            t, u, point = result
            progress = i + t
            if best is None or progress < best_progress:
                best = {
                    "idx_a": i,
                    "idx_b": j,
                    "t_a": t,
                    "t_b": u,
                    "point": np.asarray(point, dtype=float),
                }
                best_progress = progress
    return best


def _polyline_prefix(curve, split):
    curve = np.asarray(curve, dtype=float)
    if split is None:
        return curve.copy()
    idx = int(split["idx_a"])
    point = np.asarray(split["point"], dtype=float)
    prefix = curve[:idx + 1].copy()
    if prefix.shape[0] == 0 or not np.allclose(prefix[-1], point, atol=1e-9):
        prefix = np.vstack((prefix, point))
    else:
        prefix[-1] = point
    return prefix


def _polyline_suffix(curve, split, which="b"):
    curve = np.asarray(curve, dtype=float)
    if split is None:
        return curve.copy()
    idx_key = "idx_b" if which == "b" else "idx_a"
    idx = int(split[idx_key])
    point = np.asarray(split["point"], dtype=float)
    suffix = curve[idx + 1:].copy()
    if suffix.shape[0] == 0:
        return point[np.newaxis, :]
    if not np.allclose(suffix[0], point, atol=1e-9):
        suffix = np.vstack((point, suffix))
    else:
        suffix[0] = point
    return suffix


def _concat_curve_segments(segments):
    cleaned = []
    for segment in segments:
        arr = np.asarray(segment, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        if not cleaned:
            cleaned.append(arr.copy())
            continue
        prev = cleaned[-1]
        if np.allclose(prev[-1], arr[0], atol=1e-9):
            cleaned.append(arr[1:].copy())
        else:
            cleaned.append(arr.copy())
    if not cleaned:
        return np.empty((0, 2), dtype=float)
    return np.vstack(cleaned)


def _sample_mode1_inner_curve(csm, length_samples=240):
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
    return _sample_state_curve(csm, inner_states)


def _sample_mode2_theta_sweep_curve(csm, Lr, angle_samples=240):
    states = []
    for theta_2 in np.linspace(csm.theta2_limit, 0.0, angle_samples):
        states.append(
            {
                "mode": 2,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": Lr,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode2_theta_curve(csm, Lr, theta_start, theta_end, angle_samples=240):
    states = []
    for theta_2 in np.linspace(theta_start, theta_end, angle_samples):
        states.append(
            {
                "mode": 2,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": Lr,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode2_lr_curve(csm, theta_2, lr_start, lr_end, length_samples=240):
    start_point = _sample_state_curve(
        csm,
        [
            {
                "mode": 2,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": lr_start,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        ],
    )[0]
    end_point = _sample_state_curve(
        csm,
        [
            {
                "mode": 2,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": lr_end,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        ],
    )[0]
    radii = np.linspace(float(start_point[0]), float(end_point[0]), length_samples)
    heights = np.linspace(float(start_point[1]), float(end_point[1]), length_samples)
    return np.column_stack((radii, heights))


def _sample_mode3_theta2_curve(csm, L1, theta_1, theta_start, theta_end, angle_samples=240):
    states = []
    for theta_2 in np.linspace(theta_start, theta_end, angle_samples):
        states.append(
            {
                "mode": 3,
                "L1": L1,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": 0.0,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode3_theta1_curve(csm, L1, theta1_start, theta1_end, theta_2, angle_samples=240):
    states = []
    for theta_1 in np.linspace(theta1_start, theta1_end, angle_samples):
        states.append(
            {
                "mode": 3,
                "L1": L1,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": 0.0,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode3_l1_curve(csm, theta_2, l1_start, l1_end, length_samples=240):
    states = []
    for L1 in np.linspace(l1_start, l1_end, length_samples):
        states.append(
            {
                "mode": 3,
                "L1": L1,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": 0.0,
                "theta_1": min(csm.kappa_10 * L1, csm.theta1_limit),
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode4_theta1_curve(csm, Ls, theta1_start, theta1_end, theta_2, angle_samples=240):
    states = []
    for theta_1 in np.linspace(theta1_start, theta1_end, angle_samples):
        states.append(
            {
                "mode": 4,
                "L1": csm.L_10,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": Ls,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode4_theta2_curve(csm, Ls, theta_1, theta_start, theta_end, angle_samples=240):
    states = []
    for theta_2 in np.linspace(theta_start, theta_end, angle_samples):
        states.append(
            {
                "mode": 4,
                "L1": csm.L_10,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": Ls,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode4_ls_curve(csm, theta_1, theta_2, ls_start, ls_end, length_samples=240):
    states = []
    for Ls in np.linspace(ls_start, ls_end, length_samples):
        states.append(
            {
                "mode": 4,
                "L1": csm.L_10,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": Ls,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def build_mode1_profile(csm, length_samples=240, angle_samples=240):
    inner_curve = _sample_mode1_inner_curve(csm, length_samples=length_samples)
    outer_curve = _sample_mode2_theta_sweep_curve(csm, Lr=0.0, angle_samples=angle_samples)
    axis_closure = _build_axis_closure(outer_curve[-1], inner_curve[0])
    closed_profile = np.vstack((inner_curve, outer_curve[1:], axis_closure[1:]))

    return WorkspaceProfile(
        mode=1,
        inner_segments=[inner_curve],
        outer_segments=[outer_curve],
        outer_open_curve_rz=outer_curve,
        unreachable_open_curves_rz=[],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[],
        debug_data=None,
    )


def build_mode2_profile(csm, length_samples=240, angle_samples=240):
    theta_break = min(csm.theta2_limit, 0.5 * np.pi)

    primitives = {
        "tau0": BoundaryPrimitive(
            "tau0",
            _sample_mode2_theta_curve(csm, Lr=0.0, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples),
        ),
        "tau1": BoundaryPrimitive(
            "tau1",
            _sample_mode2_theta_curve(csm, Lr=csm.L_r0, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples),
        ),
        "tau2": BoundaryPrimitive(
            "tau2",
            _sample_mode2_lr_curve(csm, theta_2=csm.theta2_limit, lr_start=csm.L_r0, lr_end=0.0, length_samples=length_samples),
        ),
    }

    if csm.theta2_limit > 0.5 * np.pi + 1e-9:
        outer_segments = [
            _sample_mode2_theta_curve(csm, Lr=0.0, theta_start=csm.theta2_limit, theta_end=theta_break, angle_samples=angle_samples),
            _sample_mode2_lr_curve(csm, theta_2=theta_break, lr_start=0.0, lr_end=csm.L_r0, length_samples=length_samples),
            _sample_mode2_theta_curve(csm, Lr=csm.L_r0, theta_start=theta_break, theta_end=0.0, angle_samples=angle_samples),
        ]
    else:
        outer_segments = [
            _sample_mode2_lr_curve(csm, theta_2=csm.theta2_limit, lr_start=0.0, lr_end=csm.L_r0, length_samples=length_samples),
            _sample_mode2_theta_curve(csm, Lr=csm.L_r0, theta_start=csm.theta2_limit, theta_end=0.0, angle_samples=angle_samples),
        ]

    hit_tau1 = _first_polyline_intersection(primitives["tau0"].points_rz, primitives["tau1"].points_rz)
    hit_tau2 = _first_polyline_intersection(primitives["tau0"].points_rz, primitives["tau2"].points_rz)
    candidates = []
    if hit_tau1 is not None:
        candidates.append(("tau1", hit_tau1))
    if hit_tau2 is not None:
        candidates.append(("tau2", hit_tau2))

    chosen_hit_name = None
    chosen_hit = None
    if candidates:
        hit_name, hit = min(candidates, key=lambda item: item[1]["idx_a"] + item[1]["t_a"])
        chosen_hit_name = hit_name
        chosen_hit = hit
        tau0_prefix = _polyline_prefix(primitives["tau0"].points_rz, hit)
        if hit_name == "tau1":
            tau1_suffix = _polyline_suffix(primitives["tau1"].points_rz, hit, which="b")
            inner_segments = [tau0_prefix, tau1_suffix, primitives["tau2"].points_rz]
        else:
            tau2_suffix = _polyline_suffix(primitives["tau2"].points_rz, hit, which="b")
            inner_segments = [tau0_prefix, tau2_suffix]
        inner_curve = _concat_curve_segments(inner_segments)
    else:
        inner_curve = primitives["tau0"].points_rz.copy()

    axis_connector = _build_axis_closure(inner_curve[-1], outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))

    return WorkspaceProfile(
        mode=2,
        inner_segments=[inner_curve],
        outer_segments=outer_segments,
        outer_open_curve_rz=_concat_curve_segments(outer_segments),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "primitives": {name: primitive.points_rz.copy() for name, primitive in primitives.items()},
            "outer_segments": [segment.copy() for segment in outer_segments],
            "inner_segments": [segment.copy() for segment in inner_segments] if candidates else [inner_curve.copy()],
            "hit_tau1": None if hit_tau1 is None else dict(hit_tau1),
            "hit_tau2": None if hit_tau2 is None else dict(hit_tau2),
            "chosen_hit_name": chosen_hit_name,
            "chosen_hit": None if chosen_hit is None else dict(chosen_hit),
            "theta_break": float(theta_break),
        },
    )


def build_mode3_profile(csm, length_samples=240, angle_samples=240):
    primitives = {
        "tau0": BoundaryPrimitive(
            "tau0",
            _sample_mode3_theta2_curve(
                csm,
                L1=0.0,
                theta_1=0.0,
                theta_start=0.0,
                theta_end=csm.theta2_limit,
                angle_samples=angle_samples,
            ),
        ),
        "tau1": BoundaryPrimitive(
            "tau1",
            _sample_mode3_l1_curve(
                csm,
                theta_2=csm.theta2_limit,
                l1_start=0.0,
                l1_end=csm.L_10,
                length_samples=length_samples,
            ),
        ),
        "tau2": BoundaryPrimitive(
            "tau2",
            _sample_mode3_theta2_curve(
                csm,
                L1=csm.L_10,
                theta_1=csm.theta1_limit,
                theta_start=csm.theta2_limit,
                theta_end=0.0,
                angle_samples=angle_samples,
            ),
        ),
        "tau3": BoundaryPrimitive(
            "tau3",
            _sample_mode3_theta1_curve(
                csm,
                L1=csm.L_10,
                theta1_start=csm.theta1_limit,
                theta1_end=0.0,
                theta_2=0.0,
                angle_samples=angle_samples,
            ),
        ),
    }

    inner_segments = [
        primitives["tau0"].points_rz,
        primitives["tau1"].points_rz,
    ]
    outer_segments = [
        primitives["tau2"].points_rz,
        primitives["tau3"].points_rz,
    ]

    inner_curve = _concat_curve_segments(inner_segments)
    axis_connector = _build_axis_closure(inner_curve[-1], outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))

    return WorkspaceProfile(
        mode=3,
        inner_segments=[segment.copy() for segment in inner_segments],
        outer_segments=[segment.copy() for segment in outer_segments],
        outer_open_curve_rz=_concat_curve_segments(outer_segments),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "primitives": {name: primitive.points_rz.copy() for name, primitive in primitives.items()},
            "outer_segments": [segment.copy() for segment in outer_segments],
            "inner_segments": [segment.copy() for segment in inner_segments],
        },
    )


def build_mode4_profile(csm, length_samples=240, angle_samples=240):
    theta1_break = min(csm.theta1_limit, 0.5 * np.pi)

    primitives = {
        "outer_theta1": BoundaryPrimitive(
            "outer_theta1",
            _sample_mode4_theta1_curve(
                csm,
                Ls=csm.L_s0,
                theta1_start=0.0,
                theta1_end=theta1_break,
                theta_2=0.0,
                angle_samples=angle_samples,
            ),
        ),
        "outer_ls": BoundaryPrimitive(
            "outer_ls",
            _sample_mode4_ls_curve(
                csm,
                theta_1=theta1_break,
                theta_2=0.0,
                ls_start=csm.L_s0,
                ls_end=0.0,
                length_samples=length_samples,
            ),
        ),
        "outer_theta2": BoundaryPrimitive(
            "outer_theta2",
            _sample_mode4_theta2_curve(
                csm,
                Ls=0.0,
                theta_1=theta1_break,
                theta_start=0.0,
                theta_end=csm.theta2_limit,
                angle_samples=angle_samples,
            ),
        ),
        "inner_base_theta1": BoundaryPrimitive(
            "inner_base_theta1",
            _sample_mode3_theta1_curve(
                csm,
                L1=csm.L_10,
                theta1_start=0.0,
                theta1_end=csm.theta1_limit,
                theta_2=0.0,
                angle_samples=angle_samples,
            ),
        ),
        "inner_base_theta2": BoundaryPrimitive(
            "inner_base_theta2",
            _sample_mode3_theta2_curve(
                csm,
                L1=csm.L_10,
                theta_1=csm.theta1_limit,
                theta_start=0.0,
                theta_end=csm.theta2_limit,
                angle_samples=angle_samples,
            ),
        ),
        "inner_alt_theta2": BoundaryPrimitive(
            "inner_alt_theta2",
            _sample_mode3_theta2_curve(
                csm,
                L1=0.0,
                theta_1=0.0,
                theta_start=0.0,
                theta_end=csm.theta2_limit,
                angle_samples=angle_samples,
            ),
        ),
        "inner_alt_l1": BoundaryPrimitive(
            "inner_alt_l1",
            _sample_mode3_l1_curve(
                csm,
                theta_2=csm.theta2_limit,
                l1_start=0.0,
                l1_end=csm.L_10,
                length_samples=length_samples,
            ),
        ),
        "inner_alt_ls": BoundaryPrimitive(
            "inner_alt_ls",
            _sample_mode4_ls_curve(
                csm,
                theta_1=csm.theta1_limit,
                theta_2=csm.theta2_limit,
                ls_start=0.0,
                ls_end=csm.L_s0,
                length_samples=length_samples,
            ),
        ),
    }

    outer_segments = [
        primitives["outer_theta1"].points_rz,
        primitives["outer_ls"].points_rz,
        primitives["outer_theta2"].points_rz,
    ]

    base_inner_segments = [
        primitives["inner_base_theta1"].points_rz,
        primitives["inner_base_theta2"].points_rz,
    ]

    inner_segments = base_inner_segments
    chosen_inner_mode = "base_mode3_outer"
    alt_hit = None

    alt_ls_trimmed = primitives["inner_alt_ls"].points_rz[1:].copy()
    if alt_ls_trimmed.shape[0] >= 2:
        alt_hit = _first_polyline_intersection(primitives["inner_alt_l1"].points_rz, alt_ls_trimmed)

    if alt_hit is not None:
        alt_l1_prefix = _polyline_prefix(primitives["inner_alt_l1"].points_rz, alt_hit)
        alt_ls_suffix = _polyline_suffix(alt_ls_trimmed, alt_hit, which="b")
        inner_segments = [
            primitives["inner_alt_theta2"].points_rz,
            alt_l1_prefix,
            alt_ls_suffix,
        ]
        chosen_inner_mode = "alt_inner_with_ls_cover"

    inner_curve = _concat_curve_segments(inner_segments)
    axis_connector = _build_axis_closure(inner_curve[-1], outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))

    return WorkspaceProfile(
        mode=4,
        inner_segments=[segment.copy() for segment in inner_segments],
        outer_segments=[segment.copy() for segment in outer_segments],
        outer_open_curve_rz=_concat_curve_segments(outer_segments),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "primitives": {name: primitive.points_rz.copy() for name, primitive in primitives.items()},
            "outer_segments": [segment.copy() for segment in outer_segments],
            "inner_segments": [segment.copy() for segment in inner_segments],
            "theta1_break": float(theta1_break),
            "alt_hit": None if alt_hit is None else dict(alt_hit),
            "chosen_inner_mode": chosen_inner_mode,
        },
    )


def build_workspace_profile(csm, mode):
    if mode == 1:
        return build_mode1_profile(csm)
    if mode == 2:
        return build_mode2_profile(csm)
    if mode == 3:
        return build_mode3_profile(csm)
    if mode == 4:
        return build_mode4_profile(csm)
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


def _profile_mesh(profile_rz, revolve_samples=REVOLVE_SAMPLES):
    phi = np.linspace(0.0, 2.0 * np.pi, revolve_samples)
    radii = profile_rz[:, 0][:, None] * 1000.0
    heights = profile_rz[:, 1][:, None] * 1000.0
    x = radii * np.cos(phi)[None, :]
    y = radii * np.sin(phi)[None, :]
    z = np.repeat(heights, phi.size, axis=1)
    return x, y, z


def draw_revolved_shell(ax, profile_rz, color, alpha):
    profile_rz = _decimate_curve(profile_rz)
    if RENDER_3D_MODE == "fast_surface":
        x, y, z = _profile_mesh(profile_rz)
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


def draw_revolved_profile(ax, profile, color):
    draw_revolved_shell(ax, profile.outer_open_curve_rz, color=color, alpha=REACH_ALPHA)
    for unreachable_profile in profile.unreachable_open_curves_rz:
        draw_revolved_shell(ax, unreachable_profile, color=UNREACHABLE_COLOR, alpha=UNREACHABLE_ALPHA)

def draw_side_profile(ax, profile, color, label):
    symmetric_outline = _symmetric_fill_polygon(profile.outer_open_curve_rz) * 1000.0
    ax.fill(
        symmetric_outline[:, 0],
        symmetric_outline[:, 1],
        color=color,
        alpha=0.30,
        linewidth=0.0,
        label=label,
    )

    for idx, unreachable_profile in enumerate(profile.unreachable_closed_profiles_rz):
        symmetric_void = _symmetric_fill_polygon(unreachable_profile) * 1000.0
        ax.fill(
            symmetric_void[:, 0],
            symmetric_void[:, 1],
            color=UNREACHABLE_COLOR,
            alpha=0.80,
            linewidth=0.0,
            label="Unreachable" if idx == 0 else None,
        )

    for idx, curve in enumerate(profile.inner_segments):
        curve_mm = np.asarray(curve, dtype=float) * 1000.0
        ax.plot(
            curve_mm[:, 0],
            curve_mm[:, 1],
            color=UNREACHABLE_COLOR,
            linewidth=2.0,
            label="Inner contour" if idx == 0 else None,
        )
        ax.plot(
            -curve_mm[:, 0],
            curve_mm[:, 1],
            color=UNREACHABLE_COLOR,
            linewidth=2.0,
        )

    for idx, curve in enumerate(profile.outer_segments):
        curve_mm = np.asarray(curve, dtype=float) * 1000.0
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


def _configure_side_axes(ax, profiles):
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


def save_profile_debug_figure(profile):
    if not SAVE_DEBUG_FIGURES or profile.debug_data is None:
        return

    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    debug = profile.debug_data
    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    primitives = debug.get("primitives", {})
    color_map = {"tau0": "#C96868", "tau1": "#7C5CFC", "tau2": "#2E8B57"}
    for name, curve in primitives.items():
        _plot_debug_curve(ax, curve, label=name, color=color_map.get(name, "#444444"), linewidth=1.4, alpha=0.9)

    for idx, segment in enumerate(debug.get("outer_segments", [])):
        _plot_debug_curve(ax, segment, label=f"outer_seg_{idx+1}", color="#1F5D78", linewidth=2.2, alpha=0.95)

    for idx, segment in enumerate(debug.get("inner_segments", [])):
        _plot_debug_curve(ax, segment, label=f"inner_seg_{idx+1}", color=UNREACHABLE_COLOR, linewidth=2.2, alpha=0.95)

    for hit_key, color in (("hit_tau1", "#7C5CFC"), ("hit_tau2", "#2E8B57"), ("chosen_hit", "#111111")):
        hit = debug.get(hit_key)
        if hit is None:
            continue
        point = np.asarray(hit["point"], dtype=float) * 1000.0
        ax.scatter([point[0], -point[0]], [point[1], point[1]], s=50, color=color, zorder=5, label=hit_key)

    ax.set_title(f"Mode {profile.mode} Debug Primitives")
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Z [mm]")
    _configure_side_axes(ax, [profile])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    debug_path = DEBUG_OUTPUT_DIR / f"mode{profile.mode}_debug.png"
    fig.savefig(debug_path, dpi=240, bbox_inches="tight")
    print(f"Saved debug figure to: {debug_path.resolve()}")
    plt.close(fig)


def main():
    csm = CSM.from_config(CONFIG_PATH)
    profiles = [build_workspace_profile(csm, mode) for mode in PLOT_MODES]

    n_modes = max(len(profiles), 1)
    figsize = FIGSIZE if FIGSIZE is not None else (11.5, 5.2 * n_modes)
    fig = plt.figure(figsize=figsize)

    for idx, profile in enumerate(profiles, start=1):
        ax3d = fig.add_subplot(n_modes, 2, 2 * idx - 1, projection="3d")
        ax2d = fig.add_subplot(n_modes, 2, 2 * idx)
        color = MODE_COLORS.get(profile.mode, "#9FD4EA")

        draw_revolved_profile(ax3d, profile, color)
        draw_side_profile(ax2d, profile, color, label=f"Mode {profile.mode}")

        _configure_3d_axes(ax3d, [profile])
        _configure_side_axes(ax2d, [profile])
        ax3d.set_title(f"Mode {profile.mode} Workspace")
        ax2d.set_title(f"Mode {profile.mode} Side View")

        handles, labels = ax2d.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax2d.legend(
                unique.values(),
                unique.keys(),
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
        save_profile_debug_figure(profile)

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
