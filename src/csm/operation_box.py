"""
Reusable helpers for drawing and placing operation boxes inside the workspace.
"""
from __future__ import annotations

from dataclasses import dataclass

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .workspace_boundary_scan.core import WorkspaceProfile


@dataclass(frozen=True)
class OperationBox:
    center_xyz: np.ndarray
    size_xyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_xyz", np.asarray(self.center_xyz, dtype=float).reshape(3))
        object.__setattr__(self, "size_xyz", np.asarray(self.size_xyz, dtype=float).reshape(3))
        if np.any(self.size_xyz <= 0.0):
            raise ValueError("OperationBox sizes must be strictly positive.")

    @property
    def half_size_xyz(self) -> np.ndarray:
        return 0.5 * self.size_xyz

    @property
    def bounds_min_xyz(self) -> np.ndarray:
        return self.center_xyz - self.half_size_xyz

    @property
    def bounds_max_xyz(self) -> np.ndarray:
        return self.center_xyz + self.half_size_xyz


def operation_box_vertices(box: OperationBox) -> dict[str, np.ndarray]:
    center = np.asarray(box.center_xyz, dtype=float)
    half = np.asarray(box.half_size_xyz, dtype=float)
    signs = {
        "bottom_back_left": (-1.0, -1.0, -1.0),
        "bottom_back_right": (1.0, -1.0, -1.0),
        "bottom_front_left": (-1.0, 1.0, -1.0),
        "bottom_front_right": (1.0, 1.0, -1.0),
        "top_back_left": (-1.0, -1.0, 1.0),
        "top_back_right": (1.0, -1.0, 1.0),
        "top_front_left": (-1.0, 1.0, 1.0),
        "top_front_right": (1.0, 1.0, 1.0),
    }
    return {
        name: center + half * np.asarray(sign_triplet, dtype=float)
        for name, sign_triplet in signs.items()
    }


def _profile_radius_lookup(profile: WorkspaceProfile) -> tuple[np.ndarray, np.ndarray]:
    curve = np.asarray(profile.outer_open_curve_rz, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2:
        raise ValueError("Workspace profile outer curve must have shape (N, 2) with N >= 2.")
    z_vals = curve[:, 1]
    r_vals = curve[:, 0]
    order = np.argsort(z_vals)
    z_sorted = z_vals[order]
    r_sorted = r_vals[order]

    unique_z = []
    unique_r = []
    for z in np.unique(np.round(z_sorted, decimals=9)):
        mask = np.isclose(z_sorted, z, atol=1e-9)
        unique_z.append(float(np.max(z_sorted[mask])))
        unique_r.append(float(np.max(r_sorted[mask])))

    z_unique = np.asarray(unique_z, dtype=float)
    r_unique = np.asarray(unique_r, dtype=float)
    sort_idx = np.argsort(z_unique)
    return z_unique[sort_idx], r_unique[sort_idx]


def _max_curve_height_at_radius(curve_rz: np.ndarray, radius: float, eps: float = 1.0e-9) -> float | None:
    curve = np.asarray(curve_rz, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 2:
        return None

    target_r = float(radius)
    best_z: float | None = None
    for idx in range(curve.shape[0] - 1):
        r0, z0 = curve[idx]
        r1, z1 = curve[idx + 1]
        r_min = min(float(r0), float(r1))
        r_max = max(float(r0), float(r1))
        if target_r < r_min - eps or target_r > r_max + eps:
            continue

        if abs(float(r1) - float(r0)) <= eps:
            if abs(target_r - float(r0)) > eps:
                continue
            z_val = max(float(z0), float(z1))
        else:
            t = np.clip((target_r - float(r0)) / (float(r1) - float(r0)), 0.0, 1.0)
            z_val = float(z0) + t * (float(z1) - float(z0))

        if best_z is None or z_val > best_z:
            best_z = z_val
    return best_z


def _profile_void_ceiling(profile: WorkspaceProfile, max_radius: float) -> float:
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative.")
    if not profile.unreachable_open_curves_rz:
        return float("-inf")

    query_radii = np.linspace(0.0, float(max_radius), 256)
    best_z = float("-inf")
    for curve in profile.unreachable_open_curves_rz:
        for radius in query_radii:
            z_val = _max_curve_height_at_radius(curve, radius)
            if z_val is not None and z_val > best_z:
                best_z = float(z_val)
    return best_z


def fit_centered_operation_box_in_mode3(
    profile: WorkspaceProfile,
    size_xyz_m: np.ndarray,
    top_margin_m: float,
    *,
    z_tolerance: float = 1.0e-6,
    max_iterations: int = 48,
) -> OperationBox:
    size_xyz = np.asarray(size_xyz_m, dtype=float).reshape(3)
    if profile.mode != 3:
        raise ValueError(f"Expected a mode3 profile, got mode={profile.mode}.")
    if top_margin_m < 0.0:
        raise ValueError("top_margin_m must be non-negative.")
    if z_tolerance <= 0.0:
        raise ValueError("z_tolerance must be strictly positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    z_axis, radius_axis = _profile_radius_lookup(profile)
    radial_clearance = float(np.hypot(size_xyz[0] / 2.0, size_xyz[1] / 2.0))
    half_height = float(size_xyz[2] / 2.0)
    void_ceiling_z = _profile_void_ceiling(profile, radial_clearance)
    z_min = float(np.min(z_axis))
    z_max = float(np.max(z_axis)) - float(top_margin_m)
    if z_max <= z_min:
        raise ValueError("Mode3 profile does not leave room for the requested top margin.")

    center_min = max(z_min + half_height, void_ceiling_z + half_height)
    center_max = z_max - half_height
    if center_max < center_min:
        raise ValueError("Operation box height exceeds the usable mode3 workspace height.")

    def _center_is_feasible(center_z: float) -> bool:
        sample_z = np.linspace(center_z - half_height, center_z + half_height, 64)
        if np.any(sample_z < z_axis[0]) or np.any(sample_z > z_axis[-1]):
            return False
        available_radius = np.interp(sample_z, z_axis, radius_axis)
        return bool(np.all(available_radius >= radial_clearance - 1e-9))

    if not _center_is_feasible(center_min):
        raise ValueError("Unable to fit the requested operation box inside the mode3 workspace.")

    low = float(center_min)
    high = float(center_max)
    chosen_center_z = low
    for _ in range(max_iterations):
        if high - low <= z_tolerance:
            break
        mid = 0.5 * (low + high)
        if _center_is_feasible(mid):
            chosen_center_z = mid
            low = mid
        else:
            high = mid

    if _center_is_feasible(high):
        chosen_center_z = high

    return OperationBox(
        center_xyz=np.array([0.0, 0.0, chosen_center_z], dtype=float),
        size_xyz=size_xyz,
    )


def fit_largest_centered_operation_box_in_mode3(
    profile: WorkspaceProfile,
    size_xyz_m: np.ndarray,
    top_margin_m: float,
    *,
    scale_tolerance: float = 1.0e-3,
    max_iterations: int = 32,
) -> tuple[OperationBox, float]:
    size_xyz = np.asarray(size_xyz_m, dtype=float).reshape(3)
    if np.any(size_xyz <= 0.0):
        raise ValueError("Requested operation box sizes must be strictly positive.")
    if scale_tolerance <= 0.0:
        raise ValueError("scale_tolerance must be strictly positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    requested_error: ValueError | None = None
    try:
        return fit_centered_operation_box_in_mode3(profile, size_xyz, top_margin_m), 1.0
    except ValueError as exc:
        requested_error = exc

    low = 0.0
    high = 1.0
    best_box: OperationBox | None = None
    best_scale = 0.0

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        if high - low <= scale_tolerance:
            break
        try:
            box = fit_centered_operation_box_in_mode3(profile, mid * size_xyz, top_margin_m)
        except ValueError:
            high = mid
            continue
        best_box = box
        best_scale = mid
        low = mid

    if best_box is None:
        raise ValueError(
            "Unable to fit any uniformly scaled version of the requested operation box inside the mode3 workspace."
        ) from requested_error

    return best_box, best_scale


def draw_operation_box(
    ax,
    box: OperationBox,
    *,
    face_color: str = "#8EC5E8",
    edge_color: str = "#325A73",
    alpha: float = 0.18,
    linewidth: float = 1.2,
) -> None:
    verts = operation_box_vertices(box)
    faces = [
        [verts["bottom_back_left"], verts["bottom_back_right"], verts["bottom_front_right"], verts["bottom_front_left"]],
        [verts["top_back_left"], verts["top_back_right"], verts["top_front_right"], verts["top_front_left"]],
        [verts["bottom_back_left"], verts["bottom_back_right"], verts["top_back_right"], verts["top_back_left"]],
        [verts["bottom_front_left"], verts["bottom_front_right"], verts["top_front_right"], verts["top_front_left"]],
        [verts["bottom_back_left"], verts["bottom_front_left"], verts["top_front_left"], verts["top_back_left"]],
        [verts["bottom_back_right"], verts["bottom_front_right"], verts["top_front_right"], verts["top_back_right"]],
    ]
    collection = Poly3DCollection(
        faces,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
