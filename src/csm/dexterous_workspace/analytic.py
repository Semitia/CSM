"""
Analytic CI-1 dexterous-workspace approximation adapted into the project tree.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .kinematics import (
    DexterousMode3State,
    DexterousParameters,
    analytic_fk_mode3,
    rot_y,
    rot_z,
)


@dataclass
class AnalyticRegion:
    feasible_area_sym: np.ndarray
    feasible_area_world: np.ndarray
    type1_sym: np.ndarray
    type2_sym: np.ndarray
    type1_world: np.ndarray
    type2_world: np.ndarray
    gamma: float


class AnalyticDexterousWorkspace:
    def __init__(self, params: DexterousParameters):
        self.params = params
        self._eq26_t: np.ndarray | None = None
        self._eq26_f: np.ndarray | None = None

    @staticmethod
    def to_symmetry_frame(p_target_m: np.ndarray, a_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        gamma = float(math.atan2(float(p_target_m[1]), float(p_target_m[0])))
        R_sw = rot_z(-gamma)
        p_s = R_sw @ np.asarray(p_target_m, dtype=float)
        a_s = R_sw @ np.asarray(a_target, dtype=float)
        return p_s, a_s, gamma

    @staticmethod
    def _dedupe_points(points: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points.reshape(0, points.shape[-1] if points.ndim > 1 else 0)
        decimals = max(0, int(round(-math.log10(eps))))
        _, idx = np.unique(np.round(points, decimals=decimals), axis=0, return_index=True)
        return points[np.sort(idx)]

    @staticmethod
    def _solve_quadratic(a: float, b: float, c: float, tol: float = 1e-12) -> np.ndarray:
        if abs(a) < tol:
            if abs(b) < tol:
                return np.array([], dtype=float)
            return np.array([-c / b], dtype=float)
        disc = b * b - 4.0 * a * c
        if disc < -tol:
            return np.array([], dtype=float)
        disc = max(disc, 0.0)
        s = math.sqrt(disc)
        return np.array([(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)], dtype=float)

    def _line_segment_length_mm(self, length_mm: float, theta: float) -> float:
        if abs(theta) < 1e-8:
            return 0.5 * length_mm
        return float(length_mm * math.tan(theta / 2.0) / theta)

    def ci1_line_coefficients(self, p_target_m: np.ndarray, theta2: float) -> tuple[float, float, float]:
        p_s, _, _ = self.to_symmetry_frame(np.asarray(p_target_m, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
        p_sx = 1000.0 * float(p_s[0])
        p_sz = 1000.0 * float(p_s[2])
        l2 = self._line_segment_length_mm(self.params.L20_mm, theta2)
        L2g = l2 + self.params.Lg_mm
        L2rz = l2 + self.params.Lr_mm + p_sz
        A1 = -2.0 * p_sx * (L2rz + L2g * math.cos(theta2))
        B1 = -L2g**2 + p_sx**2 - L2rz**2 - 2.0 * L2g * L2rz * math.cos(theta2)
        C1 = 2.0 * L2g * L2rz + (L2g**2 + p_sx**2 + L2rz**2) * math.cos(theta2)
        return float(A1), float(B1), float(C1)

    @staticmethod
    def _line_circle_intersections(A: float, B: float, C: float, tol: float = 1e-9) -> np.ndarray:
        points: list[list[float]] = []
        if abs(B) > tol:
            m = -A / B
            k = -C / B
            qa = 1.0 + m**2
            qb = 2.0 * m * k
            qc = k**2 - 1.0
            disc = qb**2 - 4.0 * qa * qc
            if disc < -tol:
                return np.zeros((0, 2), dtype=float)
            disc = max(disc, 0.0)
            sqrt_disc = math.sqrt(disc)
            for x in (((-qb + sqrt_disc) / (2.0 * qa)), ((-qb - sqrt_disc) / (2.0 * qa))):
                z = m * x + k
                points.append([x, z])
        elif abs(A) > tol:
            x = -C / A
            rad = 1.0 - x**2
            if rad < -tol:
                return np.zeros((0, 2), dtype=float)
            rad = max(rad, 0.0)
            z_abs = math.sqrt(rad)
            points.append([x, z_abs])
            points.append([x, -z_abs])
        else:
            return np.zeros((0, 2), dtype=float)

        pts = np.array(points, dtype=float)
        if pts.size == 0:
            return pts.reshape(0, 2)
        uniq = [pts[0]]
        for p in pts[1:]:
            if all(np.linalg.norm(p - u) > 1e-7 for u in uniq):
                uniq.append(p)
        return np.array(uniq, dtype=float)

    @staticmethod
    def _line_disk_segment(A: float, B: float, C: float, n_samples: int = 200) -> np.ndarray:
        pts = AnalyticDexterousWorkspace._line_circle_intersections(A, B, C)
        if pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        if pts.shape[0] == 1:
            return pts
        t = np.linspace(0.0, 1.0, n_samples)
        p0, p1 = pts[0], pts[1]
        return (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]

    def ci1_type2_boundary(self, p_target_m: np.ndarray, n_samples: int = 240) -> np.ndarray:
        theta_grid = np.linspace(1e-4, math.pi - 1e-4, n_samples)
        dtheta = 1e-4
        points: list[list[float]] = []
        for theta2 in theta_grid:
            A, B, C = self.ci1_line_coefficients(p_target_m, theta2)
            A_p, B_p, C_p = self.ci1_line_coefficients(p_target_m, theta2 + dtheta)
            A_m, B_m, C_m = self.ci1_line_coefficients(p_target_m, theta2 - dtheta)

            dA = (A_p - A_m) / (2.0 * dtheta)
            dB = (B_p - B_m) / (2.0 * dtheta)
            dC = (C_p - C_m) / (2.0 * dtheta)

            M = np.array([[A, B], [dA, dB]], dtype=float)
            rhs = -np.array([C, dC], dtype=float)
            if abs(np.linalg.det(M)) < 1e-10:
                continue
            xz = np.linalg.solve(M, rhs)
            if not np.all(np.isfinite(xz)):
                continue
            norm = float(np.linalg.norm(xz))
            if norm <= 1.01:
                points.append([float(xz[0]), float(xz[1]), float(theta2)])
        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.array(points, dtype=float)

    def _build_eq26_interp(self) -> None:
        if self._eq26_t is not None:
            return
        t = np.linspace(1e-6, math.pi - 1e-6, 2000)
        f = t / np.tan(t / 2.0)
        self._eq26_t = t[::-1]
        self._eq26_f = f[::-1]

    def _ci1_boundary2_points(self, p_target_m: np.ndarray, n_samples: int = 600) -> np.ndarray:
        p_s, _, _ = self.to_symmetry_frame(np.asarray(p_target_m, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
        p_sx = 1000.0 * float(p_s[0])
        p_sz = 1000.0 * float(p_s[2])
        theta1 = self.params.theta1_plus
        cos1 = math.cos(theta1)
        grid = np.linspace(1e-4, self.params.theta2_plus, n_samples)
        L20, Lg, Lr = self.params.L20_mm, self.params.Lg_mm, self.params.Lr_mm

        l2_arr = L20 * np.tan(grid / 2.0) / grid
        L2g_arr = l2_arr + Lg
        L2rz_arr = l2_arr + Lr + p_sz
        cos_grid = np.cos(grid)

        A1_arr = -2.0 * p_sx * (L2rz_arr + L2g_arr * cos_grid)
        B1_arr = -L2g_arr**2 + p_sx**2 - L2rz_arr**2 - 2.0 * L2g_arr * L2rz_arr * cos_grid
        C1_arr = 2.0 * L2g_arr * L2rz_arr + (L2g_arr**2 + p_sx**2 + L2rz_arr**2) * cos_grid

        mask = np.abs(A1_arr) > 1e-10
        if not np.any(mask):
            return np.zeros((0, 3), dtype=float)

        A1_m, B1_m, C1_m = A1_arr[mask], B1_arr[mask], C1_arr[mask]
        L2g_m, L2rz_m, theta2_m = L2g_arr[mask], L2rz_arr[mask], grid[mask]

        A2 = -2.0 * L2g_m**2
        B2 = -2.0 * L2g_m * L2rz_m * (cos1 - 1.0)
        C2 = -2.0 * L2g_m * p_sx * (1.0 + cos1)
        D2 = L2g_m**2 - L2rz_m**2 + p_sx**2 + (L2g_m**2 + L2rz_m**2 + p_sx**2) * cos1

        qa_arr = A2
        qb_arr = B2 - C2 * B1_m / A1_m
        qc_arr = D2 - C2 * C1_m / A1_m

        disc = qb_arr**2 - 4.0 * qa_arr * qc_arr
        valid = disc >= -1e-10
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float)

        disc_safe = np.maximum(disc[valid], 0.0)
        sqrt_disc = np.sqrt(disc_safe)
        qa_v, qb_v = qa_arr[valid], qb_arr[valid]
        asz1 = (-qb_v + sqrt_disc) / (2.0 * qa_v)
        asz2 = (-qb_v - sqrt_disc) / (2.0 * qa_v)

        B1_v, C1_v, A1_v = B1_m[valid], C1_m[valid], A1_m[valid]
        t2_v = theta2_m[valid]
        asx1 = -(B1_v * asz1 + C1_v) / A1_v
        asx2 = -(B1_v * asz2 + C1_v) / A1_v
        pts = np.vstack(
            [
                np.column_stack([asx1, asz1, t2_v]),
                np.column_stack([asx2, asz2, t2_v]),
            ]
        )
        mask_valid = pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0 + 1e-8
        return self._dedupe_points(pts[mask_valid])

    def _ci1_boundary34_points(self, p_target_m: np.ndarray, use_r1_limit: bool, n_samples: int = 600) -> np.ndarray:
        p_s, _, _ = self.to_symmetry_frame(np.asarray(p_target_m, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
        p_sx = 1000.0 * float(p_s[0])
        p_sz = 1000.0 * float(p_s[2])

        grid = np.linspace(1e-4, self.params.theta2_plus, n_samples)
        L20, Lg, Lr = self.params.L20_mm, self.params.Lg_mm, self.params.Lr_mm

        l2_arr = L20 * np.tan(grid / 2.0) / grid
        L2g_arr = l2_arr + Lg
        L2rz_arr = l2_arr + Lr + p_sz
        cos_grid = np.cos(grid)

        A1_arr = -2.0 * p_sx * (L2rz_arr + L2g_arr * cos_grid)
        B1_arr = -L2g_arr**2 + p_sx**2 - L2rz_arr**2 - 2.0 * L2g_arr * L2rz_arr * cos_grid
        C1_arr = 2.0 * L2g_arr * L2rz_arr + (L2g_arr**2 + p_sx**2 + L2rz_arr**2) * cos_grid

        denom = 2.0 * (L2g_arr * cos_grid + L2rz_arr)
        denom_mask = np.abs(denom) > 1e-10
        denom_idx = np.where(denom_mask)[0]
        if len(denom_idx) == 0:
            return np.zeros((0, 4), dtype=float)

        l1_vals = (
            -L2g_arr[denom_idx] ** 2
            - (Lr + l2_arr[denom_idx]) ** 2
            - 2.0 * L2g_arr[denom_idx] * (Lr + l2_arr[denom_idx]) * cos_grid[denom_idx]
            + p_sx**2
            + p_sz**2
        ) / denom[denom_idx]
        l1_finite = np.isfinite(l1_vals) & (l1_vals >= 0.0)
        valid_idx = denom_idx[l1_finite]
        l1_valid = l1_vals[l1_finite]
        if use_r1_limit:
            theta1_vals = 2.0 * np.arctan2(l1_valid, self.params.r1_minus_mm)
        else:
            self._build_eq26_interp()
            target = self.params.L10_mm / l1_valid
            mask_interp = (target > 0.0) & (target < 2.0)
            theta1_vals = np.full_like(l1_valid, np.nan)
            theta1_vals[mask_interp] = np.interp(target[mask_interp], self._eq26_f, self._eq26_t)
            theta1_vals[target >= 2.0] = 1e-6

        theta1_ok = np.isfinite(theta1_vals) & (theta1_vals >= 0.0) & (theta1_vals <= self.params.theta1_plus + 1e-8)
        final_idx = valid_idx[theta1_ok]
        theta1_ok_vals = theta1_vals[theta1_ok]
        if len(final_idx) == 0:
            return np.zeros((0, 4), dtype=float)

        cos1_v = np.cos(theta1_ok_vals)
        L1z_v = l1_valid[theta1_ok] - p_sz
        L1gr_v = l1_valid[theta1_ok] - self.params.Lg_mm + self.params.Lr_mm

        A3_v = 2.0 * p_sx * (L1z_v + L1gr_v * cos1_v)
        B3_v = p_sx**2 - L1gr_v**2 - L1z_v**2 - 2.0 * L1gr_v * L1z_v * cos1_v
        C3_v = 2.0 * L1gr_v * L1z_v + (p_sx**2 + L1gr_v**2 + L1z_v**2) * cos1_v
        A1_v, B1_v, C1_v = A1_arr[final_idx], B1_arr[final_idx], C1_arr[final_idx]

        det = A1_v * B3_v - A3_v * B1_v
        valid_det = np.abs(det) > 1e-10
        A1_v, B1_v, C1_v = A1_v[valid_det], B1_v[valid_det], C1_v[valid_det]
        A3_v, B3_v, C3_v = A3_v[valid_det], B3_v[valid_det], C3_v[valid_det]
        det = det[valid_det]

        asx = (-C1_v * B3_v + C3_v * B1_v) / det
        asz = (-A1_v * C3_v + A3_v * C1_v) / det
        t1_v = theta1_ok_vals[valid_det]
        grid_v = grid[final_idx][valid_det]
        pts = np.column_stack([asx, asz, t1_v, grid_v])
        mask_pts = pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0 + 1e-8
        out_pts = pts[mask_pts]
        if use_r1_limit:
            out_pts = np.vstack([out_pts, [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
        return self._dedupe_points(out_pts)

    def ci1_type1_boundaries(self, p_target_m: np.ndarray, n_samples: int = 600) -> np.ndarray:
        A1, B1, C1 = self.ci1_line_coefficients(p_target_m, self.params.theta2_plus)
        b1 = self._line_disk_segment(A1, B1, C1, n_samples=max(100, n_samples // 3))
        if b1.size > 0:
            b1 = np.column_stack([b1, np.full(len(b1), self.params.theta2_plus)])
        b2 = self._ci1_boundary2_points(p_target_m, n_samples=n_samples)
        b3 = self._ci1_boundary34_points(p_target_m, use_r1_limit=True, n_samples=n_samples)
        if b3.size > 0:
            b3 = b3[:, [0, 1, 3]]
        b4 = self._ci1_boundary34_points(p_target_m, use_r1_limit=False, n_samples=n_samples)
        if b4.size > 0:
            b4 = b4[:, [0, 1, 3]]
        all_pts = np.vstack([x for x in [b1, b2, b3, b4] if x.size > 0]) if any(x.size > 0 for x in [b1, b2, b3, b4]) else np.zeros((0, 3))
        return self._dedupe_points(all_pts)

    def ci1_feasible_area(self, p_target_m: np.ndarray, n_grid: int = 120) -> np.ndarray:
        p_s, _, _ = self.to_symmetry_frame(np.asarray(p_target_m, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
        p_sx = 1000.0 * float(p_s[0])
        p_sz = 1000.0 * float(p_s[2])
        if abs(p_sx) < 1e-8:
            p_sx = 1e-8

        t2_vals = np.linspace(1e-4, self.params.theta2_plus, n_grid)
        asz_vals = np.linspace(-1.0, 1.0, n_grid)
        T2, ASZ = np.meshgrid(t2_vals, asz_vals)
        t2 = T2.flatten()
        asz = ASZ.flatten()

        L20, Lg, Lr = self.params.L20_mm, self.params.Lg_mm, self.params.Lr_mm
        l2 = L20 * np.tan(t2 / 2.0) / t2
        L2g = l2 + Lg
        L2rz = l2 + Lr + p_sz
        cos_t2 = np.cos(t2)

        A1 = -2.0 * p_sx * (L2rz + L2g * cos_t2)
        B1 = -L2g**2 + p_sx**2 - L2rz**2 - 2.0 * L2g * L2rz * cos_t2
        C1 = 2.0 * L2g * L2rz + (L2g**2 + p_sx**2 + L2rz**2) * cos_t2

        valid_A1 = np.abs(A1) > 1e-10
        asx = np.zeros_like(asz)
        asx[valid_A1] = -(B1[valid_A1] * asz[valid_A1] + C1[valid_A1]) / A1[valid_A1]
        valid = (asx**2 + asz**2 <= 1.0 + 1e-8) & valid_A1
        if not np.any(valid):
            return np.zeros((0, 2), dtype=float)

        t2 = t2[valid]
        asz = asz[valid]
        asx = asx[valid]
        l2 = l2[valid]
        L2g = L2g[valid]
        cos_t2 = cos_t2[valid]

        denom = 2.0 * (L2g * cos_t2 + (l2 + Lr + p_sz))
        valid_denom = np.abs(denom) > 1e-10
        t2, asz, asx, l2, denom, L2g = t2[valid_denom], asz[valid_denom], asx[valid_denom], l2[valid_denom], denom[valid_denom], L2g[valid_denom]
        l1 = (
            -L2g**2
            - (Lr + l2) ** 2
            - 2.0 * L2g * (Lr + l2) * np.cos(t2)
            + p_sx**2
            + p_sz**2
        ) / denom

        valid_l1 = l1 >= 0.0
        t2, asz, asx, l1 = t2[valid_l1], asz[valid_l1], asx[valid_l1], l1[valid_l1]
        L1r2 = l1 + Lr + l2[valid_l1]
        valid_L1r2 = L1r2 > 1e-10
        t2, asz, asx, l1, L1r2 = t2[valid_L1r2], asz[valid_L1r2], asx[valid_L1r2], l1[valid_L1r2], L1r2[valid_L1r2]

        p2_z = p_sz - (l2[valid_l1][valid_L1r2] + Lg) * asz
        cos_t1 = (p2_z - l1) / L1r2
        valid_cos = (cos_t1 >= -1.0) & (cos_t1 <= 1.0)
        t2, asz, asx, l1, cos_t1 = t2[valid_cos], asz[valid_cos], asx[valid_cos], l1[valid_cos], cos_t1[valid_cos]

        t1 = np.arccos(cos_t1)
        m1 = np.abs(t1) < 1e-6
        L1 = np.empty_like(t1)
        L1[m1] = 2.0 * l1[m1]
        t1_nz = t1[~m1]
        L1[~m1] = l1[~m1] * t1_nz / np.tan(t1_nz / 2.0)

        valid_final = (
            (L1 >= 0.0)
            & (L1 <= self.params.L10_mm + 1e-6)
            & (t1 >= 0.0)
            & (t1 <= self.params.theta1_plus + 1e-6)
            & (L1 >= self.params.r1_minus_mm * t1 - 1e-6)
        )
        pts = np.column_stack([asx[valid_final], asz[valid_final]])
        return self._dedupe_points(pts, eps=5e-3)

    @staticmethod
    def _map_projected_to_world(gamma: float, points_xz: np.ndarray) -> np.ndarray:
        if points_xz.size == 0:
            return np.zeros((0, 3), dtype=float)
        x = points_xz[:, 0]
        z = points_xz[:, 1]
        r = 1.0 - x * x - z * z
        valid = r >= -1e-8
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float)
        x, z = x[valid], z[valid]
        y_abs = np.sqrt(np.maximum(r[valid], 0.0))
        pos = np.column_stack([x, y_abs, z])
        neg = np.column_stack([x, -y_abs, z])
        R_ws = rot_z(gamma)
        world_pos = (R_ws @ pos.T).T
        world_neg = (R_ws @ neg.T).T
        return np.vstack([world_pos, world_neg])

    def build_region(self, position_xyz_m: np.ndarray, n_boundary_samples: int = 280, n_area_samples: int = 120) -> AnalyticRegion:
        p_target = np.asarray(position_xyz_m, dtype=float)
        type1_sym = self.ci1_type1_boundaries(p_target, n_samples=n_boundary_samples)
        type2_sym = self.ci1_type2_boundary(p_target, n_samples=max(45, n_boundary_samples // 6))
        feasible_area_sym = self.ci1_feasible_area(p_target, n_grid=n_area_samples)
        _, _, gamma = self.to_symmetry_frame(p_target, np.array([0.0, 0.0, 1.0], dtype=float))
        return AnalyticRegion(
            feasible_area_sym=feasible_area_sym,
            feasible_area_world=self._map_projected_to_world(gamma, feasible_area_sym),
            type1_sym=type1_sym[:, :2] if type1_sym.size else np.zeros((0, 2), dtype=float),
            type2_sym=type2_sym[:, :2] if type2_sym.size else np.zeros((0, 2), dtype=float),
            type1_world=self._map_projected_to_world(gamma, type1_sym[:, :2] if type1_sym.size else np.zeros((0, 2), dtype=float)),
            type2_world=self._map_projected_to_world(gamma, type2_sym[:, :2] if type2_sym.size else np.zeros((0, 2), dtype=float)),
            gamma=gamma,
        )

    @staticmethod
    def _wrap_to_pi(x: float) -> float:
        return float((x + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def _safe_delta2_minus_delta1(
        p1: np.ndarray,
        p2: np.ndarray,
        p_target: np.ndarray,
        a: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        cross_p1_p2 = np.cross(p1, p2)
        sgn_val = float(np.sign(np.dot(cross_p1_p2, p_target)))
        if abs(sgn_val) < eps:
            return 0.0
        z_w = np.array([0.0, 0.0, 1.0], dtype=float)
        v1 = np.cross(z_w, p2)
        v2 = np.cross(p2 - p1, a)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < eps or n2 < eps:
            return 0.0
        v1 = v1 / n1
        v2 = v2 / n2
        cos_val = np.clip(float(np.dot(v1, v2)), -1.0, 1.0)
        return float(sgn_val * math.acos(cos_val))

    def _align_roll(self, state: DexterousMode3State, R_target: np.ndarray) -> DexterousMode3State:
        _, axis = analytic_fk_mode3(self.params, state)
        # Use the same reconstruction chain as the ref implementation.
        base_R = rot_z(state.phi) @ rot_z(state.delta1) @ rot_y(state.theta1) @ rot_z(-state.delta1) @ rot_z(state.delta2) @ rot_y(state.theta2) @ rot_z(-state.delta2)
        R_err = base_R.T @ R_target
        alpha = float(math.atan2(R_err[1, 0] - R_err[0, 1], R_err[0, 0] + R_err[1, 1]))
        del axis  # only to make the FK dependency explicit for future debugging
        return DexterousMode3State(
            phi=self._wrap_to_pi(state.phi + alpha),
            theta1=state.theta1,
            L1=state.L1,
            delta1=self._wrap_to_pi(state.delta1 - alpha),
            theta2=state.theta2,
            delta2=self._wrap_to_pi(state.delta2 - alpha),
        )

    def solve_mode3_from_theta2(
        self,
        theta2: float,
        p_target_m: np.ndarray,
        R_target: np.ndarray,
        limit_margin: float = 0.0,
    ) -> tuple[Optional[DexterousMode3State], bool]:
        a = np.asarray(R_target, dtype=float)[:, 2]
        theta2_max = self.params.theta2_plus + limit_margin
        theta2 = float(np.clip(theta2, 0.0, theta2_max))

        L20, Lr, Lg, L10 = self.params.L20_m, self.params.Lr_m, self.params.Lg_m, self.params.L10_m
        l2 = L20 * math.tan(theta2 / 2.0) / theta2 if abs(theta2) >= 1e-8 else 0.5 * L20
        a_z = float(a[2])
        p_z = float(p_target_m[2])
        c1 = -2.0 * (float(np.dot(p_target_m, a)) - Lg + Lr)
        c2 = float(np.dot(p_target_m, p_target_m - 2.0 * Lg * a) + Lg**2 - Lr**2)
        c3 = 2.0 * (1.0 - a_z)
        c4 = 2.0 * (p_z - a_z * Lg + Lr)

        denom = c3 * l2 + c4
        if abs(denom) < 1e-8:
            return None, False
        l1 = (c1 * l2 + c2) / denom
        if l1 < -1e-8:
            return None, False
        l1 = max(l1, 0.0)

        p2 = np.asarray(p_target_m, dtype=float) - (l2 + Lg) * a
        L1r2 = l1 + Lr + l2
        if L1r2 <= 1e-10:
            return None, False
        cos_theta1 = np.clip((p2[2] - l1) / L1r2, -1.0, 1.0)
        theta1 = float(math.acos(cos_theta1))
        if abs(theta1) < 1e-8:
            L1 = 2.0 * l1
        else:
            den = math.tan(theta1 / 2.0)
            if abs(den) < 1e-10:
                return None, False
            L1 = l1 * theta1 / den

        if L1 < -1e-8 or L1 > L10 + limit_margin:
            return None, False
        if theta1 < -1e-8 or theta1 > self.params.theta1_plus + limit_margin:
            return None, False
        min_L1 = self.params.r1_minus_m * theta1
        if L1 < min_L1 - limit_margin - 1e-6:
            return None, False

        L1 = float(np.clip(L1, min_L1, L10))
        theta1 = float(np.clip(theta1, 0.0, self.params.theta1_plus))
        p1 = np.array([0.0, 0.0, l1], dtype=float)
        phi_plus_delta1 = float(math.atan2(p2[1], p2[0]))
        delta2_minus_delta1 = self._safe_delta2_minus_delta1(p1, p2, np.asarray(p_target_m, dtype=float), a)
        R_temp = np.asarray(R_target, dtype=float).T @ rot_z(phi_plus_delta1) @ rot_y(theta1) @ rot_z(delta2_minus_delta1) @ rot_y(theta2)
        delta2 = float(math.atan2(R_temp[1, 0], R_temp[0, 0]))
        delta1 = float(delta2 - delta2_minus_delta1)
        phi = float(phi_plus_delta1 - delta1)
        state = DexterousMode3State(
            phi=self._wrap_to_pi(phi),
            theta1=theta1,
            L1=L1,
            delta1=self._wrap_to_pi(delta1),
            theta2=theta2,
            delta2=self._wrap_to_pi(delta2),
        )
        aligned = self._align_roll(state, np.asarray(R_target, dtype=float))
        if not np.all(np.isfinite(np.array(list(aligned.__dict__.values()), dtype=float))):
            return None, False
        return aligned, True
