from pathlib import Path

import numpy as np

from csm import CSM
from csm.workspace_boundary_scan import BoundaryScanOptions, build_workspace_profiles
from csm.workspace_boundary_scan.core import _first_polyline_intersection


def _build_default_mode3_profile():
    csm = CSM.from_config(Path("config/csm_cfg_3mm.yaml"))
    return build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=180, angle_samples=180),
    )[0]


def _build_small_angle_mode3_profile():
    baseline = CSM.from_config(Path("config/csm_cfg_3mm.yaml"))
    csm = CSM(
        L_10=baseline.L_10,
        L_20=baseline.L_20,
        L_r0=baseline.L_r0,
        L_s0=baseline.L_s0,
        L_tool=baseline.L_tool,
        theta1_max=np.deg2rad(60.0),
        theta2_max=baseline.theta2_max,
        delta_t=baseline.delta_t,
        r2_min=baseline.r2_min,
    )
    return build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=180, angle_samples=180),
    )[0]


def test_mode3_small_angle_profile_keeps_default_classification():
    profile = _build_small_angle_mode3_profile()
    debug = profile.debug_data or {}

    assert debug.get("chosen_profile_mode") == "default"
    assert debug.get("chosen_hit") is None
    assert debug.get("discarded_segments") == []
    assert len(profile.inner_segments) == 2
    assert len(profile.outer_segments) == 2
    assert profile.unreachable_open_curves_rz[0].shape[0] >= profile.inner_segments[0].shape[0]


def test_mode3_large_angle_profile_trims_overlap_into_shell():
    profile = _build_default_mode3_profile()
    debug = profile.debug_data or {}

    assert debug.get("chosen_profile_mode") == "overlap_shell"
    assert debug.get("chosen_hit_name") == "tau1_tau3"
    assert "tau1_tau3" in debug.get("overlap_hits", {})
    assert len(debug.get("discarded_segments", [])) >= 2
    assert len(profile.inner_segments) == 2
    assert len(profile.outer_segments) == 1

    final_inner = profile.unreachable_open_curves_rz[0]
    final_outer = profile.outer_open_curve_rz
    hit = _first_polyline_intersection(final_inner, final_outer)
    assert hit is not None

    intersection_point = np.asarray(hit["point"], dtype=float)
    assert np.allclose(intersection_point, final_inner[-1], atol=1.0e-8)
    assert np.allclose(intersection_point, final_outer[0], atol=1.0e-8)

    discarded_names = [name for name, _role, _curve in debug.get("discarded_segments", [])]
    assert "tau2_overlap_branch" in discarded_names
