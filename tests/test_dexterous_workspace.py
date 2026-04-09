from pathlib import Path

import numpy as np

from csm import (
    CSM,
    DexterousParameters,
    analytic_fk_mode3,
    bootstrap_position_reachable_state,
    build_dexterous_probe,
    mode3_state_from_csm,
    scan_directions,
)
from csm.dexterous_workspace.plotting import DexterousPlotOptions, _build_patch_geometry_from_symmetry_region


def _build_csm():
    return CSM.from_config(Path("config/csm_cfg_3mm.yaml"))


def test_parameter_mapping_3mm():
    csm = _build_csm()
    params = DexterousParameters.from_csm(csm, config="3mm")
    assert np.isclose(params.L10_m, csm.L_10)
    assert np.isclose(params.L20_m, csm.L_20)
    assert np.isclose(params.Lr_m, csm.L_r0)
    assert np.isclose(params.Lg_m, csm.L_tool)
    assert np.isclose(params.theta1_plus, csm.theta1_limit)
    assert np.isclose(params.theta2_plus, csm.theta2_limit)
    expected_ri_min = csm.ri_min if csm.ri_min is not None else 0.0054038
    assert np.isclose(params.r1_minus_m, expected_ri_min)


def test_fk_alignment_mode3():
    csm = _build_csm()
    csm.set_state(
        mode=3,
        phi=0.25,
        L1=0.011,
        L2=csm.L_20,
        Lr=csm.L_r0,
        Ls=0.0,
        theta_1=0.62,
        theta_2=0.85,
        delta_1=-0.30,
        delta_2=0.45,
    )
    state = mode3_state_from_csm(csm)
    params = DexterousParameters.from_csm(csm)
    pos_fk, axis_fk = analytic_fk_mode3(params, state)
    assert np.linalg.norm(pos_fk - csm.pose[:3]) < 1e-9
    assert np.linalg.norm(axis_fk - csm.pose[3:]) < 1e-9


def test_single_probe_builds_region():
    csm = _build_csm()
    point = np.array([10.87, 10.28, 29.77], dtype=float) / 1000.0
    probe = build_dexterous_probe(point, csm=csm, config="3mm", method="analytic")
    assert probe.position_xyz.shape == (3,)
    assert np.isfinite(probe.position_xyz).all()
    assert probe.feasible_directions_world.ndim == 2
    assert probe.feasible_directions_world.shape[1] == 3
    assert probe.status in {"analytic_ok", "analytic_empty", "analytic_empty_fallback_ok"}
    if probe.feasible_directions_world.size:
        assert np.isfinite(probe.feasible_directions_world).all()
        assert np.allclose(np.linalg.norm(probe.feasible_directions_world, axis=1), 1.0, atol=1e-5)


def test_bootstrap_and_fallback_scan():
    csm = _build_csm()
    point = np.array([10.87, 10.28, 29.77], dtype=float) / 1000.0
    q_seed = bootstrap_position_reachable_state(csm, point)
    if q_seed is None:
        return
    fallback = scan_directions(csm, point, q_seed=q_seed, n_directions=24)
    assert fallback.directions_world.shape == (24, 3)
    assert fallback.reachable_mask.shape == (24,)
    assert len(fallback.solved_states) == 24


def test_patch_geometry_lifts_to_both_symmetry_sides():
    csm = _build_csm()
    point = np.array([10.87, 10.28, 29.77], dtype=float) / 1000.0
    probe = build_dexterous_probe(point, csm=csm, config="3mm", method="analytic")
    geom = _build_patch_geometry_from_symmetry_region(probe, DexterousPlotOptions())
    assert geom["vertices_3d_world"].shape[1] == 3
    assert geom["triangles"].shape[1] == 3
    y = geom["vertices_3d_world"][:, 1]
    assert np.any(y > 1e-6)
    assert np.any(y < -1e-6)
