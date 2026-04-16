from pathlib import Path

import numpy as np
import pytest

from csm import CSM, fit_centered_operation_box_in_mode3, fit_largest_centered_operation_box_in_mode3
from csm.workspace_boundary_scan import BoundaryScanOptions, build_workspace_profiles


def _build_mode3_profile():
    csm = CSM.from_config(Path("config/csm_cfg_3mm.yaml"))
    return build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=220, angle_samples=220),
    )[0]


def test_largest_mode3_box_shrinks_impossible_request():
    profile = _build_mode3_profile()
    requested_size = np.array([90.0, 90.0, 80.0], dtype=float) / 1000.0

    with pytest.raises(ValueError):
        fit_centered_operation_box_in_mode3(profile, requested_size, top_margin_m=0.004)

    box, scale = fit_largest_centered_operation_box_in_mode3(profile, requested_size, top_margin_m=0.004)

    assert 0.0 < scale < 1.0
    assert np.allclose(box.size_xyz, requested_size * scale)
    assert box.center_xyz.shape == (3,)
    assert np.isfinite(box.center_xyz).all()


def test_largest_mode3_box_preserves_feasible_request():
    profile = _build_mode3_profile()
    requested_size = np.array([20.0, 20.0, 10.0], dtype=float) / 1000.0

    direct_box = fit_centered_operation_box_in_mode3(profile, requested_size, top_margin_m=0.004)
    scaled_box, scale = fit_largest_centered_operation_box_in_mode3(profile, requested_size, top_margin_m=0.004)

    assert scale == pytest.approx(1.0)
    assert np.allclose(scaled_box.center_xyz, direct_box.center_xyz)
    assert np.allclose(scaled_box.size_xyz, direct_box.size_xyz)
