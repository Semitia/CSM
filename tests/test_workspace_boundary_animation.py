from pathlib import Path

import pytest

from csm import CSM
from csm.workspace_boundary_scan import (
    BoundaryScanAnimationOptions,
    BoundaryScanOptions,
    build_workspace_animation_data,
    build_workspace_profiles,
    render_workspace_scan_animation,
)


def _build_csm():
    return CSM.from_config(Path("config/csm_cfg_3mm.yaml"))


def test_animation_data_is_built_for_all_modes():
    csm = _build_csm()
    animation_data = build_workspace_animation_data(
        csm,
        modes=[0, 1, 2, 3, 4],
        options=BoundaryScanOptions(length_samples=48, angle_samples=48),
    )

    assert [item.mode for item in animation_data] == [0, 1, 2, 3, 4]
    assert all(item.stages for item in animation_data)


def test_mode0_animation_contains_trace_and_final_boundary():
    csm = _build_csm()
    mode0_animation = build_workspace_animation_data(
        csm,
        modes=[0],
        options=BoundaryScanOptions(length_samples=48, angle_samples=48),
    )[0]

    titles = [stage.title for stage in mode0_animation.stages]
    assert "Mode 0: Source Network" in titles
    assert "Mode 0: Right-Turn Trace" in titles
    assert "Mode 0: Final Boundary" in titles


def test_render_animation_requires_output_path_when_window_is_hidden():
    csm = _build_csm()
    profiles = build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=32, angle_samples=32),
    )
    animation_data = build_workspace_animation_data(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=32, angle_samples=32),
    )

    with pytest.raises(ValueError):
        render_workspace_scan_animation(
            profiles,
            animation_data,
            BoundaryScanAnimationOptions(enabled=True, output_path=None, show_figure=False),
        )


def test_render_animation_is_skipped_when_disabled(tmp_path):
    csm = _build_csm()
    profiles = build_workspace_profiles(
        csm,
        modes=[1],
        options=BoundaryScanOptions(length_samples=24, angle_samples=24),
    )
    animation_data = build_workspace_animation_data(
        csm,
        modes=[1],
        options=BoundaryScanOptions(length_samples=24, angle_samples=24),
    )
    output_path = tmp_path / "scan.gif"

    result = render_workspace_scan_animation(
        profiles,
        animation_data,
        BoundaryScanAnimationOptions(enabled=False, output_path=output_path, show_figure=False),
    )

    assert result is None
    assert not output_path.exists()
