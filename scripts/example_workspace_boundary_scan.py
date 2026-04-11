"""
Minimal usage example for csm.workspace_boundary_scan.
"""
from pathlib import Path

from csm import CSM
from csm.workspace_boundary_scan import (
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    build_workspace_profiles,
    plot_workspace_profiles,
)


def main():
    csm = CSM.from_config(Path("./config/csm_cfg_3mm.yaml"))

    profiles = build_workspace_profiles(
        csm,
        modes=[0, 1, 2, 3],
        options=BoundaryScanOptions(
            length_samples=180,
            angle_samples=180,
            mode0_debug_output_dir=Path("./data/workspace_boundary_scan_debug"),
        ),
    )

    plot_workspace_profiles(
        profiles,
        BoundaryScanPlotOptions(
            output_path=Path("./data/workspace_boundary_scan_example.png"),
            debug_output_dir=Path("./data/workspace_boundary_scan_debug"),
            show_figure=True,
            render_3d_mode="trisurf"
        ),
    )


if __name__ == "__main__":
    main()
