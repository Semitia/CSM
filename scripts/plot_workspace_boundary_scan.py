"""
Thin example script for the structured workspace boundary scan API.
"""
from pathlib import Path

from csm import CSM
from csm.workspace_boundary_scan import (
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    build_workspace_profiles,
    plot_workspace_profiles,
)


CONFIG_NAME = "csm_cfg_3mm.yaml"
CONFIG_PATH = Path("./config") / CONFIG_NAME
PLOT_MODES = [0]
OUTPUT_PATH = Path("./data/plot_workspace_boundary_scan.png")
DEBUG_OUTPUT_DIR = Path("./data/plot_workspace_boundary_scan_debug")


def main():
    csm = CSM.from_config(CONFIG_PATH)
    scan_options = BoundaryScanOptions(
        length_samples=240,
        angle_samples=240,
        mode0_debug_output_dir=DEBUG_OUTPUT_DIR,
    )
    plot_options = BoundaryScanPlotOptions(
        output_path=OUTPUT_PATH,
        debug_output_dir=DEBUG_OUTPUT_DIR,
        show_figure=True,
        save_debug_figures=True,
    )

    profiles = build_workspace_profiles(csm, PLOT_MODES, scan_options)
    plot_workspace_profiles(profiles, plot_options)

    if plot_options.output_path is not None:
        print(f"Saved figure to: {Path(plot_options.output_path).resolve()}")
    if plot_options.save_debug_figures and plot_options.debug_output_dir is not None:
        print(f"Debug figures directory: {Path(plot_options.debug_output_dir).resolve()}")


if __name__ == "__main__":
    main()
