"""
Minimal usage example for csm.workspace_boundary_scan.
"""
from pathlib import Path

from csm import CSM
from csm.workspace_boundary_scan import (
    BoundaryScanAnimationOptions,
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    build_workspace_profile_animation_data,
    build_workspace_profiles,
    plot_workspace_profiles,
    render_workspace_scan_animation,
)

ENABLE_ANIMATION = False
ANIMATION_OUTPUT_PATH = Path("./data/workspace_boundary_scan_example.gif")
ANIMATION_FPS = 6
STATIC_OUTPUT_PATH = Path("./data/workspace_boundary_scan_example.png")
DEBUG_OUTPUT_DIR = Path("./data/workspace_boundary_scan_debug")
MODE3_SEGMENT_ROLES = {
    # Optional manual overrides for the mode 3 scan-line classification.
    # Valid keys: tau0, tau1, tau2, tau3
    # Valid values: "inner", "outer"
    #
    # Only the segments listed here are forced. Any omitted segment keeps the
    # existing automatic role.
    #
    # Example:
    # "tau2": "inner",
    "tau2": "inner",
}


def main():
    csm = CSM.from_config(Path("./config/csm_cfg_3mm_2.yaml"))
    modes = [3]
    scan_options = BoundaryScanOptions(
        length_samples=180,
        angle_samples=180,
        mode3_segment_roles=MODE3_SEGMENT_ROLES,
        mode0_debug_output_dir=DEBUG_OUTPUT_DIR,
    )
    print(
        "[workspace_boundary_scan] Building profiles "
        f"for modes={modes} with length_samples={scan_options.length_samples}, "
        f"angle_samples={scan_options.angle_samples}"
    )

    profiles = build_workspace_profiles(
        csm,
        modes=modes,
        options=scan_options,
    )
    print(f"[workspace_boundary_scan] Built {len(profiles)} workspace profile(s)")

    print(f"[workspace_boundary_scan] Rendering static figure -> {STATIC_OUTPUT_PATH}")
    plot_workspace_profiles(
        profiles,
        BoundaryScanPlotOptions(
            output_path=STATIC_OUTPUT_PATH,
            debug_output_dir=DEBUG_OUTPUT_DIR,
            show_figure=True,
            render_3d_mode="trisurf",
        ),
    )
    print(f"[workspace_boundary_scan] Static figure saved to: {STATIC_OUTPUT_PATH.resolve()}")
    print(f"[workspace_boundary_scan] Debug figures directory: {DEBUG_OUTPUT_DIR.resolve()}")

    if ENABLE_ANIMATION:
        print("[workspace_boundary_scan] Building animation stages from existing profiles")
        animation_data = [build_workspace_profile_animation_data(profile, csm=csm) for profile in profiles]
        total_stages = sum(len(item.stages) for item in animation_data)
        print(
            "[workspace_boundary_scan] Rendering process animation "
            f"({len(animation_data)} mode(s), {total_stages} stage(s), fps={ANIMATION_FPS}) "
            f"-> {ANIMATION_OUTPUT_PATH}"
        )
        render_workspace_scan_animation(
            profiles,
            animation_data,
            BoundaryScanAnimationOptions(
                enabled=True,
                output_path=ANIMATION_OUTPUT_PATH,
                fps=ANIMATION_FPS,
                show_figure=False,
            ),
        )
        print(f"[workspace_boundary_scan] Animation saved to: {ANIMATION_OUTPUT_PATH.resolve()}")
    else:
        print("[workspace_boundary_scan] Animation disabled; set ENABLE_ANIMATION = True to export GIF")


if __name__ == "__main__":
    main()
