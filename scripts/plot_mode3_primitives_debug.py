"""
Generate mode 3 primitives debug visualization.
Useful for debugging theta1-constrained boundary scan logic.
"""
from pathlib import Path

from csm import CSM
from csm.workspace_boundary_scan import (
    BoundaryScanOptions,
    build_workspace_profiles,
    plot_mode3_primitives_debug,
)


def main():
    config_path = Path("./config/csm_cfg_3mm_3.yaml")
    output_path = Path("./data/mode3_primitives_debug.png")

    print(f"Loading config: {config_path}")
    csm = CSM.from_config(config_path)

    print(f"Building mode 3 profile...")
    profiles = build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=180, angle_samples=180),
    )

    profile = profiles[0]
    print(f"Profile mode: {profile.debug_data.get('chosen_profile_mode')}")
    print(f"Primitives: {list(profile.debug_data.get('primitives', {}).keys())}")

    print(f"\nGenerating debug plot -> {output_path}")
    plot_mode3_primitives_debug(profile, output_path=output_path, show_figure=False)
    print(f"Debug plot saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
