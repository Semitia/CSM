"""
Render local dexterous workspace probes for the 3mm/mode3 workflow.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from csm import CSM, DexterousPlotOptions, build_dexterous_probe, render_dexterous_figure


DEFAULT_PROBES_MM = [
    [16.32, 2.14, 29.28],
    # [0,0,35],
    [2.63,0,35],
    [26.75,0,22.82],
    [20,0,14]
]
DEFAULT_DEBUG_OUTPUT_DIR = Path("data/dexterous_debug_patch")


def _parse_points(text: str):
    raw = ast.literal_eval(text)
    points = []
    for idx, item in enumerate(raw):
        if len(item) != 3:
            raise ValueError(f"Probe #{idx + 1} must contain exactly 3 coordinates.")
        points.append([float(v) / 1000.0 for v in item])
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Render local dexterous probes for the 3mm CSM configuration.")
    parser.add_argument("--config", default="config/csm_cfg_3mm.yaml", help="Path to the 3mm CSM yaml config.")
    parser.add_argument(
        "--points-mm",
        default=str(DEFAULT_PROBES_MM),
        help="Probe points in millimeters, e.g. '[[10.87, 10.28, 29.77], [8.0, 12.0, 31.0]]'.",
    )
    parser.add_argument("--output", default="data/dexterous_workspace_3mm.png", help="Output image path.")
    parser.add_argument("--method", default="analytic", choices=["analytic", "fallback"], help="Probe-building method.")
    parser.add_argument("--validate-with-fallback", action="store_true", help="Run fallback validation after analytic build.")
    parser.add_argument("--hide", action="store_true", help="Do not show the matplotlib window.")
    parser.add_argument(
        "--display-frame",
        default="local",
        choices=["local", "world"],
        help="Render probes in their local unit-sphere frame or mapped back into world coordinates.",
    )
    parser.add_argument("--show-robot", action="store_true", help="Overlay the robot in world-frame renders.")
    parser.add_argument(
        "--debug-output-dir",
        default=str(DEFAULT_DEBUG_OUTPUT_DIR),
        help="Directory for debug png/json outputs. Files with the same probe label will be overwritten.",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug png/json outputs.")
    args = parser.parse_args()

    csm = CSM.from_config(Path(args.config))
    points = _parse_points(args.points_mm)
    probes = []
    for idx, point in enumerate(points, start=1):
        probes.append(
            build_dexterous_probe(
                point,
                csm=csm,
                config="3mm",
                method=args.method,
                validate_with_fallback=args.validate_with_fallback,
                label=f"P{idx}",
            )
        )

    debug_output_dir = None if args.no_debug or args.debug_output_dir in {None, ""} else Path(args.debug_output_dir)
    render_dexterous_figure(
        probes,
        csm=csm,
        options=DexterousPlotOptions(
            output_path=Path(args.output),
            show_figure=not args.hide,
            display_frame=args.display_frame,
            show_robot=args.show_robot,
            save_debug_figure=debug_output_dir is not None,
            debug_output_dir=debug_output_dir,
        ),
    )

    if debug_output_dir is not None:
        print(f"Debug output dir: {debug_output_dir.resolve()}")

    for probe in probes:
        dbg = probe.debug_data or {}
        print(
            f"{probe.label}: status={probe.status}, feasible_sym={dbg.get('feasible_count_sym')}, "
            f"feasible_world={dbg.get('feasible_count_world')}, "
            f"cap_radius_deg={dbg.get('cap_angular_radius_deg')}, "
            f"cap_fit_error_deg={dbg.get('cap_fit_error_deg')}, "
            f"use_cap={dbg.get('fit_used_as_cap')}"
        )


if __name__ == "__main__":
    main()
