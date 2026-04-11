"""
Render a paper-style mode3 translation-workspace figure with an embedded operation box.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from csm import (
    CSM,
    OperationBox,
    draw_operation_box,
    fit_largest_centered_operation_box_in_mode3,
)
from csm.workspace_boundary_scan import BoundaryScanOptions, BoundaryScanPlotOptions, build_workspace_profiles
from csm.workspace_boundary_scan.plotting import (
    configure_3d_axes,
    configure_side_axes,
    draw_revolved_profile,
    draw_side_profile,
)


DEFAULT_OUTPUT = Path("data/example_mode3_translation_figure.png")
DEFAULT_BOX_SIZE_MM = np.array([50.0, 50.0, 40.0], dtype=float)
DEFAULT_TOP_MARGIN_MM = 0.0


@dataclass(frozen=True)
class FigureStyle:
    reachable_color: str = "#BFE4B5"
    reachable_alpha: float = 0.30
    unreachable_color: str = "#D3A37A"
    unreachable_alpha: float = 0.46
    outer_contour_color: str = "#5D7C5C"
    box_face_color: str = "#7EB9DF"
    box_edge_color: str = "#35566E"
    box_alpha_3d: float = 0.22
    box_alpha_side: float = 0.28
    box_linewidth_3d: float = 1.1
    box_linewidth_side: float = 1.2
    grid_alpha: float = 0.18
    main_title: str = "Mode3 Translation Workspace"
    side_title: str = "Side View"
    figure_title: str = "Mode3 Translation Workspace with Operation Box"


DEFAULT_STYLE = FigureStyle()


def _parse_box_size_mm(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.shape != (3,):
        raise ValueError("Expected --box-size-mm as 'sx,sy,sz'.")
    return values


def _resolve_box_size_mm(
    box_size_mm_text: str,
    *,
    box_width_mm: float | None,
    box_height_mm: float | None,
) -> np.ndarray:
    if box_width_mm is not None or box_height_mm is not None:
        if box_width_mm is None or box_height_mm is None:
            raise ValueError("Please provide both --box-width-mm and --box-height-mm.")
        return np.array([box_width_mm, box_width_mm, box_height_mm], dtype=float)
    return _parse_box_size_mm(box_size_mm_text)


def _box_to_display_units(box: OperationBox, scale: float) -> OperationBox:
    return OperationBox(center_xyz=scale * box.center_xyz, size_xyz=scale * box.size_xyz)


def _draw_side_box(ax, box_mm: OperationBox, style: FigureStyle) -> None:
    mins = box_mm.bounds_min_xyz
    rect = Rectangle(
        (mins[0], mins[2]),
        float(box_mm.size_xyz[0]),
        float(box_mm.size_xyz[2]),
        facecolor=style.box_face_color,
        edgecolor=style.box_edge_color,
        alpha=style.box_alpha_side,
        linewidth=style.box_linewidth_side,
    )
    ax.add_patch(rect)


def build_figure(
    *,
    csm: CSM,
    box_size_mm: np.ndarray,
    top_margin_mm: float,
    output_path: Path | None,
    show_figure: bool,
    style: FigureStyle = DEFAULT_STYLE,
) -> tuple[OperationBox, float]:
    profile = build_workspace_profiles(
        csm,
        modes=[3],
        options=BoundaryScanOptions(length_samples=220, angle_samples=220),
    )[0]
    box, box_scale = fit_largest_centered_operation_box_in_mode3(
        profile,
        size_xyz_m=box_size_mm / 1000.0,
        top_margin_m=top_margin_mm / 1000.0,
    )

    fig = plt.figure(figsize=(12.8, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.7, 1.0))

    plot_options = BoundaryScanPlotOptions(
        show_figure=False,
        save_debug_figures=False,
        reach_alpha=style.reachable_alpha,
        unreachable_alpha=style.unreachable_alpha,
        mode_colors={3: style.reachable_color},
        unreachable_color=style.unreachable_color,
        outer_contour_color=style.outer_contour_color,
        render_3d_mode="trisurf",
    )

    ax_main = fig.add_subplot(gs[0, 0], projection="3d")
    draw_revolved_profile(ax_main, profile, color=style.reachable_color, options=plot_options)
    draw_operation_box(
        ax_main,
        _box_to_display_units(box, 1000.0),
        face_color=style.box_face_color,
        edge_color=style.box_edge_color,
        alpha=style.box_alpha_3d,
        linewidth=style.box_linewidth_3d,
    )
    configure_3d_axes(ax_main, [profile])
    ax_main.view_init(elev=18, azim=-38)
    ax_main.set_title(style.main_title, pad=10.0)
    ax_main.grid(True, alpha=style.grid_alpha)

    ax_side = fig.add_subplot(gs[0, 1])
    draw_side_profile(ax_side, profile, color=style.reachable_color, label="Mode3", options=plot_options)
    _draw_side_box(ax_side, _box_to_display_units(box, 1000.0), style)
    configure_side_axes(ax_side, [profile])
    ax_side.set_title(style.side_title)
    ax_side.grid(True, alpha=style.grid_alpha)

    fig.suptitle(style.figure_title, fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    return box, box_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a mode3 translation-workspace illustration.")
    parser.add_argument("--config", default="config/csm_cfg_3mm.yaml", help="CSM config path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output figure path.")
    parser.add_argument("--box-size-mm", default="50,50,40", help="Operation box size in millimeters: sx,sy,sz.")
    parser.add_argument("--box-width-mm", type=float, help="Convenience option for symmetric square base: sets sx=sy=box-width-mm.")
    parser.add_argument("--box-height-mm", type=float, help="Convenience option for symmetric square base height sz.")
    parser.add_argument("--top-margin-mm", default=DEFAULT_TOP_MARGIN_MM, type=float, help="Top clearance to workspace roof.")
    parser.add_argument("--hide", action="store_true", help="Render without opening a window.")
    args = parser.parse_args()

    csm = CSM.from_config(Path(args.config))
    requested_box_size_mm = _resolve_box_size_mm(
        args.box_size_mm,
        box_width_mm=args.box_width_mm,
        box_height_mm=args.box_height_mm,
    )
    box, box_scale = build_figure(
        csm=csm,
        box_size_mm=requested_box_size_mm,
        top_margin_mm=float(args.top_margin_mm),
        output_path=Path(args.output),
        show_figure=not args.hide,
        style=DEFAULT_STYLE,
    )
    print(f"Saved translation figure to: {Path(args.output).resolve()}")
    if box_scale < 0.999:
        print(
            "Requested box was uniformly scaled to fit mode3:",
            f"scale={box_scale:.3f}",
            f"requested_mm={requested_box_size_mm.round(2).tolist()}",
        )
    print(f"Operation box center [mm]: {(1000.0 * box.center_xyz).round(2).tolist()}")
    print(f"Operation box size   [mm]: {(1000.0 * box.size_xyz).round(2).tolist()}")


if __name__ == "__main__":
    main()
