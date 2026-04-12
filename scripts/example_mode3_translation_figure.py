"""
Render a paper-style mode3 translation-workspace figure with an embedded operation box.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from csm import (
    CSM,
    DexterousMode3State,
    DexterousParameters,
    analytic_fk_mode3,
    OperationBox,
    draw_operation_box,
    fit_largest_centered_operation_box_in_mode3,
    make_mode3_display_csm,
)
from csm.visualizer import Visualizer
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
MM_PER_M = 1000.0
ARM_RANDOM_SEED = 20260412
ARM_RANDOM_SAMPLES = 2500


@dataclass(frozen=True)
class FigureStyle:
    reachable_color: str = "#BFE4B5"
    reachable_alpha: float = 0.4
    unreachable_color: str = "#4B4ED6"
    unreachable_alpha: float = 0.6
    outer_contour_color: str = "#5D7C5C"
    box_face_color: str = "#7EDCDF"
    box_edge_color: str = "#35566E"
    box_alpha_3d: float = 0.2
    box_alpha_side: float = 0.25
    box_linewidth_3d: float = 1.1
    box_linewidth_side: float = 1.2
    grid_alpha: float = 0.18
    main_title: str = "Mode3 Translation Workspace"
    side_title: str = "Side View"
    figure_title: str = "Mode3 Translation Workspace with Operation Box"

DEFAULT_STYLE = FigureStyle()
BOX_INFO_SCHEMA_VERSION = 1


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


def _csm_signature(csm: CSM) -> dict[str, float]:
    return {
        "L_10_m": float(csm.L_10),
        "L_20_m": float(csm.L_20),
        "L_r0_m": float(csm.L_r0),
        "L_s0_m": float(csm.L_s0),
        "L_tool_m": float(csm.L_tool),
        "theta1_limit_rad": float(csm.theta1_limit),
        "theta2_limit_rad": float(csm.theta2_limit),
        "ri_min_m": None if csm.ri_min is None else float(csm.ri_min),
    }


def _box_info_payload(
    *,
    csm: CSM,
    config_path: str,
    requested_box_size_mm: np.ndarray,
    top_margin_mm: float,
    box: OperationBox,
    box_scale: float,
) -> dict[str, object]:
    return {
        "schema_version": BOX_INFO_SCHEMA_VERSION,
        "source_script": "example_mode3_translation_figure.py",
        "config_path": config_path,
        "csm_signature": _csm_signature(csm),
        "requested_box_size_mm": np.asarray(requested_box_size_mm, dtype=float).tolist(),
        "top_margin_mm": float(top_margin_mm),
        "box_scale": float(box_scale),
        "box_center_mm": (MM_PER_M * np.asarray(box.center_xyz, dtype=float)).tolist(),
        "box_size_mm": (MM_PER_M * np.asarray(box.size_xyz, dtype=float)).tolist(),
    }


def _save_box_info(
    path: Path,
    *,
    csm: CSM,
    config_path: str,
    requested_box_size_mm: np.ndarray,
    top_margin_mm: float,
    box: OperationBox,
    box_scale: float,
) -> None:
    payload = _box_info_payload(
        csm=csm,
        config_path=config_path,
        requested_box_size_mm=requested_box_size_mm,
        top_margin_mm=top_margin_mm,
        box=box,
        box_scale=box_scale,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _state_direction(csm: CSM, state) -> np.ndarray:
    display_csm = make_mode3_display_csm(csm, state)
    return _normalize(np.asarray(display_csm.pose[3:], dtype=float))


def _state_position(csm: CSM, state) -> np.ndarray:
    display_csm = make_mode3_display_csm(csm, state)
    return np.asarray(display_csm.pose[:3], dtype=float)


def _score_overlay_state(csm: CSM, box: OperationBox, state) -> float:
    pos = _state_position(csm, state)
    direction = _state_direction(csm, state)
    center = np.asarray(box.center_xyz, dtype=float)
    half = np.maximum(np.asarray(box.half_size_xyz, dtype=float), 1e-9)
    normalized_offset = (pos - center) / half
    radial_offset = float(np.linalg.norm(normalized_offset[:2]))
    axial_offset = abs(float(normalized_offset[2]))
    tilt_score = float(np.linalg.norm(direction[:2]))
    upward_score = max(float(direction[2]), 0.0)
    centered_score = 1.0 - min(float(np.linalg.norm(normalized_offset)), 1.0)
    return 1.25 * centered_score - 0.80 * radial_offset - 0.35 * axial_offset + 0.30 * tilt_score + 0.05 * upward_score


def _sample_random_mode3_state(rng: np.random.Generator, params: DexterousParameters) -> DexterousMode3State:
    theta1 = float(rng.uniform(0.15 * params.theta1_plus, 0.98 * params.theta1_plus))
    theta2 = float(rng.uniform(0.15 * params.theta2_plus, 0.98 * params.theta2_plus))
    min_L1 = max(params.r1_minus_m * theta1, 1e-5)
    L1 = float(rng.uniform(min_L1, params.L10_m))
    return DexterousMode3State(
        phi=float(rng.uniform(-np.pi, np.pi)),
        theta1=theta1,
        L1=L1,
        delta1=float(rng.uniform(-np.pi, np.pi)),
        theta2=theta2,
        delta2=float(rng.uniform(-np.pi, np.pi)),
    )


def _point_inside_box(point_xyz: np.ndarray, box: OperationBox, *, margin_ratio: float = 0.06) -> bool:
    point = np.asarray(point_xyz, dtype=float)
    margin = margin_ratio * np.asarray(box.size_xyz, dtype=float)
    lower = np.asarray(box.bounds_min_xyz, dtype=float) + margin
    upper = np.asarray(box.bounds_max_xyz, dtype=float) - margin
    return bool(np.all(point >= lower) and np.all(point <= upper))


def _fallback_arm_state(csm: CSM, box: OperationBox):
    params = DexterousParameters.from_csm(csm)
    candidates = [
        DexterousMode3State(phi=0.35, theta1=0.72 * params.theta1_plus, L1=max(params.r1_minus_m * 0.72 * params.theta1_plus, 0.72 * params.L10_m), delta1=-1.0, theta2=0.82 * params.theta2_plus, delta2=0.55),
        DexterousMode3State(phi=-0.45, theta1=0.66 * params.theta1_plus, L1=max(params.r1_minus_m * 0.66 * params.theta1_plus, 0.78 * params.L10_m), delta1=1.15, theta2=0.74 * params.theta2_plus, delta2=-0.60),
        DexterousMode3State(phi=1.10, theta1=0.58 * params.theta1_plus, L1=max(params.r1_minus_m * 0.58 * params.theta1_plus, 0.68 * params.L10_m), delta1=0.35, theta2=0.88 * params.theta2_plus, delta2=-1.20),
    ]
    valid = [state for state in candidates if _point_inside_box(_state_position(csm, state), box, margin_ratio=0.0)]
    if not valid:
        return None
    return max(valid, key=lambda state: _score_overlay_state(csm, box, state))


def _select_arm_overlay_state(csm: CSM, box: OperationBox):
    params = DexterousParameters.from_csm(csm)
    rng = np.random.default_rng(ARM_RANDOM_SEED)
    best_state = None
    best_score = -float("inf")
    for _ in range(ARM_RANDOM_SAMPLES):
        state = _sample_random_mode3_state(rng, params)
        point_xyz, _ = analytic_fk_mode3(params, state)
        if not _point_inside_box(point_xyz, box):
            continue
        score = _score_overlay_state(csm, box, state)
        if score > best_score:
            best_score = score
            best_state = state
    if best_state is not None:
        return best_state
    fallback_state = _fallback_arm_state(csm, box)
    if fallback_state is not None:
        return fallback_state
    raise RuntimeError("Unable to choose an operation-box point with a reachable mode-3 arm pose.")


def _arm_geometry_mm(csm: CSM, box: OperationBox):
    arm_state = _select_arm_overlay_state(csm, box)
    overlay_csm = make_mode3_display_csm(csm, arm_state)
    return overlay_csm.get_visualization_segments()


def _draw_detailed_arm_3d(ax, geometry) -> None:
    visualizer = Visualizer(body_radius=1.7, disk_spacing=4.0, default_render_mode="detailed")

    radius_mm = 1.7
    tendon_radius_mm = radius_mm * visualizer.tendon_radius_ratio
    disk_radius_mm = radius_mm * 1.25
    tendon_offsets = [
        np.array([tendon_radius_mm, 0.0, 0.0]),
        np.array([0.0, tendon_radius_mm, 0.0]),
        np.array([-tendon_radius_mm, 0.0, 0.0]),
        np.array([0.0, -tendon_radius_mm, 0.0]),
    ]

    centerline = []
    for seg_idx, segment in enumerate(geometry["segments"]):
        points = MM_PER_M * np.asarray(segment["points"], dtype=float)
        rotations = np.asarray(segment["rotations"], dtype=float)
        if seg_idx > 0:
            points = points[1:]
            rotations = rotations[1:]
        centerline.append(points)

        segment_color = (
            visualizer.palette["seg1"] if segment["label"] == "seg1"
            else visualizer.palette["seg2"] if segment["label"] == "seg2"
            else visualizer.palette["straight"]
        )
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=segment_color, linewidth=3.0, alpha=0.95)

        for tendon_idx, offset in enumerate(tendon_offsets):
            tendon_points = points + np.einsum("nij,j->ni", rotations, offset)
            ax.plot(
                tendon_points[:, 0],
                tendon_points[:, 1],
                tendon_points[:, 2],
                color=visualizer.tendon_colors[tendon_idx],
                linewidth=1.2,
                alpha=0.9 if segment["kind"] == "arc" else 0.55,
            )

        scaled_segment = dict(segment)
        scaled_segment["length"] = MM_PER_M * float(segment["length"])
        scaled_segment["points"] = MM_PER_M * np.asarray(segment["points"], dtype=float)
        scaled_segment["T_start"] = np.asarray(segment["T_start"], dtype=float).copy()
        scaled_segment["T_end"] = np.asarray(segment["T_end"], dtype=float).copy()
        scaled_segment["T_start"][:3, 3] *= MM_PER_M
        scaled_segment["T_end"][:3, 3] *= MM_PER_M
        for center, rotation in visualizer._sample_disk_frames(scaled_segment):
            visualizer._draw_disk(ax, center, rotation, disk_radius_mm, visualizer.palette["disk"])

        if segment["label"] == "rigid" and segment["length"] > 0:
            visualizer._draw_disk(
                ax,
                MM_PER_M * np.asarray(segment["T_start"][:3, 3], dtype=float),
                np.asarray(segment["T_start"][:3, :3], dtype=float),
                disk_radius_mm,
                segment_color,
                alpha=0.22,
            )
            visualizer._draw_disk(
                ax,
                MM_PER_M * np.asarray(segment["T_end"][:3, 3], dtype=float),
                np.asarray(segment["T_end"][:3, :3], dtype=float),
                disk_radius_mm,
                segment_color,
                alpha=0.22,
            )

    if centerline:
        merged = np.vstack(centerline)
        ax.plot(
            merged[:, 0],
            merged[:, 1],
            merged[:, 2],
            color=visualizer.palette["backbone"],
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
        )

    tool_scaled = {
        "length": MM_PER_M * float(geometry["tool"]["length"]),
        "rotation": np.asarray(geometry["tool"]["rotation"], dtype=float),
        "start": MM_PER_M * np.asarray(geometry["tool"]["start"], dtype=float),
        "end": MM_PER_M * np.asarray(geometry["tool"]["end"], dtype=float),
    }
    visualizer._draw_tool(ax, tool_scaled, radius_mm * 1.05)


def _draw_detailed_arm_side(ax, geometry) -> None:
    visualizer = Visualizer(body_radius=1.7, disk_spacing=4.0, default_render_mode="detailed")
    radius_mm = 1.7
    tendon_radius_mm = radius_mm * visualizer.tendon_radius_ratio
    disk_radius_mm = radius_mm * 1.25
    tendon_offsets = [
        np.array([tendon_radius_mm, 0.0, 0.0]),
        np.array([0.0, tendon_radius_mm, 0.0]),
        np.array([-tendon_radius_mm, 0.0, 0.0]),
        np.array([0.0, -tendon_radius_mm, 0.0]),
    ]
    centerline = []
    for segment in geometry["segments"]:
        points = MM_PER_M * np.asarray(segment["points"], dtype=float)
        rotations = np.asarray(segment["rotations"], dtype=float)
        x = points[:, 0]
        z = points[:, 2]
        segment_color = (
            visualizer.palette["seg1"] if segment["label"] == "seg1"
            else visualizer.palette["seg2"] if segment["label"] == "seg2"
            else visualizer.palette["straight"]
        )
        ax.plot(x, z, color=segment_color, linewidth=2.8, alpha=0.95, zorder=6)
        centerline.append(np.column_stack((x, z)))

        for tendon_idx, offset in enumerate(tendon_offsets):
            tendon_points = points + np.einsum("nij,j->ni", rotations, offset)
            ax.plot(
                tendon_points[:, 0],
                tendon_points[:, 2],
                color=visualizer.tendon_colors[tendon_idx],
                linewidth=1.0,
                alpha=0.88 if segment["kind"] == "arc" else 0.5,
                zorder=7,
            )

        scaled_segment = dict(segment)
        scaled_segment["length"] = MM_PER_M * float(segment["length"])
        scaled_segment["points"] = MM_PER_M * np.asarray(segment["points"], dtype=float)
        scaled_segment["T_start"] = np.asarray(segment["T_start"], dtype=float).copy()
        scaled_segment["T_end"] = np.asarray(segment["T_end"], dtype=float).copy()
        scaled_segment["T_start"][:3, 3] *= MM_PER_M
        scaled_segment["T_end"][:3, 3] *= MM_PER_M
        for center, rotation in visualizer._sample_disk_frames(scaled_segment):
            angles = np.linspace(0.0, 2.0 * np.pi, 40)
            local = np.vstack(
                [
                    disk_radius_mm * np.cos(angles),
                    disk_radius_mm * np.sin(angles),
                    np.zeros_like(angles),
                ]
            )
            world = center[:, None] + rotation @ local
            ax.plot(
                world[0],
                world[2],
                color=visualizer.palette["disk"],
                linewidth=0.9,
                alpha=0.55,
                zorder=5,
            )

    tool_length_mm = MM_PER_M * float(geometry["tool"]["length"])
    if tool_length_mm > 0:
        tool_rotation = np.asarray(geometry["tool"]["rotation"], dtype=float)
        tool_start = MM_PER_M * np.asarray(geometry["tool"]["start"], dtype=float)
        base_radius_mm = radius_mm * 1.05 * 0.82
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        base_circle_local = np.vstack(
            [
                base_radius_mm * np.cos(angles),
                base_radius_mm * np.sin(angles),
                np.zeros_like(angles),
            ]
        )
        base_circle_world = (tool_start[:, None] + tool_rotation @ base_circle_local).T
        tip_world = tool_start + tool_rotation @ np.array([0.0, 0.0, tool_length_mm])
        closed_circle = np.vstack([base_circle_world, base_circle_world[0]])
        ax.plot(
            closed_circle[:, 0],
            closed_circle[:, 2],
            color=visualizer.palette["tool"],
            linewidth=1.0,
            alpha=0.9,
            zorder=7,
        )
        for idx in range(0, len(base_circle_world), 2):
            edge = np.vstack([base_circle_world[idx], tip_world])
            ax.plot(
                edge[:, 0],
                edge[:, 2],
                color=visualizer.palette["tool"],
                linewidth=1.0,
                alpha=0.9,
                zorder=7,
            )

    if centerline:
        merged = np.vstack(centerline)
        ax.plot(
            merged[:, 0],
            merged[:, 1],
            color=visualizer.palette["backbone"],
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            zorder=5,
        )


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


def _set_artists_zorder(artists, zorder: float) -> None:
    for artist in artists:
        try:
            artist.set_zorder(zorder)
        except Exception:
            continue


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
    arm_geometry = _arm_geometry_mm(csm, box)

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
    if hasattr(ax_main, "computed_zorder"):
        ax_main.computed_zorder = False

    collection_count = len(ax_main.collections)
    line_count = len(ax_main.lines)
    draw_revolved_profile(ax_main, profile, color=style.reachable_color, options=plot_options)
    workspace_artists = list(ax_main.collections[collection_count:]) + list(ax_main.lines[line_count:])

    collection_count = len(ax_main.collections)
    line_count = len(ax_main.lines)
    draw_operation_box(
        ax_main,
        _box_to_display_units(box, 1000.0),
        face_color=style.box_face_color,
        edge_color=style.box_edge_color,
        alpha=style.box_alpha_3d,
        linewidth=style.box_linewidth_3d,
    )
    box_artists = list(ax_main.collections[collection_count:]) + list(ax_main.lines[line_count:])

    collection_count = len(ax_main.collections)
    line_count = len(ax_main.lines)
    _draw_detailed_arm_3d(ax_main, arm_geometry)
    arm_artists = list(ax_main.collections[collection_count:]) + list(ax_main.lines[line_count:])

    _set_artists_zorder(workspace_artists, 1.0)
    _set_artists_zorder(box_artists, 2.0)
    _set_artists_zorder(arm_artists, 20.0)

    configure_3d_axes(ax_main, [profile])
    ax_main.view_init(elev=18, azim=-38)
    ax_main.set_title(style.main_title, pad=10.0)
    ax_main.grid(True, alpha=style.grid_alpha)

    ax_side = fig.add_subplot(gs[0, 1])
    draw_side_profile(ax_side, profile, color=style.reachable_color, label="Mode3", options=plot_options)
    _draw_side_box(ax_side, _box_to_display_units(box, 1000.0), style)
    _draw_detailed_arm_side(ax_side, arm_geometry)
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
    parser.add_argument("--save-box-info", help="Optional JSON path to save the fitted operation-box definition.")
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
    if args.save_box_info:
        box_info_path = Path(args.save_box_info)
        _save_box_info(
            box_info_path,
            csm=csm,
            config_path=args.config,
            requested_box_size_mm=requested_box_size_mm,
            top_margin_mm=float(args.top_margin_mm),
            box=box,
            box_scale=box_scale,
        )
        print(f"Saved box info to: {box_info_path.resolve()}")
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
