"""
Structured workspace boundary scan API.
"""
from .core import (
    BoundaryPrimitive,
    BoundaryScanOptions,
    WorkspaceAnimationCurve,
    WorkspaceAnimationData,
    WorkspaceAnimationStage,
    WorkspaceProfile,
    build_mode0_profile,
    build_mode1_profile,
    build_mode2_profile,
    build_mode3_profile,
    build_mode4_profile,
    build_workspace_animation_data,
    build_workspace_profile_animation_data,
    build_workspace_profile,
    build_workspace_profiles,
    symmetric_fill_polygon,
)
from .animation import (
    BoundaryScanAnimationOptions,
    render_workspace_scan_animation,
)
from .plotting import (
    BoundaryScanPlotOptions,
    has_interactive_display,
    plot_workspace_profiles,
    plot_mode3_primitives_debug,
)

__all__ = [
    "BoundaryPrimitive",
    "BoundaryScanAnimationOptions",
    "BoundaryScanOptions",
    "BoundaryScanPlotOptions",
    "WorkspaceAnimationCurve",
    "WorkspaceAnimationData",
    "WorkspaceAnimationStage",
    "WorkspaceProfile",
    "build_mode0_profile",
    "build_mode1_profile",
    "build_mode2_profile",
    "build_mode3_profile",
    "build_mode4_profile",
    "build_workspace_animation_data",
    "build_workspace_profile_animation_data",
    "build_workspace_profile",
    "build_workspace_profiles",
    "has_interactive_display",
    "plot_workspace_profiles",
    "plot_mode3_primitives_debug",
    "render_workspace_scan_animation",
    "symmetric_fill_polygon",
]
