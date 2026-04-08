"""
Structured workspace boundary scan API.
"""
from .core import (
    BoundaryPrimitive,
    BoundaryScanOptions,
    WorkspaceProfile,
    build_mode0_profile,
    build_mode1_profile,
    build_mode2_profile,
    build_mode3_profile,
    build_mode4_profile,
    build_workspace_profile,
    build_workspace_profiles,
    symmetric_fill_polygon,
)
from .plotting import (
    BoundaryScanPlotOptions,
    has_interactive_display,
    plot_workspace_profiles,
)

__all__ = [
    "BoundaryPrimitive",
    "BoundaryScanOptions",
    "BoundaryScanPlotOptions",
    "WorkspaceProfile",
    "build_mode0_profile",
    "build_mode1_profile",
    "build_mode2_profile",
    "build_mode3_profile",
    "build_mode4_profile",
    "build_workspace_profile",
    "build_workspace_profiles",
    "has_interactive_display",
    "plot_workspace_profiles",
    "symmetric_fill_polygon",
]
