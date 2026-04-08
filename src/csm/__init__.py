"""
Module: __init__.py
Description: Initializes the CSM package and exports key components like CSM, Visualizer, and LineGenerator.
"""
from .model import CSM
from .control_interface import ControlInterface, launch_control_interface
from .visualizer import Visualizer
from .line_generator import LineGenerator
from .workspace_boundary_scan import (
    BoundaryPrimitive,
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    WorkspaceProfile,
    build_mode0_profile,
    build_mode1_profile,
    build_mode2_profile,
    build_mode3_profile,
    build_mode4_profile,
    build_workspace_profile,
    build_workspace_profiles,
    plot_workspace_profiles,
)
