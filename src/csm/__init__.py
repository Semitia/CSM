"""
Module: __init__.py
Description: Initializes the CSM package and exports key components like CSM, Visualizer, and LineGenerator.
"""
from .model import CSM
from .visualizer import Visualizer
from .line_generator import LineGenerator
from .vsik import VSIKSolver, CSMParameters, compute_fk
from .dex_workspace import DexterousWorkspace, compute_dexterous_workspace_boundary_points
