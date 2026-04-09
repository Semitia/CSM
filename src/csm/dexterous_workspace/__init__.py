"""
Local dexterous-workspace helpers for the project's 3mm/mode3 workflow.
"""
from .analytic import AnalyticDexterousWorkspace, AnalyticRegion
from .core import DexterousProbe, build_dexterous_probe
from .fallback import FallbackScanResult, bootstrap_position_reachable_state, scan_directions
from .kinematics import (
    DexterousMode3State,
    DexterousParameters,
    analytic_fk_mode3,
    clone_csm,
    make_mode3_display_csm,
    mode3_state_from_csm,
)
from .plotting import DexterousPlotOptions, plot_dexterous_probe, render_dexterous_figure, save_probe_debug_figure

__all__ = [
    "AnalyticDexterousWorkspace",
    "AnalyticRegion",
    "DexterousMode3State",
    "DexterousParameters",
    "DexterousPlotOptions",
    "DexterousProbe",
    "FallbackScanResult",
    "analytic_fk_mode3",
    "bootstrap_position_reachable_state",
    "build_dexterous_probe",
    "clone_csm",
    "make_mode3_display_csm",
    "mode3_state_from_csm",
    "plot_dexterous_probe",
    "render_dexterous_figure",
    "save_probe_debug_figure",
    "scan_directions",
]
