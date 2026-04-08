"""
Module: control_interface_demo.py
Description: Launch an interactive simulation control window for the CSM model.
"""
import sys
from pathlib import Path

import yaml

from csm import CSM, launch_control_interface


def main():
    config_path = Path("config/csm_cfg_3mm.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    control_cfg = cfg.get("control", {})
    csm = CSM.from_config(config_path)
    launch_control_interface(
        csm,
        render_mode="detailed",
        frame_interval_ms=20,
        linear_speed=control_cfg.get("v_lim", 0.04),
        angular_speed=control_cfg.get("w_lim", 2.0),
        position_gain=15.0,
        orientation_gain=12.0,
        orientation_key_speed=1.8,
        title=f"CSM Interactive Control - {config_path.name}",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
