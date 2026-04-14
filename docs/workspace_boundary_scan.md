# Workspace Boundary Scan

`csm.workspace_boundary_scan` 把原来的 `plot_workspace_boundary_scan.py` 重构成了可复用的包内功能，分成两层：

- `core.py`
  负责边界扫描、模式轮廓构建、`mode0` 路径搜索、数据结构定义。
- `plotting.py`
  负责侧视图/旋转体绘制、图像保存、debug 图输出。

## 公开接口

常用入口：

```python
from csm.workspace_boundary_scan import (
    BoundaryScanAnimationOptions,
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    build_workspace_animation_data,
    build_workspace_profiles,
    plot_workspace_profiles,
    render_workspace_scan_animation,
)
```

核心数据结构：

- `WorkspaceProfile`
  单个 mode 的边界轮廓和不可达区域数据。
- `BoundaryScanOptions`
  控制边界扫描采样密度与 `mode0` 搜索参数。
- `BoundaryScanPlotOptions`
  控制绘图样式、输出路径、debug 图保存等。

## 快速使用

```python
from pathlib import Path

from csm import CSM
from csm.workspace_boundary_scan import (
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    build_workspace_profiles,
    plot_workspace_profiles,
)

csm = CSM.from_config(Path("./config/csm_cfg_3mm.yaml"))

profiles = build_workspace_profiles(
    csm,
    modes=[0, 1, 2, 3],
    options=BoundaryScanOptions(
        length_samples=180,
        angle_samples=180,
        mode0_debug_output_dir=Path("./data/workspace_boundary_scan_debug"),
    ),
)

plot_workspace_profiles(
    profiles,
    BoundaryScanPlotOptions(
        output_path=Path("./data/workspace_boundary_scan.png"),
        debug_output_dir=Path("./data/workspace_boundary_scan_debug"),
        show_figure=True,
    ),
)
```

## 过程动画

过程动画是附加输出，不会改变原有静态 `build_workspace_profiles(...)` 和
`plot_workspace_profiles(...)` 的使用方式。默认推荐导出 GIF，并把动画逻辑放在独立渲染入口：

```python
animation_data = build_workspace_animation_data(
    csm,
    modes=[0, 1, 2, 3, 4],
    options=BoundaryScanOptions(length_samples=180, angle_samples=180),
)

render_workspace_scan_animation(
    profiles,
    animation_data,
    BoundaryScanAnimationOptions(
        enabled=True,
        output_path=Path("./data/workspace_boundary_scan.gif"),
        fps=12,
        show_figure=False,
    ),
)
```

说明：

- 动画默认渲染 2D side view 的边界生成过程，不做 3D revolve 动画。
- `enabled=False` 时不会触发动画渲染。
- 如果 `show_figure=False`，则必须提供 `output_path`。
- GIF 导出依赖 `matplotlib` 的 Pillow writer；环境中需要安装 `Pillow`。

## 文件说明

- [core.py](/home/winslow/RII/CSM/src/csm/workspace_boundary_scan/core.py)
  轮廓构建与 `mode0` 搜索。
- [animation.py](/home/winslow/RII/CSM/src/csm/workspace_boundary_scan/animation.py)
  动画阶段渲染与 GIF 导出。
- [plotting.py](/home/winslow/RII/CSM/src/csm/workspace_boundary_scan/plotting.py)
  绘图、坐标轴配置、debug 图保存。
- [__init__.py](/home/winslow/RII/CSM/src/csm/workspace_boundary_scan/__init__.py)
  对外导出的统一入口。
- [plot_workspace_boundary_scan.py](/home/winslow/RII/CSM/scripts/plot_workspace_boundary_scan.py)
  面向当前项目的薄包装脚本。
- [example_workspace_boundary_scan.py](/home/winslow/RII/CSM/scripts/example_workspace_boundary_scan.py)
  独立的最小示例脚本。

## 参数建议

- 日常出图：
  `length_samples=180~240`，`angle_samples=180~240`
- `mode0` 想更快：
  调低 `BoundaryScanOptions.mode0_route_length_samples` 和 `mode0_route_angle_samples`
- `mode0` 想更稳：
  可以微调 `mode0_node_merge_tol` 和 `mode0_endpoint_snap_tol`
