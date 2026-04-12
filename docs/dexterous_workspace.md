# Dexterous Workspace

`csm.dexterous_workspace` 提供了当前项目里用于 `3mm` 配置、`mode 3` 场景的局部 dexterous workspace 计算与论文风格绘图能力。

当前版本的定位很明确：

- 只支持 `3mm`
- 只支持 `CSM mode 3`
- 输入 probe 时只给位置
- 默认走解析/半解析主线
- 可以按需开启 Jacobian fallback 做校验或补洞

## 公开接口

常用入口：

```python
from csm import (
    DexterousPlotOptions,
    build_dexterous_probe,
    render_dexterous_figure,
)
```

核心数据结构：

- `DexterousProbe`
  单个 probe 点对应的局部 dexterous 球面区域、边界、展示姿态和拟合结果。
- `DexterousPlotOptions`
  控制论文风格球体、可达域 patch、视角和输出路径。
- `DexterousParameters`
  把当前 `CSM` 的 `3mm + mode3` 语义映射成 dexterous 求解所需参数。

## 设计说明

模块内部拆成三层：

- [`kinematics.py`](/home/winslow/RII/CSM/src/csm/dexterous_workspace/kinematics.py)
  - `CSM mode3 <-> CI-1` 语义映射
  - meter / millimeter 单位转换集中管理
  - mode3 FK 对齐辅助函数
- [`analytic.py`](/home/winslow/RII/CSM/src/csm/dexterous_workspace/analytic.py)
  - 迁移并整理了 `CI-1` 局部 dexterous feasible area / boundary 的解析与半解析求解逻辑
  - 不依赖 `ref/CSM-DW` 导入
- [`fallback.py`](/home/winslow/RII/CSM/src/csm/dexterous_workspace/fallback.py)
  - 基于当前 `CSM` Jacobian 迭代的 bootstrap 与方向扫描备选
- [`plotting.py`](/home/winslow/RII/CSM/src/csm/dexterous_workspace/plotting.py)
  - 论文风格局部球体渲染
  - 优先渲染球冠，拟合不理想时退回平滑 patch

## 快速使用

### 1. 单个 probe

```python
from pathlib import Path

import numpy as np

from csm import CSM, build_dexterous_probe

csm = CSM.from_config(Path("./config/csm_cfg_3mm.yaml"))

probe = build_dexterous_probe(
    np.array([10.87, 10.28, 29.77]) / 1000.0,
    csm=csm,
    config="3mm",
    method="analytic",
    validate_with_fallback=True,
    label="P1",
)

print(probe.status)
print(probe.feasible_directions_world.shape)
print(probe.cap_angular_radius)
```

### 2. 渲染论文风格图

```python
from pathlib import Path

from csm import (
    CSM,
    DexterousPlotOptions,
    build_dexterous_probe,
    render_dexterous_figure,
)

csm = CSM.from_config(Path("./config/csm_cfg_3mm.yaml"))

probes = [
    build_dexterous_probe(
        [10.87 / 1000.0, 10.28 / 1000.0, 29.77 / 1000.0],
        csm=csm,
        config="3mm",
        method="analytic",
        validate_with_fallback=True,
        label="P1",
    ),
]

render_dexterous_figure(
    probes,
    csm=csm,
    options=DexterousPlotOptions(
        output_path=Path("./data/dexterous_workspace_3mm.png"),
        show_figure=False,
    ),
)
```

## 脚本入口

可以直接运行：

```bash
conda run -n csm python scripts/render_dexterous_workspace.py --hide
```

常用参数：

- `--config`
  配置文件路径，默认 `config/csm_cfg_3mm.yaml`
- `--points-mm`
  probe 点列表，单位是毫米
- `--method`
  `analytic` 或 `fallback`
- `--validate-with-fallback`
  解析结果后再做 Jacobian fallback 校验
- `--output`
  输出图片路径

示例：

```bash
conda run -n csm python scripts/render_dexterous_workspace.py \
  --points-mm '[[10.87, 10.28, 29.77], [8.0, 12.0, 30.5]]' \
  --validate-with-fallback \
  --hide
```

## 当前限制

- 只做 `3mm`
- 只做 `mode 3`
- 不支持 `mode 1/2`
- 不支持 `mode 4`
- fallback 属于数值校验/补洞路径，速度明显慢于解析主线
- 如果 `config/csm_cfg_3mm.yaml` 里没有显式填写 `ri_min`，当前实现会回退到仓库内已有的 3mm 默认值 `0.0054038 m`

## 验证文件

- 测试见 [`tests/test_dexterous_workspace.py`](/home/winslow/RII/CSM/tests/test_dexterous_workspace.py)
- 渲染脚本见 [`scripts/render_dexterous_workspace.py`](/home/winslow/RII/CSM/scripts/render_dexterous_workspace.py)


python scripts/example_mode3_translation_figure.py --save-box-info data/mode3_box.json

python scripts/example_mode3_dexterous_figure.py --load-box-info data/mode3_box.json
