# CSM

Continuum Segment Manipulator (CSM) —— 连续段机械臂运动学建模与控制仿真。

## 项目结构

```
CSM/
├── src/
│   └── csm/                    # 核心 Python 包
│       ├── __init__.py         # 包入口，导出 CSM / Visualizer / LineGenerator
│       ├── model.py            # CSM 机械臂运动学模型与控制器（核心类）
│       ├── line_generator.py   # 3D 曲线/弧线生成工具（LineGenerator 类）
│       ├── visualizer.py       # 基于 LineGenerator 的机械臂可视化（Visualizer 类）
│       └── utils.py            # 通用工具函数（角速度计算、数据加载等）
│
├── scripts/                    # 独立运行脚本
│   ├── gen_workspace.py        # 生成工作空间采样数据（输出 data/workspace_data.json）
│   ├── run_experiment.py       # 批量目标跟踪实验，记录成功/失败数据
│   ├── display.py              # 实时动画演示机械臂目标跟踪过程
│   ├── run_traj.py             # 单段轨迹演示，绘制误差与关节变量曲线
│   ├── re_experiment.py        # 重放失败实验数据进行复现分析
│   ├── plot_workspace.py       # 可视化工作空间点云（凸包线框/曲面）
│   ├── analyse_single.py       # 动画回放单条失败案例
│   ├── fail_type_statics.py    # 失败案例参数统计与分类分析
│   ├── fail_type_vis.py        # 失败案例雷达图可视化与机械臂对比展示
│   ├── step_distribution.py    # 成功案例步数分布直方图统计
│   └── confirm.py              # 雅可比矩阵符号推导验证（SymPy）
│
├── config/
│   └── csm_config.yaml         # 机器人物理参数与控制参数配置
│
├── utils/
│   └── LTI_resp_laplace.py     # 独立工具：线性时不变系统拉普拉斯响应计算
│
├── data/                       # 实验数据（运行脚本后生成，不纳入版本控制）
│   ├── workspace_data.json
│   ├── successes_*.json
│   └── failures_*.json
│
├── imgs/                       # 文档图片
├── pyproject.toml              # 项目构建配置
├── requirements.txt            # 依赖列表
├── .gitignore
└── README.md
```

## 安装

```bash
pip install -e .
```

## 快速开始

```python
from csm import CSM
import numpy as np

csm = CSM.from_config("config/csm_cfg_3.4mm.yaml")
# 或直接传参
csm = CSM(L_10=0.04, L_20=0.06, L_r0=0.02, L_s0=0.15, L_tool=0.01)
```

## 运行脚本

所有脚本在项目根目录下执行：

```bash
conda activate csm

# 生成工作空间数据（需先运行）
python3 scripts/gen_workspace.py

# 实时动画演示
python3 scripts/display.py

# 单段轨迹演示
python3 scripts/run_traj.py

# 交互式末端控制窗口
python3 scripts/example_control_interface.py

# 批量实验
python3 scripts/run_experiment.py
```

## 交互式控制窗口

现在可以通过 [`src/csm/control_interface.py`](/home/winslow/RII/CSM/src/csm/control_interface.py) 启动一个实时交互控制窗口。

设计思路：

- 保留当前已有的雅可比逆解。
- 每次键盘输入只产生一个很小的末端位姿增量。
- 求解器每个显示帧内部执行多个 `delta_t` 微步，并始终从当前构型附近继续迭代。

这很适合你现在这种“实时拖动末端、每一步都是微小位移”的交互控制方式。

### 运行示例

```bash
python3 scripts/example_control_interface.py
```

### 控制方式

- `W / S`: 沿世界坐标 `+Y / -Y` 平移
- `A / D`: 沿世界坐标 `+X / -X` 平移
- `Q / E`: 沿世界坐标 `+Z / -Z` 平移
- `J / L`: 控制偏航
- `I / K`: 控制俯仰
- 鼠标仅用于旋转观察视角
- `R`: 复位

说明：

- 当前末端朝向在项目里是一个方向向量，因此只有 2 个独立姿态自由度。
- 所以这版交互界面只有两对朝向控制键，这是和当前模型表示一致的，不是少实现了一对。

### 代码示例

```python
from csm import CSM, launch_control_interface

csm = CSM.from_config("config/csm_cfg_3mm.yaml")
launch_control_interface(
    csm,
    render_mode="detailed",
    frame_interval_ms=20,
    linear_speed=0.04,
    angular_speed=2.0,
    orientation_key_speed=1.8,
)
```

更完整的说明见 [control_interface.md](/home/winslow/RII/CSM/docs/control_interface.md)。

## 机械臂可视化

`CSM.plot_manipulator()` 现在支持两种可视化风格，通过 `render_mode` 参数控制：

- `render_mode="simple"`：原来的简洁画法，只显示中心线和工具线，适合快速调试。
- `render_mode="detailed"`：增强画法，显示 backbone、圆周排列驱动丝、沿弧分布的间隔盘和末端工具。

示例：

```python
import matplotlib.pyplot as plt
from csm import CSM

csm = CSM.from_config("config/csm_cfg_3.4mm.yaml")
csm.target_pose = csm.pose.copy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 原始简洁风格
csm.plot_manipulator(ax, render_mode="simple")

# 或增强结构化风格
# csm.plot_manipulator(ax, render_mode="detailed")

plt.show()
```

在动画脚本里也一样，例如在 [`scripts/display.py`](/home/winslow/RII/CSM/scripts/display.py) 中，把：

```python
csm.plot_manipulator(ax)
```

改成：

```python
csm.plot_manipulator(ax, render_mode="simple")
```

或：

```python
csm.plot_manipulator(ax, render_mode="detailed")
```

## 工作空间生成与绘制

工作空间相关脚本分为两步：
1. 先运行 [`scripts/gen_workspace.py`](/home/winslow/RII/CSM/scripts/gen_workspace.py) 生成采样数据。
2. 再运行 [`scripts/plot_workspace.py`](/home/winslow/RII/CSM/scripts/plot_workspace.py) 进行三维可视化。

### 1. 生成工作空间数据

[`scripts/gen_workspace.py`](/home/winslow/RII/CSM/scripts/gen_workspace.py) 会读取配置文件，遍历各模式关节参数，并将结果保存到 `./data/workspace_data_3.4mm.json`。

脚本内可直接修改以下参数：

- `config_name`：选择配置文件，例如 `csm_cfg_3.4mm.yaml`。
- `num_samples_per_mode`：随机采样时各模式样本数。
- `sampling_method`：采样方式，支持 `"random"` 和 `"uniform"`。
- `uniform_grid_res`：均匀采样时每个维度的基础分辨率。

两种采样方式的区别：

- `"random"`：Monte Carlo 随机采样，速度更友好，适合快速生成大量点云。
- `"uniform"`：均匀网格遍历，边界更规整，更适合后续绘制平滑外壳；但模式 3/4 组合数增长很快，计算时间会明显增加。

运行方式：

```bash
python3 scripts/gen_workspace.py
```

推荐：

- 想保留真实凹陷结构时，使用 `"random"` + 大样本数，再配合 `cloud` 渲染。
- 想得到更均匀的外壳边界时，使用 `"uniform"`，并适当降低 `uniform_grid_res` 控制运行时间。

### 2. 绘制工作空间

[`scripts/plot_workspace.py`](/home/winslow/RII/CSM/scripts/plot_workspace.py) 会读取 `json_path` 指定的数据文件，并按模式绘制三维工作空间。

脚本内可直接修改以下参数：

- `json_path`：输入的数据文件路径。
- `RENDER_MODE`：渲染模式。
- `SEPARATE_PLOTS`：是否拆成 4 个子图分别显示各模式。
- `SURF_ALPHA`、`WIRE_ALPHA`、`WIRE_LW`：透明度和线宽参数。

`RENDER_MODE` 支持以下选项：

- `"wire"`：只显示凸包线框。
- `"surface"`：显示无光影的半透明凸包面片。
- `"both"`：同时显示面片和线框。
- `"smooth_surface"`：显示带光照阴影的平滑半透明外壳，视觉上比普通 `surface` 更接近论文中的实体感。
- `"cloud"`：高密度低透明度点云体渲染，最适合保留底部凹陷和不可达空腔，不会被凸包“封死”。

运行方式：

```bash
python3 scripts/plot_workspace.py
```

推荐组合：

- 论文 Fig. 6 风格外壳效果：`sampling_method="uniform"` + `RENDER_MODE="smooth_surface"`。
- 强调真实可达体积和凹陷结构：`sampling_method="random"` + `RENDER_MODE="cloud"`。
- 想保留原始对比方式：继续使用 `RENDER_MODE="wire"` 或 `"both"`。

## Workspace Boundary Scan

现在 `plot_workspace_boundary_scan.py` 已经被整理成包内功能，可以直接从 `csm.workspace_boundary_scan` 调用。

常用入口：

```python
from csm.workspace_boundary_scan import (
    BoundaryScanOptions,
    BoundaryScanPlotOptions,
    build_workspace_profiles,
    plot_workspace_profiles,
)
```

说明文档见 [workspace_boundary_scan.md](/home/winslow/RII/CSM/docs/workspace_boundary_scan.md)，
示例脚本见 [example_workspace_boundary_scan.py](/home/winslow/RII/CSM/scripts/example_workspace_boundary_scan.py)。

## 配置文件

[`config/csm_cfg_3.4mm.yaml`](/home/winslow/RII/CSM/config/csm_cfg_3.4mm.yaml) 包含机器人物理参数与控制参数，例如：

```yaml
robot:
  L_10: 0.04      # 第一段最大长度 (m)
  L_20: 0.06      # 第二段最大长度 (m)
  L_r0: 0.02      # 刚性连接段长度 (m)
  L_s0: 0.15      # 滑动段最大长度 (m)
  theta1_max: 1.5708
  theta2_max: 2.0944

control:
  delta_t: 0.001  # 控制步长 (s)
  v_lim: 0.2      # 线速度限制 (m/s)
  w_lim: 2.0      # 角速度限制 (rad/s)
  max_steps: 8000
```

## pinocchio

安装
conda install pinocchio -c conda-forge
安装机器人模型数据包
conda install -c conda-forge example-robot-data
补装数据配套的的 Python loaders 包
conda install -c conda-forge example-robot-data-loaders
在 Pinocchio 中，最主流且对 Python 最友好的可视化工具是 Meshcat。它可以在你的浏览器里直接渲染出 3D 模型，轻量且不需要复杂的 GUI 依赖。
为了运行完整的示例，你需要先安装 Meshcat。请在你的 csm 环境中运行以下命令：
conda install -c conda-forge meshcat-python

## 文件说明
### cmap_build_fk_mp.py
该脚本用于构建 CSM 的正向运动学能力图（FK-Capability Map）。它利用多进程并行计算，显著提高了计算效率。
彻底解耦 Halton 序列（交织分配）：进程 0、1、2 拿到的 index 就像发牌一样轮流跳跃。因为 Halton 函数根据索引现算数值，它们相互不干扰，结果与单进程跑出来完全一模一样。

极速共享内存写入：mp.RawArray 将一块真正的 C 级连续内存暴露给所有进程，没有 Python 的对象锁机制干扰。多个进程就算同时向图谱的某块区域写入 True 也不会出问题，因为按位或运算天生是线程安全的。

主进程统筹UI：放弃让工作进程抢占控制台，改为由主进程每 1 秒读取一下各进程的 shared_steps_counter（做了几次 FK）和 共享内存（填了多少区块），统一更新到 Tqdm 进度条上，画面非常干净漂亮。

木桶效应安全锚点：在 Ctrl+C 断开时，有的进程可能多跑了 3 圈，有的少跑了 1 圈。程序会在最后找到跑得最少的那个进程圈数，计算出 safe_step_to_save。这就好比进度存档，绝对不会留下任何“跳步没算”的空洞。由于多跑的 True 已经被写入到 cmap 了，所以即便有少许重复判定，但不会有重复求解，也不会对最终图谱产生影响。
