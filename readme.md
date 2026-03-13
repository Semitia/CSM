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

csm = CSM.from_config("config/csm_config.yaml")
# 或直接传参
csm = CSM(L_10=0.04, L_20=0.06, L_r0=0.02, L_s0=0.15)
```

## 运行脚本

所有脚本在项目根目录下执行：

```bash
# 生成工作空间数据（需先运行）
python scripts/gen_workspace.py

# 实时动画演示
python scripts/display.py

# 单段轨迹演示
python scripts/run_traj.py

# 批量实验
python scripts/run_experiment.py
```

## 配置文件

[`config/csm_config.yaml`](config/csm_config.yaml) 包含机器人物理参数与控制参数：

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

## ToDo

- [x] 3.4mm手术工具实际参数
- [x] 跑3.4mm，确认ls为0时是否有问题
- [x] 添加末端工具
- [ ] 普通串联机械臂 dexterous workspace 复现
- [ ] 基于采样的 普通串联机械臂 dexterous workspace
- [ ] CSM 对接 pinocchio 求解框架
- [ ] CSM dexterous workspace 实验
- [ ] CSM 兼容 pinocchio CAD 构建与可视化

## 文件说明
### cmap_build_fk_mp.py
该脚本用于构建 CSM 的正向运动学能力图（FK-Capability Map）。它利用多进程并行计算，显著提高了计算效率。
彻底解耦 Halton 序列（交织分配）：进程 0、1、2 拿到的 index 就像发牌一样轮流跳跃。因为 Halton 函数根据索引现算数值，它们相互不干扰，结果与单进程跑出来完全一模一样。

极速共享内存写入：mp.RawArray 将一块真正的 C 级连续内存暴露给所有进程，没有 Python 的对象锁机制干扰。多个进程就算同时向图谱的某块区域写入 True 也不会出问题，因为按位或运算天生是线程安全的。

主进程统筹UI：放弃让工作进程抢占控制台，改为由主进程每 1 秒读取一下各进程的 shared_steps_counter（做了几次 FK）和 共享内存（填了多少区块），统一更新到 Tqdm 进度条上，画面非常干净漂亮。

木桶效应安全锚点：在 Ctrl+C 断开时，有的进程可能多跑了 3 圈，有的少跑了 1 圈。程序会在最后找到跑得最少的那个进程圈数，计算出 safe_step_to_save。这就好比进度存档，绝对不会留下任何“跳步没算”的空洞。由于多跑的 True 已经被写入到 cmap 了，所以即便有少许重复计算，也绝对不会对最终图谱产生影响。

