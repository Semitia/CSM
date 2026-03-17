# 机器人 Capability Map 构建工具 - 重构版

## 概述

本目录包含用于构建机器人能力图（Capability Map）的统一工具。重构后的代码通过面向对象设计，将不同类型的机器人（UR5、Panda、CSM 等）封装为统一接口，实现了代码复用和易用性。

## 目录结构

```
scripts/dexterous_analyse/
├── robots/                      # 机器人封装模块
│   ├── __init__.py             # 注册表和工厂函数
│   ├── base_robot.py           # BaseRobot 抽象基类
│   ├── pinocchio_robot.py      # Pinocchio 机器人封装（UR5/Panda）
│   └── csm_robot.py            # CSM 连续体机器人封装
├── build_fk_cmap_mp.py         # 统一 FK 多进程构建入口
├── build_ik_cmap_mp.py         # 统一 IK 多进程构建入口
├── ws_discretizer.py           # 工作空间离散化器
├── cmap_analyzer.py            # 能力图分析工具
└── pinocchio_test.py           # Pinocchio 测试脚本
```

## 快速开始

### FK Capability Map 构建

只需修改 [`build_fk_cmap_mp.py`](build_fk_cmap_mp.py:95) 中的 `ROBOT_NAME` 变量：

```python
# =============== 只需修改这里 ==================
ROBOT_NAME = "ur5"   # 可选: 'ur5', 'panda', 'csm', 'csm_0', 'csm_tool'
max_fk = 200_000_000
USE_RICH = True
# ==========================
```

然后运行：

```bash
cd scripts/dexterous_analyse
python build_fk_cmap_mp.py
```

### IK Capability Map 构建

修改 [`build_ik_cmap_mp.py`](build_ik_cmap_mp.py:100) 中的 `ROBOT_NAME` 变量：

```python
# ==== 只需修改这里 ==================
ROBOT_NAME = "ur5"  # 可选: 'ur5', 'panda' (CSM 不支持 IK)
total_samples = 100_000
# ======================
```

然后运行：

```bash
cd scripts/dexterous_analyse
python build_ik_cmap_mp.py
```

## 支持的机器人

| 机器人名称 | 类型 | FK 支持 | IK 支持 | 配置文件 |
|---------|------|---------|---------|----------|
| `ur5` | Pinocchio | ✓ | ✓ | `config/discr_cfg_ur5.json` |
| `panda` | Pinocchio | ✓ | ✓ | `config/discr_cfg_panda.json` |
| `csm` | CSM | ✓ | ✗ | `config/discr_cfg_csm.json` + `config/csm_cfg_3.4mm.yaml` |
| `csm_0` | CSM | ✓ | ✗ | `config/discr_cfg_csm.json` + `config/csm_cfg_0.yaml` |
| `csm_tool` | CSM | ✓ | ✗ | `config/discr_cfg_csm.json` + `config/csm_cfg_0_tool.yaml` |

## 添加新机器人

### 1. 基于 Pinocchio 的机器人（如 UR10、Kinova 等）

在 [`robots/__init__.py`](robots/__init__.py:25) 的 `ROBOT_REGISTRY` 中添加：

```python
"ur10": lambda: PinocchioRobot(
    robot_name="ur10",
    tcp_frame_name=None,  # 或指定末端 frame 名称
),
```

同时在 `DISCR_CONFIG_REGISTRY` 和 `SAVE_PATH_REGISTRY` 中添加对应条目。

### 2. 新的机器人类型

1. 在 `robots/` 目录下创建新的类文件（如 `my_robot.py`）
2. 继承 [`BaseRobot`](robots/base_robot.py:10) 并实现所有抽象方法
3. 在 [`robots/__init__.py`](robots/__init__.py:25) 中注册

## 核心设计

### BaseRobot 接口

所有机器人类必须实现以下接口：

- `load()`: 加载机器人模型（在 worker 进程内调用）
- `fk(q)`: 正向运动学，返回 `(tcp_position, tcp_rotation)`
- `ik(target_pose, q_init)`: 逆向运动学，返回 `(success, q_solution)`
- `has_collision(q)`: 碰撞检测，返回 `bool`
- `sample_q(halton_vals)`: 将 Halton 序列映射到关节空间
- `n_halton_dims`: 需要的 Halton 序列维度
- `supports_ik`: 是否支持 IK
- `get_config_dict()`: 返回可序列化的配置
- `from_config_dict(config)`: 从配置重建实例

### 多进程架构

- 主进程负责进度监控和数据保存
- Worker 进程独立加载机器人模型并执行 FK/IK 计算
- 使用 `mp.RawArray` 实现无锁共享内存，避免死锁
- 支持 Ctrl+C 优雅中断和断点续传

## 特性

- ✅ 统一入口，只需修改机器人名称变量
- ✅ 完全通用的 worker 函数，无机器人特定逻辑
- ✅ 支持断点续传（FK）
- ✅ Rich 进度条或 tqdm 进度条可选
- ✅ 自动碰撞检测（Pinocchio 机器人）
- ✅ 多进程并行加速
- ✅ 类型安全的抽象接口

## 输出文件

FK 构建输出：`./data/{robot_name}_fk_cmap_mp.npz`  
IK 构建输出：`./data/{robot_name}_ik_cmap_mp.npz`

文件包含：
- `cmap`: 5D 布尔数组 `(n_c, n_c, n_c, n_p, m_o)`
- `config`: 离散化器配置字典
- `step`: 当前步数（仅 FK，用于断点续传）

## 注意事项

1. CSM 机器人不支持 IK，尝试运行 IK 构建会报错
2. Pinocchio 机器人需要 `example-robot-data` 包支持
3. 多进程数量默认为 `cpu_count() - 2`，可在代码中调整
4. FK 构建支持断点续传，IK 构建不支持（每次重新开始）

## 迁移指南

旧脚本已删除，迁移方式：

| 旧脚本 | 新脚本 | 修改方式 |
|--------|--------|----------|
| `cmap_build_fk_mp.py` | `build_fk_cmap_mp.py` | 设置 `ROBOT_NAME = "ur5"` |
| `cmap_build_panda_fk_mp.py` | `build_fk_cmap_mp.py` | 设置 `ROBOT_NAME = "panda"` |
| `cmap_build_csm_fk_mp.py` | `build_fk_cmap_mp.py` | 设置 `ROBOT_NAME = "csm"` |
| `cmap_build_ik_mp.py` | `build_ik_cmap_mp.py` | 设置 `ROBOT_NAME = "ur5"` |

## 许可证

与项目主仓库保持一致。
