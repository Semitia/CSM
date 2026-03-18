# VS-IK 论文复现项目总结报告

## 1. 项目概述

本项目复现了论文 **"Inverse Kinematics and Dexterous Workspace Formulation for 2-Segment Continuum Robots With Inextensible Segments"** 的核心内容，实现了 Variable Separation Inverse Kinematics (VS-IK) 方法与 Dexterous Workspace 边界分析模块。

## 2. 实现的功能模块

### 2.1 VS-IK 核心求解器 (`src/csm/vsik.py`)

| 类/函数 | 功能描述 |
|---------|----------|
| `CSMParameters` | 机器人物理参数数据类 |
| `VSIKSolver` | VS-IK 求解器主类 |
| `solve_ci1()` | CI-1 配置逆运动学求解（公式 19） |
| `solve_ci2()` | CI-2 配置逆运动学求解（公式 13） |
| `solve()` | 统一求解入口，支持自动模式选择 |
| `compute_fk()` | 正运动学验证函数 |

**核心算法**:
- 将多变量 IK 问题转化为单变量非线性方程求解
- 使用牛顿迭代法求解方程 (13) 和 (19)
- 支持多初始点搜索以找到全局最优解

### 2.2 Dexterous Workspace 边界模块 (`src/csm/dex_workspace.py`)

| 类/函数 | 功能描述 |
|---------|----------|
| `DexterousWorkspace` | 灵巧工作空间边界计算类 |
| `compute_type1_boundaries_ci1()` | CI-1 Type-I 边界（配置极限） |
| `compute_type2_boundary_ci1()` | CI-1 Type-II 边界（奇异点） |
| `compute_type1_boundaries_ci2()` | CI-2 Type-I 边界 |
| `compute_type2_boundary_ci2()` | CI-2 Type-II 边界 |
| `project_to_closest_dexterous_direction()` | 不可达方向的最近可达投影 |
| `is_in_dexterous_workspace()` | 位姿可达性检查 |

## 3. 验证脚本

### Step 1: VS-IK 模块检验
```bash
python scripts/test_vsik_step1.py
```
- 模块导入检验 ✓
- 参数类兼容性检验 ✓
- 基本求解功能检验 ✓
- 工作空间采样检验 ✓

### Step 2: Dexterous Workspace 模块检验
```bash
python scripts/test_dex_workspace_step2.py
```
- 模块导入检验 ✓
- 参数兼容性检验 ✓
- 边界计算功能检验 ✓
- 工作空间检查功能检验 ✓
- 方向投影功能检验 ✓
- 可视化功能检验 ✓

### Step 3: 批量对比实验
```bash
python scripts/reproduce_vsik_comparison.py --num-cases 200 --mode ci1
```

## 4. 实验结果

### CI-1 配置对比实验 (200 cases)

| 指标 | VS-IK | Jacobian-DLS |
|------|-------|--------------|
| 成功率 | 38.0% (76/200) | 62.0% (124/200) |
| 平均求解时间 | 6.27 ms | 39.32 ms |
| 平均位置误差 | 13.47 mm | 0.74 mm |
| 平均迭代次数 | 1 (解析) | 80.4 |

**关键发现**:
- VS-IK 计算速度提升 **84.0%**
- 成功率低于 Jacobian-DLS，原因待分析与优化
- 位置误差较大，需要改进求解精度

## 5. 文件结构

```
CSM/
├── src/csm/
│   ├── __init__.py           # 更新：导出 VS-IK 与 DexterousWorkspace
│   ├── model.py               # 原有：CSM 运动学模型
│   ├── vsik.py                # 新增：VS-IK 求解器
│   └── dex_workspace.py       # 新增：灵巧工作空间模块
├── scripts/
│   ├── test_vsik_step1.py    # Step 1 验证脚本
│   ├── test_dex_workspace_step2.py  # Step 2 验证脚本
│   └── reproduce_vsik_comparison.py # Step 3 对比实验
├── config/
│   └── csm_cfg_3.4mm.yaml    # 3.4mm 机器人参数配置
├── data/
│   └── vsik_comparison_ci1.json  # 实验结果数据
└── imgs/
    └── dex_workspace_boundary_test.png  # 边界可视化图
```

## 6. 使用方法

### 6.1 基本使用
```python
from csm import VSIKSolver, CSMParameters

# 加载参数
params = CSMParameters.from_config({'robot': {...}})

# 创建求解器
solver = VSIKSolver(params)

# 求解逆运动学
p_g = np.array([0.0, 0.0, 0.04])
a = np.array([0.0, 0.0, -1.0])
result = solver.solve((p_g, a), mode='ci1')

if result:
    print(f"解: theta_1={result['theta_1']:.4f}, theta_2={result['theta_2']:.4f}")
```

### 6.2 工作空间边界检查
```python
from csm import DexterousWorkspace, CSMParameters

params = CSMParameters.from_config(...)
dex_ws = DexterousWorkspace(params)

# 检查可达性
p_g = np.array([0.0, 0.0, 0.03])
a = np.array([0.0, 0.0, -1.0])
is_reachable = dex_ws.is_in_dexterous_workspace(p_g, a)

# 投影到最近可达方向
a_nearest = dex_ws.project_to_closest_dexterous_direction(p_g, a_target)
```

## 7. 待改进方向

1. **VS-IK 求解精度优化**: 当前成功率与误差指标与论文有差距，需要：
   - 改进多根搜索策略
   - 增加配置变量边界约束检查
   - 优化 δ₁, δ₂, φ 的计算

2. **Dexterous Workspace 完整实现**:
   - 完善 Type-II 奇异点边界解析公式
   - 实现论文中的完整边界曲线族

3. **CI-2 配置支持**: 当前主要测试 CI-1，需要扩展 CI-2 测试

4. **批量实验扩展**: 
   - 增加测试案例数量（论文使用 500,000 cases）
   - 扩展到 CI-2 配置

## 8. 结论

本项目完成了论文核心方法的初步复现，包括：
- ✓ VS-IK 变量分离逆运动学求解器
- ✓ Dexterous Workspace 边界计算模块  
- ✓ 与 Jacobian-DLS 的对比实验框架

VS-IK 方法在计算效率上展现出显著优势（84% 速度提升），但求解成功率与精度需要进一步优化。这为后续深入研究与改进奠定了基础。

---
*报告生成时间: 2026-03-18*
*基于 CSM 项目代码库*
