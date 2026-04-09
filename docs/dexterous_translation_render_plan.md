# Dexterous / Translation 论文风格出图方案

## 目标

围绕论文风格的工作空间效果图，补齐两类能力：

1. 继续沿用并整理现有 `workspace_boundary_scan`，负责 `translation workspace` 的轮廓构建与绘制。
2. 新增一个 `dexterous workspace` 出图模块，在若干指定评判点上绘制“单位半透明方向球”，球面着色表示该位置处可达的末端朝向。
3. 最后提供一个统一脚本，同时生成 translation shell、unreachable volume、机械臂姿态轨迹、dexterous spheres 等元素，尽量还原论文中的视觉效果。

## 当前仓库现状

- 现有可复用能力：
  - [`src/csm/workspace_boundary_scan/core.py`](/home/winslow/RII/CSM/src/csm/workspace_boundary_scan/core.py)
  - [`src/csm/workspace_boundary_scan/plotting.py`](/home/winslow/RII/CSM/src/csm/workspace_boundary_scan/plotting.py)
  - 已经能生成和论文比较接近的 `translation workspace` 外壳与内部不可达区域。
- 可参考的旧工作：
  - [`scripts/dexterous_analyse/`](/home/winslow/RII/CSM/scripts/dexterous_analyse)
  - 这套更偏 capability map / Monte Carlo / 离散化分析，适合做覆盖率统计，不适合直接复现论文里“指定点 + 单位球”的表达。
- 现有素材：
  - [`assets/translation_1.png`](/home/winslow/RII/CSM/assets/translation_1.png)
  - [`assets/translation_2.png`](/home/winslow/RII/CSM/assets/translation_2.png)
  - [`assets/dexterous_1.png`](/home/winslow/RII/CSM/assets/dexterous_1.png)
  - [`assets/dexterous_2.png`](/home/winslow/RII/CSM/assets/dexterous_2.png)
  - 这些图已经足够作为 translation / dexterous 两类目标效果的直接参考。

## 对论文视觉的理解

基于 [`assets/translation_1.png`](/home/winslow/RII/CSM/assets/translation_1.png)、[`assets/translation_2.png`](/home/winslow/RII/CSM/assets/translation_2.png)、[`assets/dexterous_1.png`](/home/winslow/RII/CSM/assets/dexterous_1.png)、[`assets/dexterous_2.png`](/home/winslow/RII/CSM/assets/dexterous_2.png) 与现有笔记，建议按下面理解实现：

- `translation workspace` 是一个大尺度的半透明三维壳体。
- `unreachable volume` 是包裹在其中、颜色更暖的内部不可达体。
- 机械臂本体和中心曲线叠加在最上层。
- `dexterous workspace` 不是再画一个全局体，而是在若干位置点放置局部球形 glyph。
- 这些球形 glyph 不是画满整个球，而是只绘制若干球冠/球面片区。
- 从图上看，主要编码的是“工具轴方向相对于某参考方向的可达锥域/可达球面区域”，仍然更接近 `S^2` 上的方向集合，而不是完整 `SO(3)`。
- [`assets/dexterous_2.png`](/home/winslow/RII/CSM/assets/dexterous_2.png) 说明还需要支持“同一点、不同参考指向”下的局部 dexterity 对比；也就是 probe 不只是一个位置，还可能绑定一个参考朝向或“inward / outward”语义。

这是目前最符合论文观感、也最容易和你现有 translation 出图能力拼接的解释。

## 从新增素材提炼出的更具体结论

### translation 图

从 [`assets/translation_1.png`](/home/winslow/RII/CSM/assets/translation_1.png) 和 [`assets/translation_2.png`](/home/winslow/RII/CSM/assets/translation_2.png) 看，translation 图至少包含四个明确图层：

1. workspace 外壳
2. functional volume / transverse plane
3. unreachable volume
4. 机械臂本体与中心线

其中：

- workspace 外壳是大尺度、浅绿、半透明、轮廓平滑的封闭体。
- unreachable volume 是内部深蓝色体，同样偏平滑，不是散点。
- functional volume 或 transverse plane 是一个带透明度的方盒/平面，需要单独的几何 primitive，而不是从 workspace 体自动推出来。

这意味着最终统一脚本不只是“画 workspace + dexterous 球”，还要支持显式叠加：

- 方盒
- 横截平面
- 文字标注箭头

### dexterous 图

从 [`assets/dexterous_1.png`](/home/winslow/RII/CSM/assets/dexterous_1.png) 和 [`assets/dexterous_2.png`](/home/winslow/RII/CSM/assets/dexterous_2.png) 看，dexterous 图的核心不是“给球面每个可达方向着一个连续 colormap”，而是：

- 在评判点放置一个半透明绿色球
- 再叠加一个或多个洋红色球冠/球面帽
- 通过球冠朝向与面积，表达该点评估方向下的定向能力

也就是说，最贴近原图的实现不是“随机三角片拼出的稀疏可达 patch”，而是：

- 先求出一个参考方向
- 再求出围绕该参考方向的可达半角或边界曲线
- 最终绘制成规则、平滑的球冠

这比“任意 reachable directions 的离散点集”更结构化，也更像论文图。

## 为什么不建议直接复用 Monte Carlo capability map

旧的 `scripts/dexterous_analyse` 主要适合：

- 体素化空间统计
- 朝向覆盖率分析
- 构建离散 capability map

但它和目标图之间有三个错位：

1. 论文视觉更像“给定位置后的方向边界/方向区域”，不是全局体素热图。
2. Monte Carlo 会天然带来颗粒感，不利于做平滑、半透明、论文式的球面着色。
3. 你现在更需要的是“在一个点上判断哪些方向可达”，而不是“全局有哪些 pose 出现过”。

所以建议保留 `dexterous_analyse` 作为数值参考或验证工具，而不是主出图路径。

## 推荐总体架构

建议新增三层：

### 1. `translation` 数据层

继续使用 `csm.workspace_boundary_scan`：

- 输入：`CSM` 模型、模式列表、扫描参数
- 输出：每个 mode 的 `WorkspaceProfile`
- 作用：提供外轮廓、不可达体轮廓、侧视图曲线、三维旋转壳体

这一层已经基本具备，不建议大改。

### 2. `dexterous` 数据层

新增一个位置条件下的方向可达性求解模块，例如：

- `src/csm/dexterous_workspace/core.py`
- `src/csm/dexterous_workspace/sampling.py`
- `src/csm/dexterous_workspace/plotting.py`

建议核心数据结构：

```python
@dataclass
class DexterousProbe:
    position_xyz: np.ndarray
    sphere_radius_mm: float
    directions: np.ndarray          # (N, 3)
    reachable_mask: np.ndarray      # (N,)
    score: np.ndarray | None = None # 可选，记录 margin / cost / manipulability
```

### 3. 统一组合脚本

例如：

- `scripts/render_workspace_figure.py`

职责：

- 调 `workspace_boundary_scan` 生成 translation 壳体
- 调 `dexterous_workspace` 生成若干 probe 球
- 叠加机械臂姿态、中心线、注释
- 输出主图和 side view

## Dexterous 部分的推荐求解路径

### 方案定位

为最大程度还原论文视觉，推荐主路径采用：

“指定位置点 + 参考方向/参考指向 + 方向边界求解 + 球冠/球面帽绘制”

而不是：

- 全局 Monte Carlo 散点
- 先生成大量 pose 再反推某点的球面分布
- 用噪声感很强的离散 patch 直接糊球面

### 输入

对每个 probe 点，需要给定：

- 位置 `p`
- 工作模式 / 构型模式
- 方向采样分辨率
- 单位球半径（仅绘图）

probe 点来源建议支持三种方式：

1. 手动指定
2. 沿参考轨迹/中心线自动取样
3. 从 translation workspace 内部按规则选择代表点

第一阶段建议优先支持手动指定，因为最有利于复现论文构图。

### 方向空间采样

建议使用 Fibonacci sphere / Saff spiral 采样 `S^2`，而不是经纬网：

- 球面分布更均匀
- 渲染不会在极点堆积
- 和旧 `WsDiscretizer` 的思路一致，可复用经验

推荐参数：

- 快速预览：`N = 300 ~ 800`
- 正式出图：`N = 1500 ~ 4000`

### 单方向可达性判断

给定位置 `p` 与候选方向 `a`，判断是否存在状态使得：

- 末端位置达到 `p`
- 工具轴方向达到 `a` 或足够接近 `a`
- 满足模式与构型约束

推荐判断顺序：

1. 先用 translation workspace 做快速过滤
2. 再做定向 IK / 定向边界判断
3. 成功则将该方向标为 reachable

如果后续实现中发现论文里的局部区域确实可以被近似成“参考轴 + 最大半角”的球冠，那么更推荐把求解结果压缩成：

- `center_dir`
- `angular_radius`
- `boundary_dirs`（可选，用于非轴对称修正）

这样绘图层就能直接生成平滑球冠，而不是依赖稠密离散点。

这里建议拆成两种实现路线。

## 两条实现路线

### 路线 A：解析/半解析优先

如果仓库里已有或准备补齐论文中的 dexterous workspace 解析边界公式，那么最优路径是：

- 对给定 `p`，直接计算该点的 dexterous boundary
- 得到边界曲线后，再在球面上构造 inside / outside mask

优点：

- 结果最接近论文
- 边界更平滑
- 计算效率可能高于大量 IK

缺点：

- 目前仓库主线里还没有稳定的可复用实现
- 接口、数值稳定性、模式兼容性还需要整理

### 路线 B：数值扫描优先

如果短期目标是先把图做出来，建议先落地：

- 对每个 `p`
- 先给定一个参考方向 `a_ref`
- 在 `a_ref` 周围做球面角向扫描
- 逐个做“位置优先、方向次优”的 IK/可达性验证
- 得到边界角或 `reachable_mask`
- 若区域接近球冠，则拟合成平滑球冠参数；若明显非球冠，再退回三角网格 patch 渲染

优点：

- 最直接
- 接口清晰
- 便于先出图再逐步替换底层求解器

缺点：

- 比解析方法慢
- 需要额外做平滑与插值

## 推荐实施策略

建议按“先可用、再逼近论文”的顺序推进：

### Phase 1

先做路线 B，尽快拿到与论文视觉一致的图：

- translation 继续走现有 `workspace_boundary_scan`
- dexterous 采用“指定点 + 参考方向 + 局部角域扫描 + IK 判定”
- 绘图时优先输出平滑球冠
- 只有在球冠拟合失败时，才退回球面三角剖分 patch

### Phase 2

在图已经能稳定输出后，再尝试把 dexterous 求解器替换成解析/半解析版本：

- 如果已有 `DexterousWorkspace` 相关代码或公式实现，就对接进去
- 保持渲染接口不变，只替换“reachable_mask 的生成方式”

这样风险最低，接口也最清晰。

## Dexterous 球的视觉表达建议

### 基本原则

单位球是“局部朝向空间”的可视化对象，不是实体障碍物，所以视觉上应满足：

- 半透明
- 表面平滑
- 可达区域颜色明确
- 不可达区域弱化或不画

### 推荐表现

对每个 probe 球：

- 在 `position_xyz` 放置一个半透明绿色基球
- 再叠加一个或多个洋红色球冠
- 若某个案例存在第二类能力区域，可继续叠加额外球面帽
- 不建议把 unreachable 区域画成密集补丁，原图里更像“只突出可用区域”

如果要兼顾通用性，建议渲染层同时支持两种 primitive：

1. `SphericalCapPrimitive`
2. `SphericalPatchPrimitive`

默认优先使用 `SphericalCapPrimitive`，因为它和论文图最像。

### 推荐颜色层次

- translation reachable shell：沿用当前淡绿色/淡蓝绿色
- unreachable volume：暖棕色半透明
- dexterous reachable patch：偏蓝或偏青，透明度略高于 shell
- 机械臂骨架：深色实线，置顶

### 遮挡与层级

建议图层顺序：

1. translation shell
2. unreachable volume
3. dexterous spheres
4. manipulator body / backbone / tip markers
5. annotation

如果 Matplotlib 的 3D 深度排序不稳定，可以保留当前“pseudo-3D / 论文风格压扁视角”的思路，不强求物理真实遮挡。

## 建议的代码组织

建议新增：

- `src/csm/dexterous_workspace/__init__.py`
- `src/csm/dexterous_workspace/core.py`
- `src/csm/dexterous_workspace/plotting.py`
- `scripts/render_workspace_figure.py`
- `docs/dexterous_translation_render_plan.md`（本文）

建议公开接口形态：

```python
from csm.dexterous_workspace import (
    DexterousProbeOptions,
    build_dexterous_probes,
    plot_dexterous_probes,
)
```

其中：

- `build_dexterous_probes(...)` 只负责算数据
- `plot_dexterous_probes(...)` 只负责球面渲染
- 最终总装由 `render_workspace_figure.py` 完成

## 统一脚本建议

建议统一脚本支持以下输入：

- 机器人配置文件路径
- mode 列表
- translation 扫描参数
- probe 点列表
- dexterous 方向采样数
- 输出图片路径
- 是否生成 side view

推荐输出：

1. 主图：三维论文风格图
2. side view：和论文右侧类似的侧视图
3. debug 图：
   - translation profile debug
   - 每个 probe 的单位球可达性 debug

## 关键难点与处理建议

### 1. 位置固定下的方向 IK

这是 dexterous 图的核心。建议先复用现有 IK/雅可比接口做一个“位置主任务，方向副任务”的判定器。

### 2. 球面边界的平滑性

数值扫描会有锯齿。建议：

- 先在球面方向上做邻域投票/形态学平滑
- 再做球面三角剖分渲染

### 3. 论文风格而非真实 3D 风格

目标不是 Blender 式真实渲染，而是论文插图风格。建议优先保证：

- 轮廓干净
- 半透明关系稳定
- 局部单位球易读

不要过早为真实光照和精确遮挡投入太多时间。

## 结论

最推荐的落地方案是：

1. 保持 `workspace_boundary_scan` 作为 translation 主实现。
2. 新增 `dexterous_workspace`，采用“指定位置点 + 单位球方向采样 + 位置条件下方向可达性判断”的方式生成 dexterous 球面着色。
3. 再用一个统一脚本把 translation shell、unreachable volume、机械臂和 dexterous spheres 组合成论文风格图。
4. 第一阶段优先用数值扫描实现稳定出图；第二阶段再视情况把 dexterous 判定替换为解析边界公式。

这条路径最符合你现在仓库的基础，也最有希望在不大改现有 translation 管线的前提下，把论文视觉效果还原出来。

## 当前确认后的实施决策

结合 `ref/CSM-DW` 现状，建议把上面的总方案进一步收束为“解析求解 + 论文风格重渲染”的混合路线：

1. `translation` 继续使用现有 `workspace_boundary_scan`。
2. `dexterous` 的求解层优先复用 [`ref/CSM-DW/src/continuum_robot_ik/workspace.py`](/home/winslow/RII/CSM/ref/CSM-DW/src/continuum_robot_ik/workspace.py) 中已经实现的解析/半解析边界与可行域接口。
3. `dexterous` 的渲染层不复用 `ref/CSM-DW` 前端当前的离散 patch 风格，而是在本仓库里单独实现论文风格球冠/平滑球面区域渲染。
4. 手动选点作为第一批 probe 点，先针对 3 mm 配置工作。

### 为什么这样定

- 你现在的真实缺口主要在“画风”，不是“数学公式完全没有”。
- `ref/CSM-DW` 里已经能从位置 `p_target` 直接计算局部 dexterous feasible area，不必强行先给参考朝向。
- 如果我们仍然完全走纯数值扫描，就会重复实现一遍已经有的边界求解，而且最终还得再做平滑。
- 更合理的做法是：
  - 位置输入只给 `p`
  - 求解层直接返回该位置下的可达方向区域
  - 渲染层把这个区域转成论文风格球冠或平滑 patch

### 对“不给参考朝向”的具体理解

现在统一改成下面这套流程：

1. 输入只有 probe 位置 `p`。
2. 先验证 `p` 是否在 translation workspace 内。
3. 如果需要一个代表姿态来摆放机械臂，可以额外做一次“只要求位置可达”的 bootstrap 求解：
   - 从多个初始状态出发
   - 位置优先 IK / VS-IK
   - 成功后保留一个可行 `q_seed`
4. dexterous 可达域本身不依赖这个 `q_seed` 作为先验，而是直接由该位置处的 dexterous workspace 公式或边界搜索给出。

这样 `q_seed` 只服务于“机械臂在图里怎么摆”，不参与定义 dexterous 球面区域。

### 3 mm 第一阶段建议

第一阶段建议只做：

- 配置：3 mm
- probe 点：手动指定
- mode：先只支持当前最关心的结构/模式
- dexterous 区域：优先用 `ref/CSM-DW` 的 feasible area / boundary 结果
- 渲染：不再默认拟合球冠，而是优先从 `2D symmetry plane` 的闭合区域重建平滑球面 patch

例如当前可先把你给的测试点作为第一批 probe：

- `p ≈ [10.87, 10.28, 29.77] mm`

而图片里显示的 orientation：

- `[0.384, 0.526, 0.759]`

现阶段可以只把它当作一个可视化参考姿态，不把它当成 dexterous 求解的输入前提。

### 需要保留的数值扫描后备路线

虽然主路线切到“解析求解 + 重渲染”，但数值扫描仍建议保留为后备：

- 当解析区域在某些位置出现异常、缺口、或和实际 IK 验证不一致时
- 可以在该局部位置开启“方向遍历 + 多初值 IK”模式做补充验证

也就是：

- 主路线：解析可行域
- 后备路线：数值扫描校验 / 补洞

这比单独押宝任意一条路线都更稳。

## 当前更新后的渲染决策

根据 [`data/dexterous_debug_patch/P1_debug.png`](/home/winslow/RII/CSM/data/dexterous_debug_patch/P1_debug.png) 的调试结果，当前结论更新为：

1. 问题主要不在解析可行域，而在渲染表达。
2. 当前局部 dexterous region 明显不是简单球冠，不能再默认用 `cap_center + angular_radius` 拟合。
3. 目前已经验证可行的过渡路径是：
   - 先在 `2D symmetry plane` 上得到闭合区域
   - 再将该区域 lift 到球面生成 3D patch
4. 但这条 mask/contour 管线仍然只是过渡方案，最终应升级为“利用已知边界族进行精确重建”的方案。

### 过渡主线：2D 区域 -> 3D 球面 patch

当前短期内继续保留的可用路线：

- 使用 `feasible_area_sym` 作为区域真值
- 将其光栅化到单位圆内
- 做轻量平滑与 contour 提取
- 在 2D 上做三角剖分
- 再 lift 到 3D 球面

这条路的作用是：

- 快速稳定出图
- 提供调试参考
- 帮助确认解析区域和最终 3D patch 是否一致

### 新的长期主线：边界族精确拟合 + 求交 + 闭区域重建

后续更推荐的正式路线是：

1. 不再以散点 mask 作为最终几何真值。
2. 转而把解析求解得到的边界族视为真值来源。
3. 针对每类边界做显式曲线建模，再求交、裁剪、闭合，最终重建精确的 2D 区域。
4. 将这个“精确 2D 区域” lift 到球面，得到光滑且在交点处保留尖锐夹角的 3D patch。

原因：

- 当前 `feasible_area_sym` 的轮廓虽然能提取，但边缘会受到采样、光栅化、平滑参数的影响。
- 而解析边界本身已经具有明确家族结构，几何上更接近论文里的“规则但带尖角”的边界。
- 如果能用边界族直接重建区域，最终结果会同时满足：
  - 更精确
  - 更平滑
  - 保留真实交角
  - 更接近论文风格

## 渲染路线的具体更新

### 1. 不再默认做单侧球冠拟合

球冠拟合保留，但只作为 debug 辅助：

- 用于判断区域是否近似球冠
- 不再作为默认输出 primitive

当前默认输出应是 patch，而不是 cap。

### 2. lift 到 3D 时需要补成双侧球面区域

当前实现只 lift 了 `a_sy >= 0` 的一侧，这是不完整的。

更新后的正确策略：

- 先在对称平面得到闭合区域 `Omega(a_sx, a_sz)`
- 对区域内每个点同时生成
  - `a_sy = +sqrt(1 - a_sx^2 - a_sz^2)`
  - `a_sy = -sqrt(1 - a_sx^2 - a_sz^2)`
- 再结合 `gamma` 旋转映射回世界坐标系

是否最终渲染双侧：

- 调试阶段：双侧都画，便于确认几何完整性
- 正式论文风图：根据论文图实际视觉需求决定只显示前半侧、后半侧，或双侧半透明叠加

但几何层必须先完整拥有双侧。

### 3. 边界重建应以“边界族”而不是“单条 contour”组织

后续要新增一个显式的边界重建层，例如：

- `type1` 边界家族
- `type2` 边界家族
- 单位圆约束边界

每个 family 都应以参数曲线或显式函数形式存在，而不是只剩散点。

目标输出不是“若干散点边界”，而是：

- 若干条有类型标识的连续曲线段
- 它们的交点
- 它们围成的闭合区域

### 4. 边界拟合建议的优先顺序

可以按下面顺序做：

1. 先识别每条边界段属于哪一类：
   - 圆弧
   - 直线
   - 由解析公式定义的曲线
2. 分别拟合对应参数
3. 求相邻边界段的交点
4. 按交点把边界裁剪成真正参与闭合的片段
5. 组合成闭合区域
6. 在闭合区域内部做规则化采样或直接做 polygon triangulation

其中：

- 单位圆边界天然就是圆弧
- 某些 Type-I 边界在局部上更接近直线族
- 其他边界则应优先直接使用其解析表达，而不是再做纯经验拟合

### 5. mask/contour 管线的地位调整

当前已经实现的 mask/contour 管线保留，但地位下调为：

- 调试工具
- 过渡方案
- 对解析边界重建结果的可视化对照

而不是最终的长期主线。

## 建议的开发顺序更新

后续实现按这个顺序推进：

1. 先补全双侧 lift，使 3D 几何完整。
2. 保留现有 mask/contour patch 路线用于快速预览。
3. 新增“边界族提取/拟合/求交/闭合区域重建”层。
4. 将最终渲染主线切换到“精确边界闭合区域”。
5. 把 mask 管线仅保留为 debug 和回退选项。

## 当前本地计划的最终判断

因此，当前本地方案更新为：

- 求解主线仍然是解析可行域
- 渲染短期主线是 `2D symmetry plane` 区域 lift 到 3D
- 渲染长期主线是“边界族精确拟合 + 求交 + 闭区域重建 + 双侧 lift”

这条路线比继续强化球冠拟合更合理，也更符合你要的“光滑、精确、交角保留”的论文效果。
