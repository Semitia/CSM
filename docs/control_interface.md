# Interactive Control Interface

`csm.control_interface` 提供了一个基于 `matplotlib` 事件系统的交互式仿真窗口，用来做末端实时微分控制。

核心思路：

- 用户输入不会直接跳到一个远距离目标位姿。
- 键盘和鼠标每一帧只生成一个很小的末端位姿增量。
- 逆解依然复用现有的雅可比微分逆解 `get_dot_PHI()`。
- 每个显示帧内部会跑多个 `delta_t` 微步，因此求解总是从“当前状态附近”继续迭代，收敛会比每次重新做大步目标跟踪更平滑。

## 控制说明

- `W / S`: 沿世界坐标 `+Y / -Y` 平移
- `A / D`: 沿世界坐标 `+X / -X` 平移
- `Q / E`: 沿世界坐标 `+Z / -Z` 平移
- `J / L`: 控制末端目标朝向绕世界 `Z` 轴偏航
- `I / K`: 控制末端目标朝向俯仰
- 鼠标: 只用于旋转和缩放观察视角，不再控制末端朝向
- `R`: 机器人状态与交互目标一起复位

说明：

- 当前位置与目标位置的误差会实时显示在窗口左上角。
- 会显示求解状态：
  - `manual_input`: 目标正在被你手动移动
  - `converging`: 误差正在下降
  - `tracking`: 正在尝试跟踪
  - `stalled`: 长时间误差没有明显下降，可能是工作空间限制或局部逆解停滞
  - `likely_unreachable`: 目标位置已经明显超过机械臂总长度上界
  - `solved`: 已经进入设定容差
- 末端朝向仍然使用项目当前已有的 3 维方向向量表示，而不是完整 6 自由度姿态。
- 因此当前朝向控制只有两对按键是正常的：方向向量只有 2 个独立自由度。
- 如果后面要做第三对朝向控制，需要把末端姿态表示从“方向向量”升级为完整旋转矩阵、欧拉角或四元数，并让逆解同时跟踪工具轴自转。
- 平移控制使用世界坐标系，不随当前工具朝向变化。

## 直接运行

在项目根目录下执行：

```bash
python3 scripts/example_control_interface.py
```

默认会读取 [config/csm_cfg_3mm.yaml](/home/winslow/RII/CSM/config/csm_cfg_3mm.yaml)。

## 代码使用样例

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

如果你希望自己控制窗口对象，也可以直接实例化：

```python
from csm import CSM, ControlInterface

csm = CSM.from_config("config/csm_cfg_3mm.yaml")
interface = ControlInterface(csm, render_mode="detailed")
interface.start()
```

## 关键参数

- `frame_interval_ms`: 窗口刷新周期，默认 20 ms
- `linear_speed`: 用户输入对应的末端平移速度上限，单位 m/s
- `angular_speed`: 用户输入对应的末端角速度上限，单位 rad/s
- `position_gain`: 位置误差到末端线速度的比例增益
- `orientation_gain`: 朝向误差到末端角速度的比例增益
- `orientation_key_speed`: 键盘姿态控制角速度，单位 rad/s
- `max_substeps`: 每个显示帧内部最多执行多少个 `delta_t` 微步

## 实现细节

交互循环分成两层：

1. 外层是窗口刷新帧，用来采样键盘和鼠标输入并重绘。
2. 内层是求解微步循环，每步都执行：

```python
csm.check_transition()
csm.update()
csm.update_jacobians()
csm.get_dot_PHI(v, w)
csm.step()
```

随后会再次 `update()` 和 `update_jacobians()`，保证绘图看到的是最新状态。

因此它本质上还是原有雅可比逆解，只是把“给定静态目标点”改成了“持续给定小增量目标”。
