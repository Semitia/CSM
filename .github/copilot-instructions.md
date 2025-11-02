<!-- .github/copilot-instructions.md - instructions for AI coding agents -->
# 快速上下文（供 AI 代码助手使用）

此仓库实现了一个基于几何/雅可比的可变构型操纵器（CSM）仿真与可视化工具。核心语言为 Python，主要依赖：numpy、matplotlib（项目中未包含 requirements.txt）。

关键点速览
- 主模块目录：`scripts/`
- 核心计算：`scripts/csm.py`（CSM 类：state、几何变换、雅可比构造、运动学求解）
- 试验/数据：`scripts/csm_experiment.py`（生成/播放 workspace 数据、输出 `workspace_data.json`、`successes_play.json`、`failures_play.json`）
- 绘图/可视化：`scripts/Visualizer.py`（依赖 `LineGenerator.py` 提供 arc/line 段生成）
- 数据目录：`data/`（已有若干 play/replay 的 json），工作空间样本存放在 `workspace_data.json`。

项目约定与可操作规则（务必遵守）
- 代码路径与运行：Python 脚本直接作为入口运行（例如 `python .\scripts\csm_experiment.py`），不要假设存在打包或 CLI 框架。
- 随机/生成函数：`scripts/csm_experiment.py` 与 `scripts/workspace.py` 用随机采样生成工作空间；修改参数时请保留现有 JSON 输出格式以保证兼容（见下）。
- 数据格式示例：`workspace_data.json` 中每项为 {"mode": <1|2|3|4>, "pose": [x,y,z,ox,oy,oz]}，pose 是 6 元向量（位置 + 方向向量），agent 写入/读取要兼容该结构。
- 模式（mode）语义：CSM 有四个配置模式（1..4），改变姿态组合和雅可比拼接逻辑；在 `scripts/csm.py` 中多处以 `if self.mode == N` 分支处理运动学/更新/绘图。

常见代码位置与 API 快参
- 构造与仿真
  - CSM 类：`scripts/csm.py`
    - 关键方法：`CSM.update()`（刷新位姿/变换）、`CSM.update_jacobians()`（重建雅可比）、`CSM.get_dot_PHI(v,w)`（由期望线速度/角速度求关节增量）、`CSM.reset()`、`CSM.set_state(...)`。
  - 约定：位置向量和方向向量均以 numpy array 表示，pose 常以 6 元数组使用（前三为位置，后三为方向向量）。

- 试验与 dataset
  - 生成：`scripts/workspace.py`（生成并保存 `workspace_data.json`）；`scripts/csm_experiment.py` 会读取 `workspace_data.json` 并在运行中产生 `successes_play.json` 与 `failures_play.json`。
  - 读取/播放：`scripts/csm_experiment.py` 的主循环依赖 `CSM.check_transition()`、`update()`、`update_jacobians()`、`get_dot_PHI()`、`step()`（注意：如果添加或修改 step/transition 行为，请同时检查播放脚本的期望字段）。

- 可视化
  - `scripts/Visualizer.py` 使用 `LineGenerator.add_arc` / `add_line` 来绘制不同 mode 的段组合；若改变坐标系或 pose 布局，请同步更新 Visualizer 中对 csm.*_pos / *_ori 字段的引用。

对 AI 的具体指令（要点、示例与注意事项）
1. 阅读/修改任何运动学或雅可比相关代码前，先打开 `scripts/csm.py` 的以下区域：初始化字段（L_10 等）、`get_jacobians()`、`get_trans_mat()`、`get_jacobian_1..4()`、`get_dot_PHI()`、`update()`。这些决定了数据流与 API。
2. 新增或调整实验时：保持 `workspace_data.json` 条目格式兼容（见上），避免修改字段名（mode / pose / steps_taken / target_pose 等）。
3. 测试运行命令示例（Windows PowerShell）：

```
python .\scripts\csm_experiment.py
python .\scripts\workspace.py   # 生成 workspace_data.json
python .\scripts\csm_experiment.py  # 运行 playback / training loop
```

4. 视觉/调试：快速绘图可以运行 `Visualizer` 中的示例或在调试会话中创建 `Visualizer().plot(csm, ax)`，注意 matplotlib 需要交互或保存图像。
5. 风险点（实现细节）：
  - 仿真中常有零除（theta==0 或 delta_t==0）的防护逻辑，请勿移除除错检查。
  - 雅可比拼接时维度不同（部分拼接使用 3×N，其他使用 6×N），修改时注意 damped pseudo-inverse 的形状适配。

引用示例（在代码中查找）
- workspace 生成：`scripts/workspace.py` → `generate_workspace_data(csm, mode)`。
- 运行并记录结果：`scripts/csm_experiment.py` 主循环（读取 `workspace_data.json`，输出 `successes_play.json` / `failures_play.json`）。

如果有不明确的地方请告诉我：例如要把依赖收集到 `requirements.txt`、添加单元测试或把运行命令包装成 Makefile/PowerShell 脚本？我可以基于你的偏好继续完善这份说明文档。
