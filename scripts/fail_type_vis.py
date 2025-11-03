import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from csm import CSM
from csm_display import normalize_vector, calculate_angular_velocity

# ==== 参数范围 ====
LIMITS = {
    "theta_1": (0, np.pi/2),
    "theta_2": (0, 2*np.pi/3),
    "L2": (0.0, 0.06),
    "Ls": (0.0, 0.15),
}

# ==== 加载失败数据 ====
with open("./data/failures_play_2.json", "r") as f:
    failures = json.load(f)

# ==== 雷达图绘制函数（改进版，参考博客） ====
def draw_radar(ax, failure, limits, title="Parameter Radar"):
    labels = list(limits.keys())
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # 计算每个参数相对范围的比例值
    vals = []
    for label in labels:
        val = failure[label]
        low, high = limits[label]
        ratio = (val - low) / (high - low)
        vals.append(np.clip(ratio, 0, 1))
    vals += vals[:1]

    # 设置雷达图方向与标签样式
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='grey', size=10)
    ax.set_rlabel_position(0)
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ylabels = [f"{int(t*100)}%" for t in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color="grey", size=8)
    ax.set_ylim(0, 1)

    # 绘制数据线与填充
    ax.plot(angles, vals, linewidth=2, linestyle='solid', color="b")
    ax.fill(angles, vals, alpha=0.25, color="b")
    ax.set_title(title, size=12, y=1.12)

# ==== 绘制单个失败样例 ====
def show_failure(failure):
    # 初始化 CSM
    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, 0.001)
    csm.set_state(
        failure["mode"], failure["phi"], failure["L1"], failure["L2"],
        failure["Lr"], failure["Ls"], failure["theta_1"],
        failure["theta_2"], failure["delta_1"], failure["delta_2"]
    )
    csm.target_pose = failure["target_pose"]

    # ==== 创建双子图 ====
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    axradar = fig.add_subplot(122, polar=True)
    plt.subplots_adjust(wspace=0.35)

    # --- 绘制机械臂 ---
    csm.plot_manipulator(ax3d)

    label_config = failure.get("label_config", None)
    if label_config:
        print("Drawing label configuration for comparison.")
        csm_ref = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, 0.001)
        # 注意：label_config 来自 workspace_data["config"]，字段一致
        csm_ref.set_state(
            failure["true_mode"], label_config["phi"], label_config["L1"], label_config["L2"],
            label_config["Lr"], label_config["Ls"], label_config["theta_1"],
            label_config["theta_2"], label_config["delta_1"], label_config["delta_2"]
        )
        csm_ref.target_pose = failure["target_pose"]
        csm_ref.plot_manipulator(ax3d, reverse_color=True)

    ax3d.set_title(f"Manipulator Configuration Comparison | Failure ID {failure['id']}")
    ax3d.legend(["Failure config", "Label config"], loc="upper right")
    
    # --- 绘制雷达图 ---
    draw_radar(axradar, failure, LIMITS, title="Parameter Radar (Relative Range)")
    plt.show()

# ==== 主循环：人工浏览 ====
idx = 0
while True:
    show_failure(failures[idx])
    cmd = input(f"当前样本 ID={failures[idx]['id']}，按回车查看下一个，或输入 q 退出：")
    if cmd.lower() == 'q':
        break
    idx = (idx + 1) % len(failures)
