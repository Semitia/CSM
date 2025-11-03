import json
import numpy as np
import matplotlib.pyplot as plt
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
with open("failures_play.json", "r") as f:
    failures = json.load(f)

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
    plt.subplots_adjust(wspace=0.3)

    # --- 绘制机械臂 ---
    csm.plot_manipulator(ax3d)
    ax3d.set_title(f"Manipulator Configuration | Failure ID {failure['id']}")
    ax3d.set_xlim(-0.1, 0.1)
    ax3d.set_ylim(-0.1, 0.1)
    ax3d.set_zlim(0, 0.25)

    # --- 绘制雷达图 ---
    labels = list(LIMITS.keys())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    vals = []

    for key in labels:
        v = failure[key]
        low, high = LIMITS[key]
        ratio = np.clip((v - low) / (high - low), 0, 1)
        vals.append(ratio)

    vals.append(vals[0])  # 闭合曲线
    axradar.plot(angles, vals, "b-", linewidth=2)
    axradar.fill(angles, vals, alpha=0.25, color="b")
    axradar.set_xticks(angles[:-1])
    axradar.set_xticklabels(labels, fontsize=10)
    axradar.set_yticklabels([])
    axradar.set_title("Parameter Radar (relative to allowed range)")

    plt.show()

# ==== 主循环：人工浏览 ====
idx = 0
while True:
    show_failure(failures[idx])
    cmd = input(f"当前样本 ID={failures[idx]['id']}，按回车查看下一个，或输入 q 退出：")
    if cmd.lower() == 'q':
        break
    idx = (idx + 1) % len(failures)
