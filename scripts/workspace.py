import json
import numpy as np
from csm import CSM
import matplotlib.pyplot as plt

def generate_workspace(csm, mode, num_samples=2000):
    csm.state_transition(csm.mode, mode)
    workspace_points = []

    # DEBUG
    # 在 特定 mode 追踪“离 z 轴最远”的点与配置
    target_mode = 4
    max_r = -np.inf
    best_point = None
    best_cfg = None
    # END DEBUG

    for _ in range(num_samples):
        # 随机生成姿态参数
        csm.phi = np.random.uniform(0, 2 * np.pi)
        if mode == 1:
            csm.L2 = np.random.uniform(0, csm.L_20)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 2:
            csm.Lr = np.random.uniform(0, csm.L_r0)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 3:
            csm.L1 = np.random.uniform(0, csm.L_10)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 4:
            csm.Ls = np.random.uniform(0, csm.L_s0)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        csm.update()
        p = csm.pose[:3].copy()
        workspace_points.append(p)

    # DEBUG
        if mode == target_mode:
            r = float(np.hypot(p[0], p[1]))  # 等价于 sqrt(x^2 + y^2)
            if r > max_r:
                max_r = r
                best_point = p.copy()
                # 保存当前配置（按你在 mode 3 中会用到的量为主，其他也一并记录以便排查）
                best_cfg = {
                    "phi": float(csm.phi),
                    "L1": float(getattr(csm, "L1", np.nan)),
                    "L2": float(getattr(csm, "L2", np.nan)),
                    "Ls": float(getattr(csm, "Ls", np.nan)),
                    "Lr": float(getattr(csm, "Lr", np.nan)),
                    "theta_1": float(getattr(csm, "theta_1", np.nan)),
                    "delta_1": float(getattr(csm, "delta_1", np.nan)),
                    "theta_2": float(getattr(csm, "theta_2", np.nan)),
                    "delta_2": float(getattr(csm, "delta_2", np.nan)),
                }

    if mode == target_mode and best_point is not None:
        print("[mode {:d}] Point with max radius from z-axis:".format(target_mode))
        print("  max radius r = {:.6f}".format(max_r))
        print("  point (x, y, z) =", best_point)
        print("  config =", best_cfg)
    # END DEBUG

    return np.array(workspace_points)

def generate_workspace_data(csm, mode, num_samples=2500):
    csm.state_transition(csm.mode, mode)
    workspace_data = []
    for _ in range(num_samples):
        # 随机生成姿态参数
        csm.phi = np.random.uniform(0, 2 * np.pi)
        if mode == 1:
            csm.L2 = np.random.uniform(0, csm.L_20)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 2:
            csm.Lr = np.random.uniform(0, csm.L_r0)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 3:
            csm.L1 = np.random.uniform(0, csm.L_10)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 4:
            csm.Ls = np.random.uniform(0, csm.L_s0)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L1)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L2)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        csm.update()
        pose = csm.pose.tolist()
        workspace_data.append({"mode": mode, "pose": pose})
        
    return workspace_data

def plot_workspace(ax, workspace, label, color):
    ax.scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], s=2, label=label, color=color)

def save_workspace_to_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 20))
    csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi/2, 2*np.pi/3, 0.001)
    
    # # 四张图分别绘制四个配置的工作空间
    # ax1 = fig.add_subplot(221, projection='3d')
    # ax2 = fig.add_subplot(222, projection='3d')
    # ax3 = fig.add_subplot(223, projection='3d')
    # ax4 = fig.add_subplot(224, projection='3d')
    
    # # Generate and plot workspace for each configuration
    # workspace_C1 = generate_workspace(csm, mode=1)
    # plot_workspace(ax1, workspace_C1, label='C1', color='r')
    # ax1.set_title('Configuration 1')
    
    # workspace_C2 = generate_workspace(csm, mode=2)
    # plot_workspace(ax2, workspace_C2, label='C2', color='g')
    # ax2.set_title('Configuration 2')
    
    # workspace_C3 = generate_workspace(csm, mode=3)
    # plot_workspace(ax3, workspace_C3, label='C3', color='b')
    # ax3.set_title('Configuration 3')
    
    # workspace_C4 = generate_workspace(csm, mode=4)
    # plot_workspace(ax4, workspace_C4, label='C4', color='y')
    # ax4.set_title('Configuration 4')

    # # 统一设置每个子图的视角和比例
    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.set_xlim([-0.1, 0.1])
    #     ax.set_ylim([-0.1, 0.1])
    #     ax.set_zlim([0, 0.25])
    #     ax.view_init(elev=20, azim=30)  # 设置视角
    #     ax.set_box_aspect([1, 1, 1])  # 确保各坐标轴比例一致

    # fig.tight_layout()
    # plt.show()

    # # 一张图上绘制四个配置的工作空间
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Generate and plot workspace for each configuration
    # workspace_C1 = generate_workspace(csm, mode=1)
    # plot_workspace(ax, workspace_C1, label='C1', color='r')
    
    # workspace_C2 = generate_workspace(csm, mode=2)
    # plot_workspace(ax, workspace_C2, label='C2', color='g')
    
    # workspace_C3 = generate_workspace(csm, mode=3)
    # plot_workspace(ax, workspace_C3, label='C3', color='b')
    
    # workspace_C4 = generate_workspace(csm, mode=4)
    # plot_workspace(ax, workspace_C4, label='C4', color='y')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([-0.1, 0.1])
    # ax.set_ylim([-0.1, 0.1])
    # ax.set_zlim([0, 0.25])
    # ax.legend()
    # plt.show()

    # 生成并保存工作空间数据
    all_workspace_data = []
    num_samples_per_mode = [2500, 2500, 5000, 5000]  # 为每个模式指定样本数量
    for mode in range(1, 5):  # 生成模式1到模式4的数据
        workspace_data = generate_workspace_data(csm, mode, num_samples=num_samples_per_mode[mode-1])
        all_workspace_data.extend(workspace_data)
    
    save_workspace_to_file("./data/workspace_data.json", all_workspace_data)
