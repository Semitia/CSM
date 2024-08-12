import json
import numpy as np
from csm import CSM
import matplotlib.pyplot as plt

def generate_workspace(csm, mode, num_samples=10000):
    csm.state_transition(csm.mode, mode)
    workspace_points = []
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
        workspace_points.append(csm.pose[:3])

    return np.array(workspace_points)

def generate_workspace_data(csm, mode, num_samples=100):
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
    csm = CSM(0.5, 0.5, 0.15, 0.15, 0.01)
    
    # 四张图分别绘制四个配置的工作空间
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Generate and plot workspace for each configuration
    workspace_C1 = generate_workspace(csm, mode=1)
    plot_workspace(ax1, workspace_C1, label='C1', color='r')
    ax1.set_title('Configuration 1')
    
    workspace_C2 = generate_workspace(csm, mode=2)
    plot_workspace(ax2, workspace_C2, label='C2', color='g')
    ax2.set_title('Configuration 2')
    
    workspace_C3 = generate_workspace(csm, mode=3)
    plot_workspace(ax3, workspace_C3, label='C3', color='b')
    ax3.set_title('Configuration 3')
    
    workspace_C4 = generate_workspace(csm, mode=4)
    plot_workspace(ax4, workspace_C4, label='C4', color='y')
    ax4.set_title('Configuration 4')

    # 统一设置每个子图的视角和比例
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1.5])
        ax.view_init(elev=20, azim=30)  # 设置视角
        ax.set_box_aspect([1, 1, 1])  # 确保各坐标轴比例一致

    fig.tight_layout()
    plt.show()

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
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([0, 1.5])
    # ax.legend()
    # plt.show()

    # 生成并保存工作空间数据
    # all_workspace_data = []
    # for mode in range(1, 5):  # 生成模式1到模式4的数据
    #     workspace_data = generate_workspace_data(csm, mode)
    #     all_workspace_data.extend(workspace_data)
    
    # save_workspace_to_file("workspace_data.json", all_workspace_data)
