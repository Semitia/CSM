import numpy as np
from csm import CSM
import matplotlib.pyplot as plt

def generate_workspace(csm, mode, num_samples=10000):
    csm.mode = mode
    workspace_points = []
    for _ in range(num_samples):
        # 随机生成姿态参数
        csm.phi = np.random.uniform(0, 2 * np.pi)
        if mode == 1:
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L_20)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
            csm.L2 = np.random.uniform(0, csm.L_20)
        elif mode == 2:
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L_20)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
            csm.Lr = np.random.uniform(0, csm.L_r0)
        elif mode == 3:
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L_10)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.L1 = np.random.uniform(0, csm.L_10)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L_20)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)
        elif mode == 4:
            csm.Ls = np.random.uniform(0, csm.L_s0)
            csm.theta_1 = np.random.uniform(0, csm.kappa_10 * csm.L_10)
            csm.delta_1 = np.random.uniform(0, 2 * np.pi)
            csm.theta_2 = np.random.uniform(0, csm.kappa_20 * csm.L_20)
            csm.delta_2 = np.random.uniform(0, 2 * np.pi)

        csm.update()
        workspace_points.append(csm.pose[:3])
    
    return np.array(workspace_points)

def plot_workspace(ax, workspace, label, color):
    ax.scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], s=2, label=label, color=color)

if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    csm = CSM(0.5, 0.5, 0.15, 0.15, 0.01)
    
    # Generate and plot workspace for each configuration
    workspace_C1 = generate_workspace(csm, mode=1)
    plot_workspace(ax, workspace_C1, label='C1', color='r')
    
    # workspace_C2 = generate_workspace(csm, mode=2)
    # plot_workspace(ax, workspace_C2, label='C2', color='g')
    
    # workspace_C3 = generate_workspace(csm, mode=3)
    # plot_workspace(ax, workspace_C3, label='C3', color='b')
    
    # workspace_C4 = generate_workspace(csm, mode=4)
    # plot_workspace(ax, workspace_C4, label='C4', color='y')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.5])
    ax.legend()
    plt.show()
