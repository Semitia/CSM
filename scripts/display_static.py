"""
Module: display_static.py
Description: Script to display random static configurations for all 4 modes of the CSM.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from csm.model import CSM

def set_random_state(csm, mode):
    """
    根据模式随机设置 CSM 状态，参考 gen_workspace.py 的逻辑
    """
    csm.state_transition(csm.mode, mode)
    
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

if __name__ == "__main__":
    # 使用正确的配置文件路径
    config_path = Path("./config/csm_config_3.4mm.yaml")
    
    print(f"Loading config from: {config_path}")

    # 创建画布，2x2 布局展示 4 种模式
    fig = plt.figure(figsize=(15, 12))
    
    # 实例化一个 CSM 对象，初始默认为 Mode 1
    # 通过循环依次切换到 2, 3, 4，避免直接从 1 跳到 3 的非法操作
    csm = CSM.from_config(config_path)
    
    modes = [1, 2, 3, 4]
    
    for i, mode in enumerate(modes):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # 复用同一个 csm 实例，逐步过渡状态
        # set_random_state 内部调用 csm.state_transition(csm.mode, mode)
        # 从而实现 1->1, 1->2, 2->3, 3->4 的合法转换序列
        set_random_state(csm, mode)
        csm.target_pose = csm.pose
        
        # 绘制
        csm.plot_manipulator(ax)
        ax.set_title(f"Mode {mode} (Random Config)")
        
        # 设置坐标轴比例一致，方便观察
        ax.set_box_aspect([1,1,1]) 
    
    plt.tight_layout()
    plt.show()