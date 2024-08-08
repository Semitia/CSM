import numpy as np
from LineGenerator import LineGenerator
from mpl_toolkits.mplot3d import Axes3D



def get_trans_mat(theta_t, L_t, delta_t):
    R_b_1 = np.array([  [0, np.cos(delta_t), np.sin(delta_t)],
                        [0, -np.sin(delta_t), np.cos(delta_t)],
                        [1, 0, 0]])

    R_1_2 = np.array(  [[np.cos(theta_t), -np.sin(theta_t), 0],
                        [np.sin(theta_t), np.cos(theta_t), 0],
                        [0, 0, 1]])

    R_2_e = np.array([  [0, 0, 1],
                        [np.cos(delta_t), -np.sin(delta_t), 0],
                        [np.sin(delta_t), np.cos(delta_t), 0]])

    R_te = R_b_1 @ R_1_2 @ R_2_e

    if theta_t == 0:
        P_te = np.array([0, 0, L_t])
    else:
        P_te = np.array([L_t * np.cos(delta_t) * (1 - np.cos(theta_t)) / theta_t,
                         L_t * np.sin(delta_t) * (np.cos(theta_t) - 1) / theta_t,
                         L_t * np.sin(theta_t) / theta_t])

    T = np.eye(4)
    T[:3, :3] = R_te
    T[:3, 3] = P_te
    return T


def plot_manipulator(mode, theta_1, L1, delta_1, theta_2, L2, delta_2, Lr, Ls):
    init_pos = np.array([0, 0, 0, 1])
    init_ori = np.array([0, 0, 1])
    lg = LineGenerator()

    if mode == 1:
        T = get_trans_mat(theta_2, L2, delta_2)
        end_pos = T @ init_pos
        end_ori = T[:3, :3] @ init_ori
        lg.add_arc(init_pos[:3], end_pos[:3], init_ori, end_ori)

    elif mode == 2:
        T1 = np.eye(4); T1[2, 3] = Lr
        T2 = get_trans_mat(theta_2, L2, delta_2)

        base2_pos = T1 @ init_pos
        base2_ori = T1[:3, :3] @ init_ori
        end2_pos = T1 @ T2 @ init_pos 
        end2_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori

        lg.add_line(init_pos[:3], base2_pos[:3])
        lg.add_arc(base2_pos[:3], end2_pos[:3], base2_ori, end2_ori)

    elif mode == 3:
        T1 = get_trans_mat(theta_1, L1, delta_1)
        T2 = np.eye(4); T2[2, 3] = Lr
        T3 = get_trans_mat(theta_2, L2, delta_2)

        end1_pos = T1 @ init_pos
        end1_ori = T1[:3, :3] @ init_ori
        base2_pos = T1 @ T2 @ init_pos
        base2_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
        end2_pos =T1 @ T2 @ T3 @ init_pos
        end2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ init_ori

        lg.add_arc(init_pos[:3], end1_pos[:3], init_ori, end1_ori)
        lg.add_line(end1_pos[:3], base2_pos[:3])
        lg.add_arc(base2_pos[:3], end2_pos[:3], base2_ori, end2_ori)
    
    elif mode == 4:
        T1 = np.eye(4); T1[2, 3] = Ls
        T2  =get_trans_mat(theta_1, L1, delta_1)
        T3 = np.eye(4); T3[2, 3] = Lr
        T4 = get_trans_mat(theta_2, L2, delta_2)

        base1_pos = T1 @ init_pos
        base1_ori = T1[:3, :3] @ init_ori
        end1_pos = T1 @ T2 @ init_pos
        end1_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
        base2_pos = T1 @ T2 @ T3 @ init_pos
        base2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ init_ori
        end2_pos = T1 @ T2 @ T3 @ T4 @ init_pos
        end2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ T4[:3, :3] @ init_ori

        lg.add_line(init_pos[:3], base1_pos[:3])
        lg.add_arc(base1_pos[:3], end1_pos[:3], base1_ori, end1_ori)
        lg.add_line(end1_pos[:3], base2_pos[:3])
        lg.add_arc(base2_pos[:3], end2_pos[:3], base2_ori, end2_ori)

    lg.plot_segments()


if __name__ == "__main__":
    # Example parameters
    L_10 = 0.5
    L_20 = 0.5
    L_r0 = 0.15
    L_s0 = 0.15
    phi = 0
    L1 = 0.5
    L2 = 0.5
    Lr = 0.15
    Ls = 0.1
    theta_1 = np.pi / 4
    delta_1 = -np.pi / 6
    theta_2 = np.pi / 3
    delta_2 = np.pi / 6
    mode = 4

    plot_manipulator(mode, theta_1, L1, delta_1, theta_2, L2, delta_2, Lr, Ls)