import numpy as np
from LineGenerator import LineGenerator
from mpl_toolkits.mplot3d import Axes3D

def skew_symmetric_matrix(p):
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

def damped_pseudo_inverse(J, damping_factor = 0.01):
    """
    计算阻尼最小二乘法的伪逆

    参数:
    J: 输入矩阵
    damping_factor: 阻尼因子

    返回:
    J_damped_pinv: 阻尼最小二乘法伪逆
    """
    m, n = J.shape
    if m >= n:
        J_damped_pinv = np.linalg.inv(J.T @ J + (damping_factor**2) * np.eye(n)) @ J.T
    else:
        J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + (damping_factor**2) * np.eye(m))
    return J_damped_pinv

class CSM:
    def __init__(self, L_10, L_20, L_r0, L_s0):
        self.mode = 1
        self.L_10 = L_10
        self.L_20 = L_20
        self.L_r0 = L_r0
        self.L_s0 = L_s0
        self.phi = 0
        self.theta_1 = 0
        self.theta_2 = 0
        self.delta_1 = 0
        self.delta_2 = 0

        self.J_1v2 = None
        self.J_1w2 = None
        self.J_1v3 = None
        self.J_1w3 = None
        self.J_2v2 = None
        self.J_2w2 = None
        self.J_2v3 = None
        self.J_2w3 = None
        # 四种配置下的雅可比矩阵
        self.J1_v = None
        self.J1_w = None
        self.J2_v = None
        self.J2_w = None
        self.J3_v = None
        self.J3_w = None
        self.J4_v = None
        self.J4_w = None
        # 旋转、平移矩阵
        self.w_P_1b_2e = None
        self.w_R_1b = None
        self.w_R_2b = None
        self.b1_P_1e_2e = None
        # 末端执行器位姿
        self.pose = np.array([0, 0, 0, 0, 0, 0, 1])
        self.target_pose = np.array([0, 0, 0, 0, 0, 0, 1])
        self.vel = np.zeros(6)



    def get_jacobians(theta, L, delta):
        if theta == 0:
            J_v3 = np.array([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]
            ])
        else:
            h = (1 - np.cos(theta)) / theta
            J_v3 = np.array([
                [np.cos(delta) * (L / theta) * (np.sin(theta) - h), np.cos(delta) * h, -L * np.sin(delta) * h],
                [-np.sin(delta) * (L / theta) * (np.sin(theta) - h), -np.sin(delta) * h, -L * np.cos(delta) * h],
                [(L / theta) * (np.cos(theta) - np.sin(theta) / theta), np.sin(theta) / theta, 0]
            ])

        J_w3 = np.array([
            [np.sin(delta), 0, np.cos(delta) * np.sin(theta)],
            [np.cos(delta), 0, -np.sin(delta) * np.sin(theta)],
            [0, 0, np.cos(theta) - 1]
        ])
        J_v2 = J_v3[:, [0, 2]]
        J_w2 = J_w3[:, [0, 2]]
        return J_v3, J_w3, J_v2, J_w2

    def get_jacobian_1(self):
        z_w = np.array([0, 0, 1])  # Assuming z_w is along the z-axis
        self.J1_v = np.block(-skew_symmetric_matrix(self.w_P_1b_2e) @ z_w, self.w_R_2b@self.J_2v3)
        self.J1_ω = np.block(z_w, self.w_R_2b@self.J_2w3)
        return

    def get_jacobian_2(self):
        z_w = np.array([0, 0, 1])
        self.J2_v = np.block(-skew_symmetric_matrix(self.w_P_1b_2e)@z_w, z_w, self.w_R_2b@self.J_2v2)
        self.J2_ω = np.block(z_w, np.zeros((3, 1)), self.w_R_2b@self.J_2w2)
        return
    
    def get_jacobian_3(self):
        z_w = np.array([0, 0, 1])
        W2 = -skew_symmetric_matrix(self.b1_P_1e_2e)@self.J_1w3 + self.J_1v3
        self.J3_v = np.block(-skew_symmetric_matrix(self.w_P_1b_2e)@z_w, self.w_R_1b@W2, self.w_R_2b@self.J_2v2)
        self.J3_w = np.block(z_w, self.w_R_1b@self.J_1w3, self.w_R_2b@self.J_2w2)
        return

    def get_jacobian_4(self):
        z_w = np.array([0, 0, 1])
        W1 = -skew_symmetric_matrix(self.b1_P_1e_2e)@self.J_1w2 + self.J_1v2
        self.J4_v = np.block(-skew_symmetric_matrix(self.w_P_1b_2e)@z_w, z_w, self.w_R_1b@W1, self.w_R_2b@self.J_2v2)
        self.J4_w = np.block(z_w, np.zeros((3, 1)), self.w_R_1b@self.J_1w2, self.w_R_2b@self.J_2w2)
        return

    def get_dot_PHI(self, v, w):
        J1_v_p = damped_pseudo_inverse(self.J1_v)
        J2_v_p = damped_pseudo_inverse(self.J2_v)
        tem = (np.eye(3) - J1_v_p @ self.J1_v)
        tem2 = (np.eye(3) - J2_v_p @ self.J2_v)
        self.d_PHI_1 = J1_v_p@v + tem@damped_pseudo_inverse(self.J1_w@tem)@(w - self.J1_w@J1_v_p@v)
        self.d_PHI_2 = J2_v_p@v + tem2@damped_pseudo_inverse(self.J2_w@tem2)@(w - self.J2_w@J2_v_p@v)

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

    def check_transition(self):
        if self.mode == 1 and self.L_20 > self.L_r0:
            self.state_transition(1, 2)
        elif self.mode == 2 and self.L_r0 < 0:
            self.state_transition(2, 1)
        elif self.mode == 2 and self.L_r0 > self.L_r0:
            self.state_transition(2, 3)
        elif self.mode == 3 and self.L_10 < 0:
            self.state_transition(3, 2)
        elif self.mode == 3 and self.L_10 > self.L_10:
            self.state_transition(3, 4)
        elif self.mode == 4 and self.L_s0 < 0:
            self.state_transition(4, 3)

    def state_transition(self, current_mode, new_mode):
        if current_mode == 1 and new_mode == 2:
            self.L_r0 = self.L_20 - self.L_20
            self.mode = 2
        elif current_mode == 2 and new_mode == 1:
            self.L_20 = self.L_r0 + self.L_20
            self.mode = 1
        elif current_mode == 2 and new_mode == 3:
            self.L_10 = self.L_r0 - self.L_r0
            self.theta_1 = 0
            self.delta_1 = 0
            self.mode = 3
        elif current_mode == 3 and new_mode == 2:
            self.L_r0 = self.L_10 + self.L_r0
            self.mode = 2
        elif current_mode == 3 and new_mode == 4:
            self.L_s0 = self.L_10 - self.L_10
            self.mode = 4
        elif current_mode == 4 and new_mode == 3:
            self.L_10 = self.L_s0 + self.L_10
            self.mode = 3
        else:
            raise ValueError("Invalid mode transition")

        self.update_jacobians()

    def update_jacobians(self):
        if self.mode == 1:
            self.get_jacobian_1()
        elif self.mode == 2:
            self.get_jacobian_2()
        elif self.mode == 3:
            self.get_jacobian_3()
        elif self.mode == 4:
            self.get_jacobian_4()
    
    def step(self):
        """
        根据输出PHI更新状态
        模拟现实世界运动情况
        """
        if self.mode == 1:
            self.L_10 += self.d_PHI_1
            self.L_20 += self.d_PHI_2


if __name__ == "__main__":
    csm = CSM(0.5, 0.5, 0.15, 0.15)
    csm.target_pose = np.array([0.5, 0.5, 0.5, 0, 0, 1, 0])
    finshed = False
    try:
        while not finshed:
            err_pos = csm.target_pose[:3] - csm.pose[:3]
            err_ori = csm.target_pose[3:] - csm.pose[3:]
            v = 1 * err_pos/np.linalg.norm(err_pos)
            w = 1 * err_ori/np.linalg.norm(err_ori)
            csm.get_dot_PHI(v, w)
            
            csm.step()

            csm.check_transition()
            csm.get_dot_PHI(csm.vel[:3], csm.vel[3:])
            csm.pose += csm.vel

            if np.linalg.norm(csm.pose - csm.target_pose) < 1e-3:
                finshed = True
        print("Finished")
    except KeyboardInterrupt:
        print("Interrupted")
