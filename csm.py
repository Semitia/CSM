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

    # print("J_damped_pinv (shape {}):\n{}".format(J_damped_pinv.shape, J_damped_pinv))
    return J_damped_pinv

class CSM:
    def __init__(self, L_10, L_20, L_r0, L_s0):
        self.mode = 1
        self.L_10 = L_10
        self.L_20 = L_20
        self.L_r0 = L_r0
        self.L_s0 = L_s0
        self.phi = 0
        self.L1 = 0
        self.L2 = 0.2
        self.Lr = 0
        self.Ls = 0
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
        self.w_P_2b_2e = None
        self.w_R_1b = None
        self.w_R_2b = None
        self.b1_P_1e_2e = None
        self.d_PHI = 0
        # 关键坐标位姿
        self.pose = np.array([0, 0, 0, 0, 0, 1])            # 末端执行器
        self.target_pose = np.array([0, 0, 0, 0, 0, 1])     # 目标位姿
        self.base1_pos = np.array([0, 0, 0, 1])
        self.base1_ori = np.array([0, 0, 1])
        self.end1_pos = np.array([0, 0, 0, 1])
        self.end1_ori = np.array([0, 0, 1])
        self.base2_pos = np.array([0, 0, 0, 1])
        self.base2_ori = np.array([0, 0, 1])
        self.end2_pos = np.array([0, 0, 0, 1])
        self.end2_ori = np.array([0, 0, 1])


    def get_jacobians(self, theta, L, delta):
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
        z_w = np.array([[0], [0], [1]])  # 假设 z_w 是沿着 z 轴的向量

        # 横向拼接J1_v
        term1_v = -skew_symmetric_matrix(self.w_P_2b_2e) @ z_w
        term2_v = self.w_R_2b @ self.J_2v3
        self.J1_v = np.hstack([term1_v, term2_v])
        # print("J1_v (shape {}):\n{}".format(self.J1_v.shape, self.J1_v))

        # 横向拼接J1_w
        term1_w = z_w
        term2_w = self.w_R_2b @ self.J_2w3
        self.J1_w = np.hstack([term1_w, term2_w])
        # print("J1_w (shape {}):\n{}".format(self.J1_w.shape, self.J1_w))
        return
    
    def get_jacobian_2(self):
        z_w = np.array([[0], [0], [1]])  # 假设 z_w 是沿着 z 轴的列向量

        term1_v = -skew_symmetric_matrix(self.w_P_2b_2e) @ z_w
        term2_v = z_w
        term3_v = self.w_R_2b @ self.J_2v2
        self.J2_v = np.hstack([term1_v, term2_v, term3_v])
        # print("J2_v (shape {}):\n{}".format(self.J2_v.shape, self.J2_v))

        term1_w = z_w
        term2_w = np.zeros((3, 1))
        term3_w = self.w_R_2b @ self.J_2w2
        self.J2_w = np.hstack([term1_w, term2_w, term3_w])
        # print("J2_w (shape {}):\n{}".format(self.J2_w.shape, self.J2_w))
        return

    def get_jacobian_3(self):
        z_w = np.array([[0], [0], [1]])  # 假设 z_w 是沿着 z 轴的列向量
        W2 = -skew_symmetric_matrix(self.b1_P_1e_2e) @ self.J_1w3 + self.J_1v3

        term1_v = -skew_symmetric_matrix(self.w_P_1b_2e) @ z_w
        term2_v = self.w_R_1b @ W2
        term3_v = self.w_R_2b @ self.J_2v2
        self.J3_v = np.hstack([term1_v, term2_v, term3_v])
        # print("J3_v (shape {}):\n{}".format(self.J3_v.shape, self.J3_v))

        term1_w = z_w
        term2_w = self.w_R_1b @ self.J_1w3
        term3_w = self.w_R_2b @ self.J_2w2
        self.J3_w = np.hstack([term1_w, term2_w, term3_w])
        # print("J3_w (shape {}):\n{}".format(self.J3_w.shape, self.J3_w))
        return
    
    def get_jacobian_4(self):
        z_w = np.array([[0], [0], [1]])  # 假设 z_w 是沿着 z 轴的列向量
        W1 = -skew_symmetric_matrix(self.b1_P_1e_2e) @ self.J_1w2 + self.J_1v2

        term1_v = -skew_symmetric_matrix(self.w_P_1b_2e) @ z_w
        term2_v = z_w
        term3_v = self.w_R_1b @ W1
        term4_v = self.w_R_2b @ self.J_2v2
        self.J4_v = np.hstack([term1_v, term2_v, term3_v, term4_v])
        # print("J4_v (shape {}):\n{}".format(self.J4_v.shape, self.J4_v))

        term1_w = z_w
        term2_w = np.zeros((3, 1))
        term3_w = self.w_R_1b @ self.J_1w2
        term4_w = self.w_R_2b @ self.J_2w2
        self.J4_w = np.hstack([term1_w, term2_w, term3_w, term4_w])
        # print("J4_w (shape {}):\n{}".format(self.J4_w.shape, self.J4_w))
        return

    def get_dot_PHI(self, v, w):
        if self.mode == 1:
            J1_v_p = damped_pseudo_inverse(self.J1_v)
            tem = (np.eye(4) - J1_v_p @ self.J1_v)
            self.d_PHI = J1_v_p@v + tem@damped_pseudo_inverse(self.J1_w@tem)@(w - self.J1_w@J1_v_p@v)
        elif self.mode == 2:
            J2_v_p = damped_pseudo_inverse(self.J2_v)
            tem = (np.eye(4) - J2_v_p @ self.J2_v)
            self.d_PHI = J2_v_p@v + tem@damped_pseudo_inverse(self.J2_w@tem)@(w - self.J2_w@J2_v_p@v)
        elif self.mode == 3:
            J3_v_p = damped_pseudo_inverse(self.J3_v)
            tem = (np.eye(6) - J3_v_p @ self.J3_v)
            self.d_PHI = J3_v_p@v + tem@damped_pseudo_inverse(self.J3_w@tem)@(w - self.J3_w@J3_v_p@v)
        elif self.mode == 4:
            J4_v_p = damped_pseudo_inverse(self.J4_v)
            tem = (np.eye(6) - J4_v_p @ self.J4_v)
            self.d_PHI = J4_v_p@v + tem@damped_pseudo_inverse(self.J4_w@tem)@(w - self.J4_w@J4_v_p@v)

    def get_trans_mat(self, theta_t, L_t, delta_t):
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

    def get_wR(self):
        """
        计算phi产生的旋转矩阵,从世界坐标系到stem
        1,2模式是2b, 3,4模式是1b
        """
        w = np.array([  [np.cos(self.phi), -np.sin(self.phi), 0],
                        [np.sin(self.phi), np.cos(self.phi), 0],
                        [0, 0, 1]])
        return w

    def plot_manipulator(self):
        init_pos = np.array([0, 0, 0, 1])
        init_ori = np.array([0, 0, 1])
        lg = LineGenerator()

        if self.mode == 1:
            T = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)
            self.end_pos = T @ init_pos
            self.end_ori = T[:3, :3] @ init_ori
            self.pose = np.block([self.end_pos[:3], self.end_ori])
            self.w_P_2b_2e = T[:3, 3]
            self.w_R_2b = self.get_wR()

            lg.add_arc(init_pos[:3], self.end_pos[:3], init_ori, self.end_ori)

        elif self.mode == 2:
            T1 = np.eye(4)
            T1[2, 3] = self.Lr
            T2 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)

            self.base2_pos = T1 @ init_pos
            self.base2_ori = T1[:3, :3] @ init_ori
            self.end2_pos = T1 @ T2 @ init_pos 
            self.end2_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
            self.pose = np.block([self.end2_pos[:3], self.end2_ori])
            self.w_P_2b_2e = T2[:3, 3]
            self.w_R_2b = self.get_wR()

            lg.add_line(init_pos[:3], self.base2_pos[:3])
            lg.add_arc(self.base2_pos[:3], self.end2_pos[:3], self.base2_ori, self.end2_ori)

        elif self.mode == 3:
            T1 = self.get_trans_mat(self.theta_1, self.L1, self.delta_1)
            T2 = np.eye(4)
            T2[2, 3] = self.Lr
            T3 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)

            self.end1_pos = T1 @ init_pos
            self.end1_ori = T1[:3, :3] @ init_ori
            self.base2_pos = T1 @ T2 @ init_pos
            self.base2_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
            self.end2_pos = T1 @ T2 @ T3 @ init_pos
            self.end2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ init_ori
            self.pose = np.block([self.end2_pos[:3], self.end2_ori])
            self.w_R_1b = self.get_wR()
            self.w_R_2b = self.w_R_1b @ T1[:3, :3]
            self.b1_P_1e_2e = self.end2_pos[:3] - self.end1_pos[:3]
            self.w_P_1b_2e = self.end2_pos[:3] - init_pos[:3]

            lg.add_arc(init_pos[:3], self.end1_pos[:3], init_ori, self.end1_ori)
            lg.add_line(self.end1_pos[:3], self.base2_pos[:3])
            lg.add_arc(self.base2_pos[:3], self.end2_pos[:3], self.base2_ori, self.end2_ori)
        
        elif self.mode == 4:
            T1 = np.eye(4)
            T1[2, 3] = self.Ls
            T2 = self.get_trans_mat(self.theta_1, self.L1, self.delta_1)
            T3 = np.eye(4)
            T3[2, 3] = self.Lr
            T4 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)

            base1_pos = T1 @ init_pos
            base1_ori = T1[:3, :3] @ init_ori
            end1_pos = T1 @ T2 @ init_pos
            end1_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
            base2_pos = T1 @ T2 @ T3 @ init_pos
            base2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ init_ori
            end2_pos = T1 @ T2 @ T3 @ T4 @ init_pos
            end2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ T4[:3, :3] @ init_ori
            self.pose = np.block([end2_pos[:3], end2_ori])
            self.w_R_1b = self.get_wR()
            self.w_R_2b = self.w_R_1b @ T2[:3, :3]
            self.b1_P_1e_2e = end2_pos[:3] - end1_pos[:3]
            self.w_P_1b_2e = end2_pos[:3] - base1_pos[:3]

            lg.add_line(init_pos[:3], base1_pos[:3])
            lg.add_arc(base1_pos[:3], end1_pos[:3], base1_ori, end1_ori)
            lg.add_line(end1_pos[:3], base2_pos[:3])
            lg.add_arc(base2_pos[:3], end2_pos[:3], base2_ori, end2_ori)

        # lg.plot_segments()
        print("Current mode: ", self.mode, "Current pose: ", self.pose)

    def check_transition(self):
        if self.mode == 1 and self.L2 > self.L_20:
            self.state_transition(1, 2)
        elif self.mode == 2 and self.Lr < 0:
            self.state_transition(2, 1)
        elif self.mode == 2 and self.Lr > self.L_r0:
            self.state_transition(2, 3)
        elif self.mode == 3 and self.L1 < 0:
            self.state_transition(3, 2)
        elif self.mode == 3 and self.L1 > self.L_10:
            self.state_transition(3, 4)
        elif self.mode == 4 and self.Ls < 0:
            self.state_transition(4, 3)

    def state_transition(self, current_mode, new_mode):
        if current_mode == 1 and new_mode == 2:
            self.mode = 2
            self.Lr = self.L2 - self.L_20
            self.L2 = self.L_20

        elif current_mode == 2 and new_mode == 1:
            self.mode = 1
            self.L2 = self.Lr + self.L_20
            self.Lr = 0

        elif current_mode == 2 and new_mode == 3:
            self.mode = 3
            self.L1 = self.Lr - self.L_r0
            self.Lr = self.L_r0

        elif current_mode == 3 and new_mode == 2:
            self.mode = 2
            self.Lr = self.L1 + self.L_r0
            self.L1 = 0
            self.theta_1 = 0
            self.delta_1 = 0

        elif current_mode == 3 and new_mode == 4:
            self.mode = 4
            self.Ls = self.L1 - self.L_10
            
        elif current_mode == 4 and new_mode == 3:
            self.mode = 3
            self.L1 = self.Ls + self.L_10
            self.Ls = 0

        else:
            raise ValueError("Invalid mode transition")

    def update_jacobians(self):
        self.J_1v3, self.J_1w3, self.J_1v2, self.J_1w2 = self.get_jacobians(self.theta_1, self.L1, self.delta_1)
        self.J_2v3, self.J_2w3, self.J_2v2, self.J_2w2 = self.get_jacobians(self.theta_2, self.L2, self.delta_2)
        if self.mode == 1:
            self.get_jacobian_1()
        elif self.mode == 2:
            self.get_jacobian_2()
        elif self.mode == 3:
            self.get_jacobian_3()
        elif self.mode == 4:
            self.get_jacobian_4()
    
    def step(self, time):
        """
        根据输出PHI更新状态
        模拟现实世界运动情况
        """
        mode_mapping = {
            1: ['phi', 'theta_2', 'L2', 'delta_2'],
            2: ['phi', 'Lr', 'theta_2', 'delta_2'],
            3: ['phi', 'theta_1', 'L1', 'delta_1', 'theta_2', 'delta_2'],
            4: ['phi', 'Ls', 'theta_1', 'delta_1', 'theta_2', 'delta_2'],
        }
        # 根据当前模式更新对应的属性
        attrs_to_update = mode_mapping.get(self.mode, [])
        # 循环遍历每个属性
        for i, attr in enumerate(attrs_to_update):
            # 获取当前属性的值
            current_value = getattr(self, attr)     
            new_value = current_value + self.d_PHI[i] * time
            setattr(self, attr, new_value)


if __name__ == "__main__":
    csm = CSM(0.5, 0.5, 0.15, 0.15)
    csm.target_pose = np.array([0.5, 0.5, 0.5, 0, 1, 0])
    finshed = False
    step_size = 0.1
    try:
        while not finshed:
            if np.linalg.norm(csm.pose - csm.target_pose) < 1e-3:
                finshed = True
            csm.check_transition()
            csm.plot_manipulator()
            csm.update_jacobians()

            err_pos = csm.target_pose[:3] - csm.pose[:3]
            err_ori = csm.target_pose[3:] - csm.pose[3:]
            v = 1 * err_pos/np.linalg.norm(err_pos)
            w = 1 * err_ori/np.linalg.norm(err_ori)
            csm.get_dot_PHI(v, w)
            csm.step(step_size)


        print("Finished")
    except KeyboardInterrupt:
        print("Interrupted")
