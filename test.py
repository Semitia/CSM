def plot_manipulator(self):
        init_pos = np.array([0, 0, 0, 1])
        init_ori = np.array([0, 0, 1])
        lg = LineGenerator()

        if self.mode == 1:
            T = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)
            end_pos = T @ init_pos
            end_ori = T[:3, :3] @ init_ori
            self.pose = np.block([end_pos[:3], end_ori])
            self.w_P_2b_2e = T[:3, 3]
            self.w_R_2b = self.get_wR()

            lg.add_arc(init_pos[:3], end_pos[:3], init_ori, end_ori)

        elif self.mode == 2:
            T1 = np.eye(4)
            T1[2, 3] = self.Lr
            T2 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)

            base2_pos = T1 @ init_pos
            base2_ori = T1[:3, :3] @ init_ori
            end2_pos = T1 @ T2 @ init_pos 
            end2_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
            self.pose = np.block([end2_pos[:3], end2_ori])
            self.w_P_2b_2e = T2[:3, 3]
            self.w_R_2b = self.get_wR()

            lg.add_line(init_pos[:3], base2_pos[:3])
            lg.add_arc(base2_pos[:3], end2_pos[:3], base2_ori, end2_ori)

        elif self.mode == 3:
            T1 = self.get_trans_mat(self.theta_1, self.L1, self.delta_1)
            T2 = np.eye(4)
            T2[2, 3] = self.Lr
            T3 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)

            end1_pos = T1 @ init_pos
            end1_ori = T1[:3, :3] @ init_ori
            base2_pos = T1 @ T2 @ init_pos
            base2_ori = T1[:3, :3] @ T2[:3, :3] @ init_ori
            end2_pos = T1 @ T2 @ T3 @ init_pos
            end2_ori = T1[:3, :3] @ T2[:3, :3] @ T3[:3, :3] @ init_ori
            self.pose = np.block([end2_pos[:3], end2_ori])
            self.w_R_1b = self.get_wR()
            self.w_R_2b = self.w_R_1b @ T1[:3, :3]
            self.b1_P_1e_2e = end2_pos[:3] - end1_pos[:3]
            self.w_P_1b_2e = end2_pos[:3] - init_pos[:3]

            lg.add_arc(init_pos[:3], end1_pos[:3], init_ori, end1_ori)
            lg.add_line(end1_pos[:3], base2_pos[:3])
            lg.add_arc(base2_pos[:3], end2_pos[:3], base2_ori, end2_ori)
        
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