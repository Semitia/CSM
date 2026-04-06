"""
Module: model.py
Description: Core CSM (Continuum Sugery Manipulator) model class defining kinematics, jacobians, and state updates.
"""
import numpy as np
import matplotlib.pyplot as plt
from .utils import calculate_angular_velocity, skew_symmetric_matrix, damped_pseudo_inverse


class CSM:
    def __init__(self, L_10, L_20, L_r0, L_s0, L_tool,
                 theta1_max=np.pi/2, theta2_max=2*np.pi/3, delta_t=0.01, ri_min=None):
        """
        初始化连续体机器人模型
        参数:
            L_10: segment 1的长度 (m)
            L_20: segment 2的长度 (m)
            L_r0: rigid 段长度 (m)
            L_s0: base  段长度 (m)
            L_tool: tool  长度 (m)
            theta1_max: segment 1的最大弯曲角度 (rad)
            theta2_max: segment 2的最大弯曲角度 (rad)
            delta_t: 每步时间间隔 (s)
            ri_min: 最小弯曲半径 (m)，用于限制最大曲率
        """
        self.mode = 1
        self.delta_t = delta_t
        self.L_10 = L_10
        self.L_20 = L_20
        self.L_r0 = L_r0
        self.L_s0 = L_s0
        self.L_tool = L_tool
        self.phi = 0
        self.L1 = 0
        self.L2 = self.L_20
        self.Lr = 0
        self.Ls = 0
        self.theta_1 = 0
        self.theta_2 = 0
        self.delta_1 = 0
        self.delta_2 = 0
        self.theta1_max = float(theta1_max)
        self.theta2_max = float(theta2_max)
        self.ri_min = None if ri_min is None else float(ri_min)

        self.kappa_10 = self.theta1_max / L_10
        self.kappa_20 = self.theta2_max / L_20
        if self.ri_min is not None:
            if self.ri_min <= 0:
                raise ValueError("ri_min must be positive.")
            curvature_limit = 1.0 / self.ri_min
            self.kappa_10 = min(self.kappa_10, curvature_limit)
            self.kappa_20 = min(self.kappa_20, curvature_limit)

        self.theta1_limit = self.kappa_10 * self.L_10
        self.theta2_limit = self.kappa_20 * self.L_20
        self._seg_J = {
            1: {"v2": None, "w2": None, "v3": None, "w3": None},
            2: {"v2": None, "w2": None, "v3": None, "w3": None},
        }
        self._mode_J = {m: {"v": None, "w": None} for m in (1, 2, 3, 4)}
        self.w_P_1b_2e = None
        self.w_P_2b_2e = None
        self.w_R_1b = None
        self.w_R_2b = None
        self.b1_P_1e_2e = None
        self.d_PHI = np.zeros(4)
        self.pose = np.array([0, 0, 0, 0, 0, 1], dtype=float)
        self.last_pose = np.array([0, 0, 0, 0, 0, 1], dtype=float)
        self.target_pose = np.array([0, 0, 0, 0, 0, 1], dtype=float)
        self.target_delta_pos = np.zeros(3)
        self.target_delta_ori = np.zeros(3)
        self.pre_delta_pos = np.zeros(3)
        self.pre_delta_ori = np.zeros(3)
        self.base1_pos = np.array([0, 0, 0, 1], dtype=float)
        self.base1_ori = np.array([0, 0, 1], dtype=float)
        self.end1_pos = np.array([0, 0, 0, 1], dtype=float)
        self.end1_ori = np.array([0, 0, 1], dtype=float)
        self.base2_pos = np.array([0, 0, 0, 1], dtype=float)
        self.base2_ori = np.array([0, 0, 1], dtype=float)
        self.end2_pos = np.array([0, 0, 0, 1], dtype=float)
        self.end2_ori = np.array([0, 0, 1], dtype=float)
        self.update()
        self.update_jacobians()

    def reset(self):
        self.mode = 1
        self.phi = 0
        self.L1 = 0
        self.L2 = self.L_20
        self.Lr = 0
        self.Ls = 0
        self.theta_1 = 0
        self.theta_2 = 0
        self.delta_1 = 0
        self.delta_2 = 0
        self.update()
        self.update_jacobians()

    def get_jacobians(self, theta, L, delta):
        if theta == 0:
            J_v3 = np.array([
                [ np.cos(delta)*L/2, 0, 0],
                [-np.sin(delta)*L/2, 0, 0],
                [0, 1, 0]
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
        z_w = np.array([[0], [0], [1]])
        J2 = self._seg_J[2]
        self._mode_J[1]["v"] = np.hstack([-skew_symmetric_matrix(self.w_P_2b_2e) @ z_w, self.w_R_2b @ J2["v3"]])
        self._mode_J[1]["w"] = np.hstack([z_w, self.w_R_2b @ J2["w3"]])

    def get_jacobian_2(self):
        z_w = np.array([[0], [0], [1]])
        J2 = self._seg_J[2]
        self._mode_J[2]["v"] = np.hstack([-skew_symmetric_matrix(self.w_P_2b_2e) @ z_w, z_w, self.w_R_2b @ J2["v2"]])
        self._mode_J[2]["w"] = np.hstack([z_w, np.zeros((3, 1)), self.w_R_2b @ J2["w2"]])

    def get_jacobian_3(self):
        z_w = np.array([[0], [0], [1]])
        J1, J2 = self._seg_J[1], self._seg_J[2]
        W2 = -skew_symmetric_matrix(self.b1_P_1e_2e) @ J1["w3"] + J1["v3"]
        self._mode_J[3]["v"] = np.hstack([-skew_symmetric_matrix(self.w_P_1b_2e) @ z_w, self.w_R_1b @ W2, self.w_R_2b @ J2["v2"]])
        self._mode_J[3]["w"] = np.hstack([z_w, self.w_R_1b @ J1["w3"], self.w_R_2b @ J2["w2"]])

    def get_jacobian_4(self):
        z_w = np.array([[0], [0], [1]])
        J1, J2 = self._seg_J[1], self._seg_J[2]
        W1 = -skew_symmetric_matrix(self.b1_P_1e_2e) @ J1["w2"] + J1["v2"]
        self._mode_J[4]["v"] = np.hstack([-skew_symmetric_matrix(self.w_P_1b_2e) @ z_w, z_w, self.w_R_1b @ W1, self.w_R_2b @ J2["v2"]])
        self._mode_J[4]["w"] = np.hstack([z_w, np.zeros((3, 1)), self.w_R_1b @ J1["w2"], self.w_R_2b @ J2["w2"]])

    def get_trans_mat(self, theta_t, L_t, delta_t):
        """
        计算单个连续体段的齐次变换矩阵
        参数:
            theta_t: 弯曲角度 (rad)
            L_t: 段长度 (m)
            delta_t: 偏转角度 (rad)
        返回:
            T: 齐次变换矩阵 (4,4)
        """
        R_b_1 = np.array([[0, np.cos(delta_t), np.sin(delta_t)],
                           [0, -np.sin(delta_t), np.cos(delta_t)],
                           [1, 0, 0]])
        R_1_2 = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                           [np.sin(theta_t), np.cos(theta_t), 0],
                           [0, 0, 1]])
        R_2_e = np.array([[0, 0, 1],
                           [np.cos(delta_t), -np.sin(delta_t), 0],
                           [np.sin(delta_t), np.cos(delta_t), 0]])
        T = np.eye(4)
        T[:3, :3] = R_b_1 @ R_1_2 @ R_2_e
        if theta_t == 0:
            T[:3, 3] = [0, 0, L_t]
        else:
            T[:3, 3] = [L_t * np.cos(delta_t) * (1 - np.cos(theta_t)) / theta_t,
                        L_t * np.sin(delta_t) * (np.cos(theta_t) - 1) / theta_t,
                        L_t * np.sin(theta_t) / theta_t]
        return T

    def get_w_T(self):
        """
        计算phi产生的齐次变换矩阵,从世界坐标系到stem
        1,2模式是2b, 3,4模式是1b
        """
        c, s = np.cos(self.phi), np.sin(self.phi)
        return np.array([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]])

    def _sample_constant_curvature(self, T_start, theta, length, delta, num_points):
        points = np.zeros((num_points, 3), dtype=float)
        rotations = np.zeros((num_points, 3, 3), dtype=float)
        if num_points == 1:
            s_vals = np.array([length], dtype=float)
        else:
            s_vals = np.linspace(0.0, length, num_points)
        if length <= 0:
            points[:] = T_start[:3, 3]
            rotations[:] = T_start[:3, :3]
            return points, rotations

        for i, s in enumerate(s_vals):
            theta_s = theta * (s / length)
            T_local = self.get_trans_mat(theta_s, s, delta)
            T_world = T_start @ T_local
            points[i] = T_world[:3, 3]
            rotations[i] = T_world[:3, :3]
        return points, rotations

    def _sample_straight_segment(self, T_start, length, num_points):
        points = np.zeros((num_points, 3), dtype=float)
        rotations = np.repeat(T_start[:3, :3][None, :, :], num_points, axis=0)
        if num_points == 1:
            s_vals = np.array([length], dtype=float)
        else:
            s_vals = np.linspace(0.0, length, num_points)
        for i, s in enumerate(s_vals):
            T_world = T_start.copy()
            T_world[:3, 3] = T_start[:3, 3] + T_start[:3, :3] @ np.array([0.0, 0.0, s])
            points[i] = T_world[:3, 3]
        return points, rotations

    def get_visualization_segments(self, arc_points=25, straight_points=8):
        T = self.get_w_T()
        segments = []

        def append_segment(kind, length, theta=0.0, delta=0.0, label=""):
            nonlocal T
            if length < 0:
                length = 0.0
            T_start = T.copy()
            if kind == "arc":
                points, rotations = self._sample_constant_curvature(
                    T_start, theta, length, delta, max(2, arc_points)
                )
                T = T_start @ self.get_trans_mat(theta, length, delta)
            else:
                points, rotations = self._sample_straight_segment(
                    T_start, length, max(2, straight_points)
                )
                T = T_start.copy()
                T[:3, 3] = T_start[:3, 3] + T_start[:3, :3] @ np.array([0.0, 0.0, length])
            segments.append({
                "kind": kind,
                "label": label,
                "length": length,
                "theta": theta,
                "delta": delta,
                "points": points,
                "rotations": rotations,
                "T_start": T_start,
                "T_end": T.copy(),
            })

        if self.mode == 1:
            append_segment("arc", self.L2, self.theta_2, self.delta_2, "seg2")
        elif self.mode == 2:
            append_segment("straight", self.Lr, label="rigid")
            append_segment("arc", self.L2, self.theta_2, self.delta_2, "seg2")
        elif self.mode == 3:
            append_segment("arc", self.L1, self.theta_1, self.delta_1, "seg1")
            append_segment("straight", self.Lr, label="rigid")
            append_segment("arc", self.L2, self.theta_2, self.delta_2, "seg2")
        elif self.mode == 4:
            append_segment("straight", self.Ls, label="base")
            append_segment("arc", self.L1, self.theta_1, self.delta_1, "seg1")
            append_segment("straight", self.Lr, label="rigid")
            append_segment("arc", self.L2, self.theta_2, self.delta_2, "seg2")

        tool_start = T.copy()
        tool_end = tool_start.copy()
        tool_end[:3, 3] = tool_start[:3, 3] + tool_start[:3, :3] @ np.array([0.0, 0.0, self.L_tool])
        return {
            "segments": segments,
            "tool": {
                "length": self.L_tool,
                "T_start": tool_start,
                "T_end": tool_end,
                "start": tool_start[:3, 3].copy(),
                "end": tool_end[:3, 3].copy(),
                "rotation": tool_start[:3, :3].copy(),
            }
        }

    def get_dot_PHI(self, v, w):
        Jv = self._mode_J[self.mode]["v"]
        Jw = self._mode_J[self.mode]["w"]
        n = Jv.shape[1]
        Jv_p = damped_pseudo_inverse(Jv)
        tem = np.eye(n) - Jv_p @ Jv
        self.d_PHI = Jv_p @ v + tem @ damped_pseudo_inverse(Jw @ tem) @ (w - Jw @ Jv_p @ v)

    def set_state(self, mode, phi, L1, L2, Lr, Ls, theta_1, theta_2, delta_1, delta_2):
        self.mode = mode
        self.phi = phi
        self.L1 = L1
        self.L2 = L2
        self.Lr = Lr
        self.Ls = Ls
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.update()
        self.update_jacobians()

    def update(self):
        init_pos = np.array([0, 0, 0, 1])
        init_ori = np.array([0, 0, 1])
        T0 = self.get_w_T()
        R0 = T0[:3, :3]

        if self.mode == 1:
            T1 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)
            T01 = T0 @ T1
            self.base2_pos = T0 @ init_pos
            self.base2_ori = R0 @ init_ori
            self.end2_pos = T01 @ init_pos
            self.end2_ori = T01[:3, :3] @ init_ori
            self.rotation_matrix = T01[:3, :3]
            self.w_P_2b_2e = self.end2_pos[:3] - self.base2_pos[:3]
            self.w_R_2b = R0

        elif self.mode == 2:
            T1 = np.eye(4); T1[2, 3] = self.Lr
            T2 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)
            T01 = T0 @ T1
            T012 = T01 @ T2
            self.base2_pos = T01 @ init_pos
            self.base2_ori = T01[:3, :3] @ init_ori
            self.end2_pos = T012 @ init_pos
            self.end2_ori = T012[:3, :3] @ init_ori
            self.rotation_matrix = T012[:3, :3]
            self.w_P_2b_2e = self.end2_pos[:3] - self.base2_pos[:3]
            self.w_R_2b = R0

        elif self.mode == 3:
            T1 = self.get_trans_mat(self.theta_1, self.L1, self.delta_1)
            T2 = np.eye(4); T2[2, 3] = self.Lr
            T3 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)
            T01 = T0 @ T1
            T012 = T01 @ T2
            T0123 = T012 @ T3
            self.base1_pos = T0 @ init_pos
            self.base1_ori = R0 @ init_ori
            self.end1_pos = T01 @ init_pos
            self.end1_ori = T01[:3, :3] @ init_ori
            self.base2_pos = T012 @ init_pos
            self.base2_ori = T012[:3, :3] @ init_ori
            self.end2_pos = T0123 @ init_pos
            self.end2_ori = T0123[:3, :3] @ init_ori
            self.rotation_matrix = T0123[:3, :3]
            self.w_R_1b = R0
            self.w_R_2b = R0 @ T1[:3, :3]
            self.b1_P_1e_2e = np.linalg.inv(R0) @ (self.end2_pos[:3] - self.end1_pos[:3])
            self.w_P_1b_2e = self.end2_pos[:3] - init_pos[:3]

        elif self.mode == 4:
            T1 = np.eye(4); T1[2, 3] = self.Ls
            T2 = self.get_trans_mat(self.theta_1, self.L1, self.delta_1)
            T3 = np.eye(4); T3[2, 3] = self.Lr
            T4 = self.get_trans_mat(self.theta_2, self.L2, self.delta_2)
            T01 = T0 @ T1
            T012 = T01 @ T2
            T0123 = T012 @ T3
            T01234 = T0123 @ T4
            self.base1_pos = T01 @ init_pos
            self.base1_ori = T01[:3, :3] @ init_ori
            self.end1_pos = T012 @ init_pos
            self.end1_ori = T012[:3, :3] @ init_ori
            self.base2_pos = T0123 @ init_pos
            self.base2_ori = T0123[:3, :3] @ init_ori
            self.end2_pos = T01234 @ init_pos
            self.end2_ori = T01234[:3, :3] @ init_ori
            self.rotation_matrix = T01234[:3, :3]
            self.w_R_1b = R0
            self.w_R_2b = R0 @ T2[:3, :3]
            self.b1_P_1e_2e = np.linalg.inv(R0) @ (self.end2_pos[:3] - self.end1_pos[:3])
            self.w_P_1b_2e = self.end2_pos[:3] - self.base1_pos[:3]

        # 计算工具末端在世界坐标系下的位置
        r_tool = self.end2_ori * self.L_tool
        self.tool_pos = self.end2_pos[:3] + r_tool
        # 更新 pose 为工具末端的位姿
        self.pose = np.block([self.tool_pos, self.end2_ori])
        # self.pose = np.block([self.end2_pos[:3], self.end2_ori])

    def debug(self):
        Jv = self._mode_J[self.mode]["v"]
        Jw = self._mode_J[self.mode]["w"]
        if Jv is None or Jw is None:
            return
        self.pre_delta_pos = Jv @ self.d_PHI * self.delta_t
        self.pre_delta_ori = Jw @ self.d_PHI
        delta_ori = calculate_angular_velocity(self.last_pose[3:], self.pose[3:], self.delta_t)
        self.last_pose = self.pose
        print(f"mode {self.mode} , pre_omega: [{', '.join([f'{x:.3f}' for x in self.pre_delta_ori])}] , omega: [{', '.join([f'{x:.3f}' for x in delta_ori])}]", "d_PHI: ", self.d_PHI * self.delta_t)

    def plot_manipulator(
        self,
        ax,
        reverse_color=False,
        render_mode="detailed",
        clear_ax=True,
        draw_target=True,
        configure_axes=True,
        title="Manipulator Movement",
    ):
        from .visualizer import Visualizer

        if not hasattr(self, "_visualizer"):
            self._visualizer = Visualizer(default_render_mode=render_mode)
        self._visualizer.plot(
            self,
            ax,
            reverse_color=reverse_color,
            render_mode=render_mode,
            clear_ax=clear_ax,
            configure_axes=configure_axes,
            title=title,
        )
        
        total_length = self.L_10 + self.L_20 + self.L_r0 + self.L_s0 + self.L_tool
        total_weight = total_length - self.L_s0
        if draw_target:
            tp, to = self.target_pose[:3], self.target_pose[3:6]
            ax.quiver(
                tp[0], tp[1], tp[2], to[0], to[1], to[2],
                length=0.15 * total_length, color='g', linewidth=2, arrow_length_ratio=0.6
            )
        if configure_axes:
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim([-total_weight, total_weight])
            ax.set_ylim([-total_weight, total_weight])
            ax.set_zlim([0, total_length])
            if title is not None:
                plt.title(title)
            plt.grid(True)

    def check_transition(self):
        if self.mode == 1 and self.L2 > self.L_20:
            self.state_transition(1, 2)
        elif self.mode == 2 and self.Lr < 0:
            self.state_transition(2, 1)
        elif self.mode == 2 and self.Lr > self.L_r0:
            self.state_transition(2, 3)
        elif self.mode == 3 and self.L1 < 0:
            self.state_transition(3, 2)
        elif self.mode == 3 and self.L1 > self.L_10 and self.L_s0 > 0:
            self.state_transition(3, 4)
        elif self.mode == 4 and self.Ls < 0:
            self.state_transition(4, 3)

    def state_transition(self, current_mode, new_mode):
        if current_mode == new_mode:
            print("same mode"); return
        if current_mode == 1 and new_mode == 2:
            self.mode = 2; self.Lr = self.L2 - self.L_20; self.L2 = self.L_20
        elif current_mode == 2 and new_mode == 1:
            self.mode = 1; self.L2 = self.Lr + self.L_20; self.Lr = 0
        elif current_mode == 2 and new_mode == 3:
            self.mode = 3; self.L1 = self.Lr - self.L_r0; self.Lr = self.L_r0
        elif current_mode == 3 and new_mode == 2:
            self.mode = 2; self.Lr = self.L1 + self.L_r0; self.L1 = 0
            self.theta_1 = 0; self.delta_1 = 0
        elif current_mode == 3 and new_mode == 4:
            self.mode = 4; self.Ls = self.L1 - self.L_10; self.L1 = self.L_10
        elif current_mode == 4 and new_mode == 3:
            self.mode = 3; self.L1 = self.Ls + self.L_10; self.Ls = 0
        else:
            raise ValueError(f"Invalid mode transition: {current_mode} -> {new_mode}")

    def update_jacobians(self):
        v3, w3, v2, w2 = self.get_jacobians(self.theta_1, self.L1, self.delta_1)
        self._seg_J[1] = {"v3": v3, "w3": w3, "v2": v2, "w2": w2}
        v3, w3, v2, w2 = self.get_jacobians(self.theta_2, self.L2, self.delta_2)
        self._seg_J[2] = {"v3": v3, "w3": w3, "v2": v2, "w2": w2}
        getattr(self, f"get_jacobian_{self.mode}")()
        
        if self.L_tool > 0:
            r_tool = self.tool_pos - self.end2_pos[:3]
            Jv = self._mode_J[self.mode]["v"]
            Jw = self._mode_J[self.mode]["w"]
            # 仅在内存中就地更新当前模式的线速度雅可比，不破坏原有结构
            self._mode_J[self.mode]["v"] = Jv - skew_symmetric_matrix(r_tool) @ Jw

    def apply_constraints(self, L_t, theta_t, kappa_t0):
        return max(L_t, theta_t / kappa_t0), min(theta_t, kappa_t0 * L_t)

    def step(self):
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
        for i, attr in enumerate(mode_mapping[self.mode]):
            setattr(self, attr, getattr(self, attr) + self.d_PHI[i] * self.delta_t)

        if self.L_s0 > 0:
            self.Ls = min(self.Ls, self.L_s0)
        else: # 没有base段，最多到mode3
            self.L1 = min(self.L1, self.L_10)
        if abs(self.theta_2) > self.kappa_20 * self.L2:
            self.theta_2 = self.kappa_20 * self.L2 * np.sign(self.theta_2)
        if self.mode in (3, 4) and abs(self.theta_1) > self.kappa_10 * self.L1:
            self.theta_1 = self.kappa_10 * self.L1 * np.sign(self.theta_1)

    @classmethod
    def from_config(cls, path):
        import yaml
        from pathlib import Path
        
        # 兼容传入字符串或 Path 对象
        config_path = Path(path)
        with config_path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        r = cfg["robot"]
        
        # cls 就代表 CSM 这个类本身，这里就等同于 return CSM(...)
        return cls(
            L_10=r["L_10"],
            L_20=r["L_20"],
            L_r0=r["L_r0"],
            L_s0=r["L_s0"],
            L_tool=r["L_tool"],
            theta1_max=r["theta1_max"],
            theta2_max=r["theta2_max"],
            ri_min=r.get("ri_min"),
            delta_t=r["delta_t"]
        )
