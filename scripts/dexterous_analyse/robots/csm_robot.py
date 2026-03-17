"""
CsmRobot 类
封装 CSM 连续体机器人
"""
import numpy as np
import pinocchio as pin
from pathlib import Path
from .base_robot import BaseRobot


class CsmRobot(BaseRobot):
    """CSM 连续体机器人封装类"""

    def __init__(self, csm_config_path: str, valid_modes: list = None):
        """
        :param csm_config_path: CSM 配置文件路径（字符串，可序列化）
        :param valid_modes: 有效的结构模式列表，None 表示在 load() 时自动检测
        """
        self.csm_config_path = csm_config_path
        self._valid_modes_override = valid_modes
        self.valid_modes = valid_modes
        self.csm = None

    def load(self):
        """加载 CSM 模型（在 worker 进程内调用）"""
        from csm.model import CSM
        self.csm = CSM.from_config(Path(self.csm_config_path))

        # 自动检测有效模式
        if self._valid_modes_override is None:
            self.valid_modes = [1, 2, 3]
            if self.csm.L_s0 > 0:
                self.valid_modes.append(4)

    def fk(self, q: np.ndarray) -> tuple:
        """
        正向运动学
        注意：调用前需先通过 map_halton_to_csm() 设置好 CSM 内部状态，
        q 参数在 CSM 中不使用（传 None 即可）
        """
        self.csm.update()
        tcp_pose = self.csm.pose[:3]
        tcp_rot = self.csm.rotation_matrix
        return tcp_pose, tcp_rot

    def ik(self, target_pose: pin.SE3, q_init: np.ndarray) -> tuple:
        """CSM 不支持 IK"""
        raise NotImplementedError("CSM 机器人不支持逆向运动学")

    def has_collision(self, q: np.ndarray) -> bool:
        """CSM 默认无碰撞检测，直接返回 False"""
        return False

    def sample_q(self, halton_vals: np.ndarray) -> np.ndarray:
        """
        CSM 不使用标准关节向量，此方法仅透传 halton_vals。
        实际配置映射通过 map_halton_to_csm() 完成。
        """
        return halton_vals

    def map_halton_to_csm(self, mode: int, halton_vals: np.ndarray):
        """
        将 Halton 序列映射到 CSM 配置，并更新内部状态
        :param mode: CSM 结构模式 (1-4)
        :param halton_vals: Halton 序列值 [0,1]^6
        """
        self.csm.mode = mode
        self.csm.phi = halton_vals[0] * 2 * np.pi

        if mode == 1:
            self.csm.L2 = halton_vals[1] * self.csm.L_20
            self.csm.theta_2 = halton_vals[2] * (self.csm.kappa_20 * self.csm.L2)
            self.csm.delta_2 = halton_vals[3] * 2 * np.pi
        elif mode == 2:
            self.csm.Lr = halton_vals[1] * self.csm.L_r0
            self.csm.L2 = self.csm.L_20
            self.csm.theta_2 = halton_vals[2] * (self.csm.kappa_20 * self.csm.L2)
            self.csm.delta_2 = halton_vals[3] * 2 * np.pi
        elif mode == 3:
            self.csm.L1 = halton_vals[1] * self.csm.L_10
            self.csm.theta_1 = halton_vals[2] * (self.csm.kappa_10 * self.csm.L1)
            self.csm.delta_1 = halton_vals[3] * 2 * np.pi
            self.csm.L2 = self.csm.L_20
            self.csm.Lr = self.csm.L_r0
            self.csm.theta_2 = halton_vals[4] * (self.csm.kappa_20 * self.csm.L2)
            self.csm.delta_2 = halton_vals[5] * 2 * np.pi
        elif mode == 4:
            self.csm.Ls = halton_vals[1] * self.csm.L_s0
            self.csm.L1 = self.csm.L_10
            self.csm.theta_1 = halton_vals[2] * (self.csm.kappa_10 * self.csm.L1)
            self.csm.delta_1 = halton_vals[3] * 2 * np.pi
            self.csm.L2 = self.csm.L_20
            self.csm.Lr = self.csm.L_r0
            self.csm.theta_2 = halton_vals[4] * (self.csm.kappa_20 * self.csm.L2)
            self.csm.delta_2 = halton_vals[5] * 2 * np.pi

    @property
    def n_halton_dims(self) -> int:
        """CSM 最多需要 6 维 Halton 序列（mode 3 和 4）"""
        return 6

    @property
    def supports_ik(self) -> bool:
        """CSM 不支持 IK"""
        return False

    def get_config_dict(self) -> dict:
        """返回可序列化的配置"""
        return {
            "type": "csm",
            "csm_config_path": self.csm_config_path,
            "valid_modes": self._valid_modes_override,
        }

    @classmethod
    def from_config_dict(cls, config: dict):
        """从配置字典创建实例"""
        return cls(
            csm_config_path=config["csm_config_path"],
            valid_modes=config.get("valid_modes"),
        )
