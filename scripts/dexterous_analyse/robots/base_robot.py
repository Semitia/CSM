"""
BaseRobot 抽象基类
定义所有机器人类型的统一接口，用于 FK/IK cmap 构建
"""
from abc import ABC, abstractmethod
import numpy as np
import pinocchio as pin


class BaseRobot(ABC):
    """机器人抽象基类，定义统一接口"""

    @abstractmethod
    def load(self):
        """
        加载机器人模型（在 worker 进程内调用）
        由于 pinocchio 模型不能跨进程序列化，需要在每个子进程内独立加载
        """
        pass

    @abstractmethod
    def fk(self, q: np.ndarray) -> tuple:
        """
        正向运动学
        :param q: 关节配置
        :return: (tcp_position [3], tcp_rotation [3,3])
        """
        pass

    @abstractmethod
    def ik(self, target_pose: pin.SE3, q_init: np.ndarray) -> tuple:
        """
        逆向运动学
        :param target_pose: 目标位姿 (SE3)
        :param q_init: 初始关节配置
        :return: (success, q_solution)
        """
        pass

    @abstractmethod
    def has_collision(self, q: np.ndarray) -> bool:
        """
        碰撞检测
        :param q: 关节配置
        :return: True 表示有碰撞，False 表示无碰撞
        """
        pass

    @abstractmethod
    def sample_q(self, halton_vals: np.ndarray) -> np.ndarray:
        """
        将 [0,1] 范围的 Halton 序列值映射到关节空间
        :param halton_vals: Halton 序列值 [0,1]^n
        :return: 关节配置 q
        """
        pass

    @property
    @abstractmethod
    def n_halton_dims(self) -> int:
        """需要多少维 Halton 序列"""
        pass

    @property
    @abstractmethod
    def supports_ik(self) -> bool:
        """是否支持 IK"""
        pass

    @abstractmethod
    def get_config_dict(self) -> dict:
        """
        返回可序列化的配置字典，用于跨进程传递
        子进程可以通过此配置重建机器人实例
        """
        pass

    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: dict):
        """
        从配置字典创建机器人实例
        :param config: get_config_dict() 返回的配置
        :return: 机器人实例
        """
        pass
