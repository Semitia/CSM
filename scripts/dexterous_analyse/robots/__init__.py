"""
robots 模块
提供统一的机器人注册表，通过名称字符串获取对应的机器人实例
"""
import os
from .base_robot import BaseRobot
from .pinocchio_robot import PinocchioRobot
from .csm_robot import CsmRobot

# 配置文件根目录（相对于本文件的位置）
_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config"))


def _csm_config(filename: str) -> str:
    return os.path.join(_CONFIG_DIR, filename)


# ============================================================
# 机器人注册表
# key: 用户在入口脚本中填写的 ROBOT_NAME 字符串
# value: 返回 BaseRobot 实例的工厂函数（lambda）
# ============================================================
ROBOT_REGISTRY: dict = {
    "ur5": lambda: PinocchioRobot(
        robot_name="ur5",
        tcp_frame_name=None,
    ),
    "panda": lambda: PinocchioRobot(
        robot_name="panda",
        tcp_frame_name="panda_link8",
    ),
    "csm": lambda: CsmRobot(
        csm_config_path=_csm_config("csm_cfg_3.4mm.yaml"),
    ),
    "csm_0": lambda: CsmRobot(
        csm_config_path=_csm_config("csm_cfg_0.yaml"),
    ),
    "csm_tool": lambda: CsmRobot(
        csm_config_path=_csm_config("csm_cfg_0_tool.yaml"),
    ),
}

# 各机器人对应的离散化器配置文件
DISCR_CONFIG_REGISTRY: dict = {
    "ur5":      os.path.join(_CONFIG_DIR, "discr_cfg_ur5.json"),
    "panda":    os.path.join(_CONFIG_DIR, "discr_cfg_panda.json"),
    "csm":      os.path.join(_CONFIG_DIR, "discr_cfg_csm.json"),
    "csm_0":    os.path.join(_CONFIG_DIR, "discr_cfg_csm.json"),
    "csm_tool": os.path.join(_CONFIG_DIR, "discr_cfg_csm.json"),
}

# 各机器人对应的默认保存路径
SAVE_PATH_REGISTRY: dict = {
    "ur5":      "./data/ur5_fk_cmap.npz",
    "panda":    "./data/panda_fk_cmap.npz",
    "csm":      "./data/csm_fk_cmap.npz",
    "csm_0":    "./data/csm_0_fk_cmap.npz",
    "csm_tool": "./data/csm_tool_fk_cmap.npz",
}


def create_robot(robot_name: str) -> BaseRobot:
    """
    通过名称创建机器人实例（未加载，需在 worker 中调用 load()）
    :param robot_name: 注册表中的机器人名称
    :return: BaseRobot 实例
    """
    if robot_name not in ROBOT_REGISTRY:
        available = list(ROBOT_REGISTRY.keys())
        raise ValueError(f"未知机器人名称: '{robot_name}'，可用选项: {available}")
    return ROBOT_REGISTRY[robot_name]()


def robot_from_config_dict(config: dict) -> BaseRobot:
    """
    从 get_config_dict() 返回的配置字典重建机器人实例
    用于在 worker 进程中反序列化
    """
    robot_type = config.get("type")
    if robot_type == "pinocchio":
        return PinocchioRobot.from_config_dict(config)
    elif robot_type == "csm":
        return CsmRobot.from_config_dict(config)
    else:
        raise ValueError(f"未知机器人类型: '{robot_type}'")


__all__ = [
    "BaseRobot",
    "PinocchioRobot",
    "CsmRobot",
    "ROBOT_REGISTRY",
    "DISCR_CONFIG_REGISTRY",
    "SAVE_PATH_REGISTRY",
    "create_robot",
    "robot_from_config_dict",
]
