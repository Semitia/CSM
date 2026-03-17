"""
PinocchioRobot 类
封装所有基于 Pinocchio 的机器人（UR5, Panda 等）
"""
import numpy as np
import pinocchio as pin
import example_robot_data as erd
import hppfcl
from .base_robot import BaseRobot


class PinocchioRobot(BaseRobot):
    """基于 Pinocchio 的机器人封装类"""

    def __init__(
        self,
        robot_name: str,
        tcp_frame_name: str = None,
        pedestal_height: float = 0.2,
        pedestal_radius: float = 0.15,
        pedestal_parent_joint_threshold: int = 1,
    ):
        """
        :param robot_name: example-robot-data 中的机器人名称 (如 'ur5', 'panda')
        :param tcp_frame_name: 末端执行器 Frame 名称，None 表示使用最后一个 Frame
        :param pedestal_height: 基座圆柱体高度
        :param pedestal_radius: 基座圆柱体半径
        :param pedestal_parent_joint_threshold: parentJoint > 此值才添加碰撞对
        """
        self.robot_name = robot_name
        self.tcp_frame_name = tcp_frame_name
        self.pedestal_height = pedestal_height
        self.pedestal_radius = pedestal_radius
        self.pedestal_parent_joint_threshold = pedestal_parent_joint_threshold

        # 这些属性在 load() 中初始化
        self.robot = None
        self.model = None
        self.data = None
        self.collision_model = None
        self.collision_data = None
        self.tcp_id = None

    def load(self):
        """加载 Pinocchio 模型和碰撞模型"""
        self.robot = erd.load(self.robot_name)
        self.model = self.robot.model
        self.data = self.robot.data
        self.collision_model = self.robot.collision_model

        # 添加基座碰撞体
        pedestal_geom = pin.GeometryObject(
            "pedestal",
            0,
            pin.SE3(np.eye(3), np.array([0, 0, -self.pedestal_height / 2])),
            hppfcl.Cylinder(self.pedestal_radius, self.pedestal_height),
        )
        pedestal_id = self.collision_model.addGeometryObject(pedestal_geom)

        # 精准添加碰撞对
        for i in range(len(self.collision_model.geometryObjects)):
            if i == pedestal_id:
                continue
            geom_obj = self.collision_model.geometryObjects[i]
            if geom_obj.parentJoint > self.pedestal_parent_joint_threshold:
                self.collision_model.addCollisionPair(pin.CollisionPair(i, pedestal_id))

        # 创建碰撞数据
        self.collision_data = self.collision_model.createData()

        # 获取 TCP Frame ID
        if self.tcp_frame_name:
            self.tcp_id = self.model.getFrameId(self.tcp_frame_name)
        else:
            self.tcp_id = self.model.nframes - 1

    def fk(self, q: np.ndarray) -> tuple:
        """正向运动学"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_id)
        tcp_pose = self.data.oMf[self.tcp_id].translation
        tcp_rot = self.data.oMf[self.tcp_id].rotation
        return tcp_pose, tcp_rot

    def ik(
        self,
        target_pose: pin.SE3,
        q_init: np.ndarray,
        eps: float = 1e-4,
        IT_MAX: int = 1000,
        DT: float = 1e-1,
        damp: float = 1e-12,
    ) -> tuple:
        """
        逆向运动学求解（Gauss-Newton 方法）
        :return: (success, q_solution)
        """
        q = q_init.copy()
        success = False

        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit

        for _ in range(IT_MAX):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.tcp_id)

            dMi = target_pose.actInv(self.data.oMf[self.tcp_id])
            err = pin.log(dMi).vector

            if np.linalg.norm(err) < eps:
                q_normalized = pin.normalize(self.model, q)
                if np.all(q_normalized >= q_min) and np.all(q_normalized <= q_max):
                    success = True
                break

            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.tcp_id, pin.ReferenceFrame.LOCAL
            )
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)

        return success, q

    def has_collision(self, q: np.ndarray) -> bool:
        """碰撞检测"""
        pin.computeCollisions(
            self.model, self.data, self.collision_model, self.collision_data, q, False
        )
        return any(result.isCollision() for result in self.collision_data.collisionResults)

    def sample_q(self, halton_vals: np.ndarray) -> np.ndarray:
        """将 [0,1] Halton 值映射到关节空间"""
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit
        return q_min + halton_vals * (q_max - q_min)

    @property
    def n_halton_dims(self) -> int:
        """需要的 Halton 序列维度 = 关节自由度"""
        return self.model.nq

    @property
    def supports_ik(self) -> bool:
        """Pinocchio 机器人支持 IK"""
        return True

    def get_config_dict(self) -> dict:
        """返回可序列化的配置"""
        return {
            "type": "pinocchio",
            "robot_name": self.robot_name,
            "tcp_frame_name": self.tcp_frame_name,
            "pedestal_height": self.pedestal_height,
            "pedestal_radius": self.pedestal_radius,
            "pedestal_parent_joint_threshold": self.pedestal_parent_joint_threshold,
        }

    @classmethod
    def from_config_dict(cls, config: dict):
        """从配置字典创建实例"""
        return cls(
            robot_name=config["robot_name"],
            tcp_frame_name=config.get("tcp_frame_name"),
            pedestal_height=config.get("pedestal_height", 0.2),
            pedestal_radius=config.get("pedestal_radius", 0.15),
            pedestal_parent_joint_threshold=config.get("pedestal_parent_joint_threshold", 1),
        )
