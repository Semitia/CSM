import pinocchio as pin
import example_robot_data as erd
import numpy as np
import time
import hppfcl

# 1. 加载 7-DOF 的 Franka Panda 机械臂
robot = erd.load('panda')
model = robot.model
data = robot.data
pedestal_geom = pin.GeometryObject(
    "pedestal", 0, 
    pin.SE3(np.eye(3), np.array([0, 0, -0.1])), # 向下偏移一半高度
    hppfcl.Cylinder(0.15, 0.2)
)
pedestal_geom.meshColor = np.array([0.0, 0.0, 0.0, 1.0]) 
# 关键一步：将底座添加到机器人的视觉模型中
robot.visual_model.addGeometryObject(pedestal_geom)
# 如果你需要底座参与碰撞检测，也要添加到碰撞模型
# robot.collision_model.addGeometryObject(pedestal_geom)

print(f"成功加载模型: {model.name}")
print(f"关节配置空间维度 (nq): {model.nq}")

# 2. 初始化 Meshcat 可视化器
try:
    from pinocchio.visualize import MeshcatVisualizer
    robot.setVisualizer(MeshcatVisualizer())
    
    # 初始化并自动在默认浏览器中打开可视化窗口
    robot.initViewer(open=True) 
    # 将机器人的视觉模型 (Mesh/URDF) 加载到场景中
    robot.loadViewerModel()     
    print("Meshcat 可视化器已启动，请在浏览器中查看（如果未自动打开，请点击终端输出的 URL）。")
except ImportError as e:
    print("导入 Meshcat 失败，请确保运行了: conda install -c conda-forge meshcat-python")
    exit()

# 3. 让机械臂“动起来”
# 获取默认的初始关节位姿 (home position)
q_init = robot.q0 

# 运动仿真参数
t = 0.0
dt = 0.05       # 时间步长 (50ms)
duration = 5.0  # 仿真总时长 5 秒

print("\n开始运动仿真...")
while t < duration:
    # 构造一条简单的正弦轨迹：在初始位姿的基础上，给所有关节加上基于时间的正弦偏移
    # np.ones(model.nq) 表示给全部 7 个关节同步加上相同的角度变化
    q = q_init + 0.3 * np.sin(2 * t) * np.ones(model.nq)

    # 4. 核心计算：正运动学 (Forward Kinematics)
    # 这会根据当前的关节角度 q，更新 data 对象里所有连杆的位置和速度信息
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # 5. 更新可视化器中的机器人姿态
    robot.display(q)

    # 模拟真实时间推进
    time.sleep(dt)
    t += dt

print("运动仿真结束！")

# 6. 获取特定时刻的末端执行器位姿
# 假设我们想知道运动结束后，夹爪 ('panda_hand') 的三维坐标
frame_name = 'panda_hand'
if model.existFrame(frame_name):
    frame_id = model.getFrameId(frame_name)
    # data.oMf 保存了从世界坐标系 (o) 到各个局部坐标系 (f) 的位姿 (SE3对象)
    end_effector_pose = data.oMf[frame_id] 
    
    print(f"\n最终时刻 [{frame_name}] 的三维位姿:")
    print("平移向量 (x, y, z) 米:\n", end_effector_pose.translation)
    print("旋转矩阵 (3x3):\n", end_effector_pose.rotation)