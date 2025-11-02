# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 定义4个不同半径的“工作空间壳层”，由内到外
# radii = [50, 70, 90, 110]
# colors = ['r', 'orange', 'yellow', 'grey']

# # 创建球面网格
# theta = np.linspace(0, np.pi, 40)
# phi = np.linspace(0, 2 * np.pi, 40)
# theta, phi = np.meshgrid(theta, phi)
# alpha_list = [0.8, 0.1, 0.1, 0.08]
# i=0
# for r, color in zip(radii, colors):
#     x = r * np.sin(theta) * np.cos(phi)
#     y = r * np.sin(theta) * np.sin(phi)
#     z = r * np.cos(theta)
#     ax.plot_surface(x, y, z, color=color, alpha=alpha_list[i], linewidth=0, edgecolor='none')
#     i+=1
    
# # 设置可视化效果
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_box_aspect([1,1,1])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# 创建3D图
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])  # 坐标比例一致

def draw_shell(points, face_color, label=None):
    hull = ConvexHull(points)
    # 绘制外壳（半透明表面 + 轻边线，制造“纹理感”）
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                    triangles=hull.simplices,
                    linewidth=0.15,                # 边线宽度
                    edgecolor=(0, 0, 0, 0.12),     # 半透明黑边线
                    antialiased=True,
                    alpha=0.30,                    # 表面透明度
                    color=face_color,              # 填充颜色
                    shade=False)                   # 关闭自动阴影

# 生成4个不同构型的模拟点云
rng = np.random.default_rng(0)
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

for i, c in enumerate(colors):
    r = 60 + i * 12  # 每层半径
    # 生成外层点（球壳 + 噪声）
    P = rng.normal(size=(2000, 3))
    P = P / np.linalg.norm(P, axis=1, keepdims=True) * (r + rng.normal(0, 3, size=(2000, 1)))
    draw_shell(P, c, label=f'C{i+1}')

# 坐标轴设置
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)

# 去掉3D背景面（新版 Matplotlib 写法）
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((1, 1, 1, 0))  # 背景面透明
    axis.pane.fill = False                 # 关闭填充

# 可选：设定观察角度，让形状更立体（可调整）
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()
