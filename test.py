import numpy as np

def find_intersection(p1, d1, p2, d2):
    # 转换为numpy数组
    p1 = np.array(p1, dtype=np.float64)
    d1 = np.array(d1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    d2 = np.array(d2, dtype=np.float64)

    # 计算两个方向向量的叉乘
    cross_d1_d2 = np.cross(d1, d2)
    norm_cross_d1_d2 = np.linalg.norm(cross_d1_d2)

    # 如果叉乘的范数接近于零，说明射线平行或共线
    if norm_cross_d1_d2 < 1e-8:
        print("射线平行或共线，无交点")
        return None

    # 计算线性方程组的右侧常数项
    diff_p = p2 - p1

    # 计算方程组的解，即参数 t 和 s
    t = np.linalg.det([diff_p, d2, cross_d1_d2]) / norm_cross_d1_d2**2
    s = np.linalg.det([diff_p, d1, cross_d1_d2]) / norm_cross_d1_d2**2

    # 计算交点坐标
    intersection1 = p1 + t * d1
    intersection2 = p2 + s * d2

    # 验证交点是否相同（即射线相交）
    if np.allclose(intersection1, intersection2):
        return intersection1
    else:
        print("射线不相交")
        return None

# 示例用法
p1 = [0, 0, 0]
d1 = [1, 1, 1]
p2 = [1, 0, 0]
d2 = [0, 1, 1]

intersection = find_intersection(p1, d1, p2, d2)
if intersection is not None:
    print("射线的交点为：", intersection)
