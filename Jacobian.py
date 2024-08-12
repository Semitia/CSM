import sympy as sp

# 定义符号变量
theta_t, delta_t, L_t = sp.symbols('theta_t delta_t L_t')
theta_dot, delta_dot, L_dot = sp.symbols('theta_dot delta_dot L_dot')

# 定义旋转矩阵 R_b_1, R_1_2, R_2_e
R_b_1 = sp.Matrix([
    [0, sp.cos(delta_t), sp.sin(delta_t)],
    [0, -sp.sin(delta_t), sp.cos(delta_t)],
    [1, 0, 0]
])

R_1_2 = sp.Matrix([
    [sp.cos(theta_t), -sp.sin(theta_t), 0],
    [sp.sin(theta_t), sp.cos(theta_t), 0],
    [0, 0, 1]
])

R_2_e = sp.Matrix([
    [0, 0, 1],
    [sp.cos(delta_t), -sp.sin(delta_t), 0],
    [sp.sin(delta_t), sp.cos(delta_t), 0]
])

# 计算旋转矩阵 R_te
R_te = R_b_1 * R_1_2 * R_2_e

# 计算 R_te 的时间导数
R_te_dot = R_te.diff(theta_t) * theta_dot + R_te.diff(delta_t) * delta_dot + R_te.diff(L_t) * L_dot

# 计算角速度反对称矩阵 omega_hat
omega_hat = R_te.T * R_te_dot

# 提取角速度向量 omega
omega = sp.Matrix([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

# 计算角速度雅可比矩阵 J_w
J_w = omega.jacobian([theta_dot, delta_dot, L_dot])

# 显示结果
sp.pprint(J_w)
