# 通用状态响应计算模板（SymPy 版）
import sympy as sp

# 符号变量
s, t = sp.symbols('s t', real=True)
x01, x02 = sp.symbols('x01 x02')

# 系统矩阵/向量
A = sp.Matrix([[0, 5], [-1, -4]])
B = sp.Matrix([2, 3])
C = sp.Matrix([[1, 1]])
x0 = sp.Matrix([x01, x02])

I = sp.eye(A.shape[0])

# ====== 定义输入信号 ======
# (1) δ(t)      -> U(s) = 1
# (2) ε(t)      -> U(s) = 1/s
# (3) t         -> U(s) = 1/s^2
# (4) sin(t)    -> U(s) = 1/(s^2 + 1)
U_s = 1  # 在此修改不同输入

# ====== 求解 ======
Phi_s = (s*I - A).inv()                           # (sI - A)^(-1)
X_s   = sp.simplify(Phi_s * (x0 + B * U_s))       # 拉普拉斯域状态向量

# 对矩阵逐元素做反拉普拉斯
x_t = X_s.applyfunc(lambda expr: sp.inverse_laplace_transform(expr, s, t))
y_t = sp.simplify(C * x_t)

print("状态响应 x(t) =")
sp.pprint(x_t)
print("\n输出响应 y(t) =")
sp.pprint(y_t)
