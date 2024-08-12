import numpy as np
x_0 = np.array([[0.631],
                [0.549],
                [0.549]])
x_1 = np.array([[0.64],
                [0.56],
                [0.526]])

dot_phi = np.array([[-0.33961426],
                    [-0.67493065],
                    [0.07833683],
                    [-0.40514172]])
# dot_phi_pinv = np.linalg.pinv(dot_phi)
# J = dot_x @ dot_phi_pinv
# print(J)

delta = 1.369
theta = -1.017
phi = -1.054

R = np.array([  [0.494, 0.87, 0],
                [-0.87, 0.494, 0],
                [0, 0, 1]])

J_2w3 = np.array([  [np.sin(delta), 0, np.cos(delta)*np.sin(theta)],
                    [np.cos(delta), 0, -np.sin(delta)*np.sin(theta)],
                    [0, 0, np.cos(theta)-1]])

z_w = np.array([[0],
                [0],
                [1]])

J = np.hstack([z_w, R @ J_2w3])

expect_dx = J @ dot_phi
print(expect_dx)