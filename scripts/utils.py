import json
import random
import numpy as np


def calculate_angular_velocity(v1, v2, delta_t):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    n = np.cross(v1, v2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if delta_t == 0:
        raise ValueError("delta_t cannot be zero")
    if np.linalg.norm(n) == 0:
        return np.array([0, 0, 0])
    return (theta / delta_t) * (n / np.linalg.norm(n))


def axis_angle_from_vectors(v1, v2, eps=1e-8):
    n1 = np.asarray(v1, dtype=float)
    n2 = np.asarray(v2, dtype=float)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    cross = np.cross(n1, n2)
    s = np.linalg.norm(cross)
    theta = np.arctan2(s, np.clip(np.dot(n1, n2), -1.0, 1.0))
    axis_hat = cross if s < eps else cross / s
    return axis_hat, theta


def normalize_vector(v, threshold=1e-6):
    norm = np.linalg.norm(v)
    return v if norm < threshold else v / norm


def load_workspace_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_random_target(data):
    target = random.choice(data)
    return target["mode"], np.array(target["pose"]), target.get("config", None)
