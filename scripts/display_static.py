import time
import numpy as np
from csm.model import CSM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from csm.utils import calculate_angular_velocity, axis_angle_from_vectors, normalize_vector, load_workspace_data, get_random_target

config_path = Path("./config/csm_config1.yaml")
csm = CSM.from_config(config_path)
csm.set_state(4, 0, 0.04, 0.06, 0.02, 0.10, np.pi/4, -np.pi/4, 0, 0)
# mode 3
# csm.set_state(3, 0.816, 0.0396, 0.06, 0.02, 0, 1.49, 0.171, 0.303, 1.172)
# mode 4
# csm.set_state(4, 3.673, 0.04, 0.06, 0.02, 0.07, 1.57, 0.032, 4.66, 2.78)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
csm.target_pose = csm.pose
csm.plot_manipulator(ax)
plt.show()