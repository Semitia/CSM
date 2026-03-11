import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from csm.model import CSM
from csm.utils import axis_angle_from_vectors, normalize_vector

delta_t = 0.001
v_lim = 0.3
w_lim = 3

csm = CSM(0.04, 0.06, 0.02, 0.15, np.pi / 2, 2 * np.pi / 3, delta_t)

# start_cfg = (1, 1.34, 0, 0.036, 0, 0, 0, 0.95, 0, 1.43)
# end_cfg   = (4, 1.77, 0.04, 0.06, 0.02, 0.127, 1.09, 0.99, -0.8, 2.2)
start_cfg = (4, 2.74, 0.04, 0.06, 0.02, 0.055, 1.13, 0.9, 0.53, 1.83)
end_cfg   = (1, -0.95, 0, 0.026, 0, 0, 0, 0.5, 0, 0.03)

csm.set_state(*end_cfg)
target_pose = csm.pose.copy()
print("Target pose:", target_pose)
csm.reset()
csm.set_state(*start_cfg)
csm.target_pose = target_pose.copy()

position_errors = []
orientation_errors = []
theta1_vals, theta2_vals = [], []
L1_vals, L2_vals = [], []

max_iter = 1000
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.05, top=0.95)


def compute_errors(csm, target_pose):
    pos_err = np.linalg.norm(csm.pose[:3] - target_pose[:3]) * 1000
    a = csm.pose[3:] / np.linalg.norm(csm.pose[3:])
    b = target_pose[3:] / np.linalg.norm(target_pose[3:])
    ang_err = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    return pos_err, ang_err


finished = False
def animate(i):
    global csm, finished, ani
    if finished:
        return
    csm.check_transition()
    csm.update()
    csm.update_jacobians()

    v = normalize_vector(csm.target_pose[:3] - csm.pose[:3]) * v_lim
    axis_hat, _ = axis_angle_from_vectors(csm.pose[3:], csm.target_pose[3:])
    w = w_lim * axis_hat

    csm.get_dot_PHI(v, w)
    csm.step()

    pos_err, ang_err = compute_errors(csm, target_pose)
    position_errors.append(pos_err)
    orientation_errors.append(ang_err)
    theta1_vals.append(csm.theta_1)
    theta2_vals.append(csm.theta_2)
    L1_vals.append(csm.L1)
    L2_vals.append(csm.L2)

    ax.clear()
    csm.plot_manipulator(ax)
    ax.set_title(f"Step {i} | PosErr={pos_err:.2f} mm | AngErr={ang_err:.3f} rad")

    if pos_err < 1.0 and ang_err < 0.02:
        print(f"Converged at step {i}")
        finished = True
        if ani is not None and ani.event_source is not None:
            ani.event_source.stop()
    elif i >= max_iter - 1:
        print("Max iteration reached")
        finished = True
        if ani is not None and ani.event_source is not None:
            ani.event_source.stop()


def plot_results():
    t = np.arange(len(position_errors)) * delta_t
    fig2, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(t, position_errors, 'b-', label='Position Error (mm)')
    ax2.plot(t, orientation_errors, 'r--', label='Orientation Error (rad)')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Position Error (mm)', color='b')
    ax2.set_ylabel('Orientation Error (rad)', color='r')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.title('Tracking Errors Over Time'); plt.tight_layout(); plt.show()

    fig3, ax3 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax3[0].plot(t, theta1_vals, label='θ1'); ax3[0].plot(t, theta2_vals, label='θ2')
    ax3[0].legend(); ax3[0].set_ylabel('Angle (rad)')
    ax3[1].plot(t, L1_vals, label='L1'); ax3[1].plot(t, L2_vals, label='L2')
    ax3[1].legend(); ax3[1].set_xlabel('Time (s)'); ax3[1].set_ylabel('Length (m)')
    plt.tight_layout(); plt.show()


ani = FuncAnimation(fig, animate, frames=max_iter, interval=20)
plt.show()
plot_results()
