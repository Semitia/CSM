"""
Module: control_interface.py
Description: Interactive matplotlib control interface for real-time CSM simulation.
"""
from __future__ import annotations

import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .model import CSM
from .utils import axis_angle_from_vectors, normalize_vector


def _rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class ControlInterface:
    """
    Interactive end-effector controller based on incremental Jacobian IK.

    Translation is defined in the world frame:
    - W/S: +Y / -Y
    - A/D: +X / -X
    - Q/E: +Z / -Z

    Orientation is represented by the end-effector direction vector:
    - J/L: yaw around world Z
    - I/K: pitch around the current local right axis
    """

    def __init__(
        self,
        csm: CSM,
        fig=None,
        ax=None,
        render_mode="detailed",
        frame_interval_ms=20,
        linear_speed=0.02,
        angular_speed=2.5,
        tracking_linear_speed=None,
        tracking_angular_speed=None,
        position_gain=12.0,
        orientation_gain=10.0,
        orientation_key_speed=1.6,
        max_substeps=50,
        title="CSM Interactive Control",
    ):
        self.csm = csm
        self.render_mode = render_mode
        self.frame_interval_ms = int(frame_interval_ms)
        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        self.tracking_linear_speed = float(
            tracking_linear_speed if tracking_linear_speed is not None else max(linear_speed * 6.0, 0.08)
        )
        self.tracking_angular_speed = float(
            tracking_angular_speed if tracking_angular_speed is not None else max(angular_speed * 3.0, 3.0)
        )
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.orientation_key_speed = float(orientation_key_speed)
        self.max_substeps = int(max_substeps)
        self.title = title

        self.fig = fig if fig is not None else plt.figure(figsize=(10, 8))
        self.ax = ax if ax is not None else self.fig.add_subplot(111, projection="3d")
        self._animation = None
        self._closed = False
        self._closing = False
        self._status_artist = None
        self._keys_pressed = set()
        self.target_pose = np.asarray(self.csm.pose, dtype=float).copy()
        self.csm.target_pose = self.target_pose.copy()
        self._last_wall_time = None
        self._previous_sigint_handler = None
        self._frame_count = 0
        self._best_pos_error = np.inf
        self._best_ori_error = np.inf
        self._stall_frame_count = 0
        self._last_progress_frame = 0
        self._solver_status = "idle"
        self._solver_reason = "waiting for input"
        self._key_to_direction = {
            "w": np.array([0.0, 1.0, 0.0]),
            "s": np.array([0.0, -1.0, 0.0]),
            "a": np.array([1.0, 0.0, 0.0]),
            "d": np.array([-1.0, 0.0, 0.0]),
            "q": np.array([0.0, 0.0, 1.0]),
            "e": np.array([0.0, 0.0, -1.0]),
        }
        self._orientation_keys = {"i", "k", "j", "l"}

    def _disable_default_keymaps(self):
        for key in ("save", "home", "back", "forward", "quit", "fullscreen"):
            plt.rcParams[f"keymap.{key}"] = []

    def _connect_events(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_key_press(self, event):
        key = (event.key or "").lower()
        if key in self._key_to_direction or key in self._orientation_keys:
            self._keys_pressed.add(key)
        elif key == "escape":
            self.close()
        elif key == "r":
            self.csm.reset()
            self.target_pose = self.csm.pose.copy()
            self.csm.target_pose = self.target_pose.copy()
            self._reset_solver_status()

    def _on_key_release(self, event):
        key = (event.key or "").lower()
        self._keys_pressed.discard(key)

    def _on_close(self, _event):
        self.close()

    def _install_signal_handler(self):
        try:
            self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
        except ValueError:
            self._previous_sigint_handler = None

    def _restore_signal_handler(self):
        if self._previous_sigint_handler is None:
            return
        try:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
        except ValueError:
            pass
        self._previous_sigint_handler = None

    def _handle_sigint(self, _signum, _frame):
        self.close()

    def _request_event_loop_stop(self):
        canvas = getattr(self.fig, "canvas", None)
        if canvas is not None and hasattr(canvas, "stop_event_loop"):
            try:
                canvas.stop_event_loop()
            except Exception:
                pass

        manager = getattr(canvas, "manager", None)
        window = getattr(manager, "window", None)
        for attr in ("quit", "destroy", "close"):
            if window is not None and hasattr(window, attr):
                try:
                    getattr(window, attr)()
                except Exception:
                    pass

    def close(self):
        if self._closing:
            return
        self._closing = True
        self._closed = True
        if self._animation is not None and self._animation.event_source is not None:
            try:
                self._animation.event_source.stop()
            except Exception:
                pass
        self._request_event_loop_stop()
        try:
            plt.close(self.fig)
            plt.close("all")
        except Exception:
            pass

    def _reset_solver_status(self):
        self._frame_count = 0
        self._best_pos_error = np.inf
        self._best_ori_error = np.inf
        self._stall_frame_count = 0
        self._last_progress_frame = 0
        self._solver_status = "idle"
        self._solver_reason = "waiting for input"

    def _rotate_target_orientation(self, yaw_delta, pitch_delta):
        orientation = normalize_vector(self.target_pose[3:].copy())
        world_up = np.array([0.0, 0.0, 1.0])
        right_axis = np.cross(orientation, world_up)
        if np.linalg.norm(right_axis) < 1e-8:
            right_axis = np.array([1.0, 0.0, 0.0])
        else:
            right_axis = right_axis / np.linalg.norm(right_axis)

        yaw = _rotation_matrix(world_up, yaw_delta)
        pitch = _rotation_matrix(right_axis, pitch_delta)
        rotated = pitch @ (yaw @ orientation)
        self.target_pose[3:] = normalize_vector(rotated)

    def _target_translation_direction(self):
        direction = np.zeros(3)
        for key in self._keys_pressed:
            if key in self._key_to_direction:
                direction += self._key_to_direction[key]
        return normalize_vector(direction)

    def _advance_target_orientation(self, dt):
        yaw_sign = 0.0
        pitch_sign = 0.0
        if "j" in self._keys_pressed:
            yaw_sign += 1.0
        if "l" in self._keys_pressed:
            yaw_sign -= 1.0
        if "i" in self._keys_pressed:
            pitch_sign += 1.0
        if "k" in self._keys_pressed:
            pitch_sign -= 1.0
        if yaw_sign == 0.0 and pitch_sign == 0.0:
            return
        self._rotate_target_orientation(
            yaw_delta=self.orientation_key_speed * dt * yaw_sign,
            pitch_delta=self.orientation_key_speed * dt * pitch_sign,
        )

    def _has_manual_input(self):
        return bool(self._keys_pressed)

    def _advance_target_pose(self, dt):
        translation_direction = self._target_translation_direction()
        self.target_pose[:3] += translation_direction * self.linear_speed * dt
        self._advance_target_orientation(dt)
        self.target_pose[3:] = normalize_vector(self.target_pose[3:])
        self.csm.target_pose = self.target_pose.copy()

    def _solve_incremental_step(self):
        self.csm.check_transition()
        self.csm.update()
        self.csm.update_jacobians()

        pos_error = self.csm.target_pose[:3] - self.csm.pose[:3]
        pos_norm = np.linalg.norm(pos_error)
        linear_velocity = np.zeros(3)
        if pos_norm > 1e-9:
            linear_velocity = pos_error * min(self.position_gain, self.csm.delta_t ** -1)
            speed = np.linalg.norm(linear_velocity)
            if speed > self.tracking_linear_speed:
                linear_velocity *= self.tracking_linear_speed / speed

        axis_hat, theta = axis_angle_from_vectors(self.csm.pose[3:], self.csm.target_pose[3:])
        angular_velocity = axis_hat * min(self.tracking_angular_speed, self.orientation_gain * theta)

        self.csm.get_dot_PHI(linear_velocity, angular_velocity)
        self.csm.step()
        self.csm.check_transition()
        self.csm.update()
        self.csm.update_jacobians()

    def _update_solver_status(self):
        self._frame_count += 1
        pos_error = np.linalg.norm(self.csm.target_pose[:3] - self.csm.pose[:3])
        _, ori_error = axis_angle_from_vectors(self.csm.pose[3:], self.csm.target_pose[3:])
        manual_input = self._has_manual_input()

        pos_improved = pos_error < self._best_pos_error - 2e-4
        ori_improved = ori_error < self._best_ori_error - 1e-2
        if pos_improved:
            self._best_pos_error = pos_error
        if ori_improved:
            self._best_ori_error = ori_error
        if pos_improved or ori_improved:
            self._last_progress_frame = self._frame_count
            self._stall_frame_count = 0
        else:
            self._stall_frame_count += 1

        total_length = self.csm.L_10 + self.csm.L_20 + self.csm.L_r0 + self.csm.L_s0 + self.csm.L_tool
        radial_distance = np.linalg.norm(self.target_pose[:3])
        obviously_unreachable = radial_distance > total_length + 5e-3

        settled = pos_error < 1e-3 and ori_error < 0.03
        recently_progressed = (self._frame_count - self._last_progress_frame) <= 15
        stalled = self._stall_frame_count >= 45 and not manual_input and not settled

        if manual_input:
            self._solver_status = "manual_input"
            self._solver_reason = "target is being moved interactively"
        elif settled:
            self._solver_status = "solved"
            self._solver_reason = "target is tracked within tolerance"
        elif obviously_unreachable:
            self._solver_status = "likely_unreachable"
            self._solver_reason = "target radius exceeds total manipulator length"
        elif stalled:
            self._solver_status = "stalled"
            self._solver_reason = "error stopped decreasing; possible workspace limit or local IK stall"
        elif recently_progressed:
            self._solver_status = "converging"
            self._solver_reason = "tracking error is decreasing"
        else:
            self._solver_status = "tracking"
            self._solver_reason = "solver is attempting to follow the current target"

    def _status_text(self):
        pos_error = np.linalg.norm(self.csm.target_pose[:3] - self.csm.pose[:3]) * 1000.0
        _, theta = axis_angle_from_vectors(self.csm.pose[3:], self.csm.target_pose[3:])
        key_text = "".join(sorted(k.upper() for k in self._keys_pressed)) or "-"
        target_text = ", ".join(f"{x:.4f}" for x in self.target_pose[:3])
        ori_text = ", ".join(f"{x:.3f}" for x in normalize_vector(self.target_pose[3:]))
        return (
            "Controls: W/S:+/-Y  A/D:+/-X  Q/E:+/-Z  J/L:yaw  I/K:pitch  Mouse:view  R:reset  Esc/Ctrl+C:close\n"
            f"mode={self.csm.mode}  keys={key_text}  "
            f"pos_err={pos_error:.2f} mm  ori_err={theta:.3f} rad\n"
            f"solver={self._solver_status}  detail={self._solver_reason}\n"
            f"target_pos=[{target_text}]  target_dir=[{ori_text}]"
        )

    def _draw(self):
        self.csm.plot_manipulator(
            self.ax,
            render_mode=self.render_mode,
            title=self.title,
        )
        self._status_artist = self.ax.text2D(
            0.02,
            0.98,
            self._status_text(),
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#d1d5db"},
        )

    def _update_frame(self, _frame):
        if self._closed:
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.stop()
            return
        now = time.perf_counter()
        if self._last_wall_time is None:
            self._last_wall_time = now
        self._last_wall_time = now

        substeps = max(1, int(round((self.frame_interval_ms / 1000.0) / self.csm.delta_t)))
        substeps = min(substeps, self.max_substeps)
        for _ in range(substeps):
            self._advance_target_pose(self.csm.delta_t)
            self._solve_incremental_step()
        self._update_solver_status()
        self._draw()

    def start(self):
        self._disable_default_keymaps()
        self._install_signal_handler()
        self._connect_events()
        self._draw()
        self._animation = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=self.frame_interval_ms,
            cache_frame_data=False,
        )
        try:
            plt.show(block=True)
        finally:
            self.close()
            self._restore_signal_handler()


def launch_control_interface(
    csm: CSM,
    render_mode="detailed",
    frame_interval_ms=20,
    linear_speed=0.02,
    angular_speed=2.5,
    tracking_linear_speed=None,
    tracking_angular_speed=None,
    position_gain=12.0,
    orientation_gain=10.0,
    orientation_key_speed=1.6,
    max_substeps=50,
    title="CSM Interactive Control",
):
    interface = ControlInterface(
        csm=csm,
        render_mode=render_mode,
        frame_interval_ms=frame_interval_ms,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        tracking_linear_speed=tracking_linear_speed,
        tracking_angular_speed=tracking_angular_speed,
        position_gain=position_gain,
        orientation_gain=orientation_gain,
        orientation_key_speed=orientation_key_speed,
        max_substeps=max_substeps,
        title=title,
    )
    interface.start()
    return interface
