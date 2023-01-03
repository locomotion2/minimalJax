import time
from typing import Tuple

import numpy as np
import optuna
from bert_utils.geometry import Box2D

from custom_envs.constants import COMMANDS, RESET_MOTOR_PARAM
from custom_envs.env.bert_base_env import BaseBertEnv
from custom_envs.reflex.reflex import ReflexStateMachine

# from gym import spaces


class ModesBertEnv(BaseBertEnv):
    """
    The Gym interface for learning a trot gait using modes.

    :param verbose:
    :param max_frequency: limit control frequency (in Hz)
    :param is_real_robot:
    :param one_param_per_leg: Optimize parameters of each leg separately
    """

    def __init__(
        self,
        verbose: int = 1,
        control_frequency: float = 60,
        is_real_robot: bool = False,
        limit_action_space_factor: float = 1.0,
        optimize_parameters: bool = False,
        one_param_per_leg: bool = False,
        backward: bool = False,
    ):
        super().__init__(verbose, control_frequency, is_real_robot)

        # Limit to consider the robot has fallen
        self.roll_over_limit = np.deg2rad(40) if self.is_real_robot else np.deg2rad(70)

        self.state_machine = ReflexStateMachine()

        # self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.get_params()),))

        self.optimize_parameters = optimize_parameters
        self.one_param_per_leg = one_param_per_leg

        self.backward = backward
        # Check for space behind (when using real robot, backward mode)
        if self.backward:
            self.box_limits = Box2D(x_min=-0.35, x_max=0.0, y_min=-0.10, y_max=0.10)
        else:
            # Forward mode:
            self.box_limits = Box2D(x_min=0.0, x_max=0.35, y_min=-0.10, y_max=0.10)

        print("RESET_MOTOR_PARAM", RESET_MOTOR_PARAM)

    def sample_params(self, trial: optuna.Trial) -> None:
        """
        Sampler for Optuna.

        :param trial: Optuna trial object.
        """
        self.state_machine.sample_params(trial)

    def set_params(self, params: np.ndarray):
        # Do not update parameters
        if not self.optimize_parameters:
            return
        # Rescale
        self.bang_bang_amplitude_angles = self.unscale_angle_amplitude(params[: len(params) // 2])
        self.bang_bang_thresholds = self.unscale_threshold(params[len(params) // 2 :])

    def get_params(self) -> np.ndarray:
        scaled_amplitudes = self.scale_angle_amplitude(self.bang_bang_amplitude_angles)
        scaled_thresholds = self.scale_threshold(self.bang_bang_thresholds)
        return np.array((scaled_amplitudes, scaled_thresholds)).flatten()

    def reset(self) -> np.ndarray:
        _ = super().reset()

        # Initial motor pos (default to "backward")
        self.motors_at_rest = RESET_MOTOR_PARAM.copy()

        # Change default motor pos
        self.server_step(COMMANDS.STEP, self.motors_at_rest.copy())
        time.sleep(1)

        # Important: wait to have the correct updated values
        # TODO(antonin): probably with Gazebo only
        obs = self.server_step(COMMANDS.STEP, self.motors_at_rest.copy())
        time.sleep(1)

        self.state_machine.reset(self.data["joint_torques"], self.data["joint_positions"])

        # self.current_motor_pos = inverse_polar(self.get_polar_coord(), self.motors_at_rest)
        # obs = self.server_step(COMMANDS.STEP, self.current_motor_pos.copy())

        self.start_time = time.time()

        return obs.astype(np.float32)

    def compute_reward(self, scaled_action: np.ndarray, done: bool) -> float:
        """
        Compute the reward for the current task.
        Currently only travelled distance.

        :param scaled_action: Action from the controller scaled in [-1, 1],
            may be used to compute continuity cost
        :return: task reward.
        """

        # use delta in x direction as distance that was travelled
        if self.backward:
            distance_reward = -self.delta_world_position.x
        else:
            distance_reward = self.delta_world_position.x

        # Do not reward agent if it has terminated due to fall/not headed/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            distance_reward = 0.0

        reward = distance_reward
        return reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        :param action:
        :return:
        """
        # Action could be the parameters?
        scaled_action = action
        # # Unscale to real world action (positions of the joints)
        # action = self.unscale_action(action)

        self.current_motor_pos = self.state_machine.update(self.data["joint_torques"])

        obs = self.server_step(COMMANDS.STEP, self.current_motor_pos.copy())
        # Update internal state if needed
        # (for instance n steps at targets, that should be decoupled from compute reward)
        self._on_step()
        done = self.is_terminal_state()
        if self.tracking_enabled:
            reward = self.compute_reward(scaled_action, done)
        else:
            reward = 0.0

        info = dict(current_state=self.current_state)

        info.update(self._additional_infos())

        if self.is_real_robot:
            # check if robot has left the boundaries or it lost tracking, should only happen when it moved too far
            if self.reset_needed():
                done = True
            # if done:
            #     self.print_reset_debug_info()
            #     # print(f"Termination reason: {self.termination_reason}")

        return obs.astype(np.float32), reward, done, info

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        if not self.tracking_enabled:
            return False
        has_fallen = self.has_fallen()
        return has_fallen

    def scale_angle_amplitude(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Rescale the amplitude from [low, high] to [-1, 1]
        """
        return self.linear_scale(amplitude, self.min_angle_amplitude, self.max_angle_amplitude)

    def scale_threshold(self, threshold: np.ndarray) -> np.ndarray:
        """
        Rescale the threshold from [low, high] to [-1, 1]
        """
        return self.linear_scale(threshold, self.min_threshold, self.max_threshold)

    def unscale_angle_amplitude(self, scaled_amplitude: np.ndarray) -> np.ndarray:
        """
        Rescale from [-1, 1] to [low, high]
        """
        return self.unscale(scaled_amplitude, self.min_angle_amplitude, self.max_angle_amplitude)

    def unscale_threshold(self, scaled_threshold: np.ndarray) -> np.ndarray:
        """
        Rescale from [-1, 1] to [low, high]
        """
        return self.unscale(scaled_threshold, self.min_threshold, self.max_threshold)
