import math
from typing import Tuple

import gym
import numpy as np
from bert_utils import inverse_polar, joint_angles_to_polar
from bert_utils.geometry import normalize_angle

from custom_envs.constants import RESET_MOTOR_PARAM
from custom_envs.env.bert_base_env import BaseBertEnv


class WalkingBertEnv(BaseBertEnv):
    """
    The Gym interface for learning a walking gait.

    :param threshold_center_deviation: how far the robot may deviate from the center until the episode is stopped
    :param weight_center_deviation: weight for the off center derivation in y axis
    :param weight_distance_traveled: weight for the distance travelled in x axis
    :param weight_continuity: weight for the enforcing continuity/smoothness in the command
    :param weight_heading_deviation: weight for not walking with the right heading
    :param weight_angular_velocity: weight for any angular velocity
    :param verbose:
    :param is_real_robot:
    """

    def __init__(
        self,
        threshold_center_deviation: float = 0.2,
        weight_center_deviation: float = 1,
        weight_distance_traveled: float = 50,
        weight_continuity: float = 0.1,
        weight_heading_deviation: float = 1,
        weight_angular_velocity: float = 1.0,
        weight_energy_consumption: float = 0.4,
        verbose: int = 1,
        control_frequency: float = 20,
        is_real_robot: bool = False,
        limit_action_space_factor: float = 1.0,
        use_polar_coords: bool = False,
        desired_linear_speed: float = 0.3,  # in m/s
        weight_linear_speed: float = 0.0,
    ):
        super().__init__(verbose, control_frequency, is_real_robot, limit_action_space_factor)

        self.weight_continuity = weight_continuity
        self.weight_center_deviation = weight_center_deviation
        self.weight_distance_traveled = weight_distance_traveled
        self.weight_heading_deviation = weight_heading_deviation
        self.weight_angular_velocity = weight_angular_velocity
        self.weight_energy_consumption = weight_energy_consumption
        self.threshold_center_deviation = threshold_center_deviation
        self.weight_linear_speed = weight_linear_speed
        self.desired_linear_speed = desired_linear_speed

        # definitions for cost functions
        self.early_termination_penalty = 2
        self.heading_deviation_threshold_radians = np.deg2rad(45.0)
        # TODO(antonin): add polar coords to obs space
        self.use_polar_coords = use_polar_coords

        self.motors_at_rest = RESET_MOTOR_PARAM

        self.set_polar_coord(joint_angles_to_polar(self.motors_at_rest.copy()))
        # self.set_polar_coord(joint_angles_to_polar(self.motors_at_rest))
        self.radii_at_rest = self.current_radii.copy()
        self.angles_at_rest = self.current_angles.copy()

        self.min_angle_delta = np.deg2rad(-20) * np.ones((self.num_legs,))  # in degrees
        self.max_angle_delta = np.deg2rad(20) * np.ones((self.num_legs,))

        self.min_radius_delta = -0.02 * np.ones((self.num_legs,))
        self.max_radius_delta = 0.01 * np.ones((self.num_legs,))

        if self.use_polar_coords:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_legs * 2,))

    def set_polar_coord(self, polar_coords: np.ndarray) -> None:
        self.current_angles = polar_coords[::2]
        self.current_radii = polar_coords[1::2]

    def get_polar_coord(self) -> np.ndarray:
        vector = np.zeros((self.num_legs * 2,))
        vector[::2] = self.current_angles
        vector[1::2] = self.current_radii
        return vector

    def inverse_polar_coords(self, polar_action: np.ndarray) -> np.ndarray:
        angle_deltas = self.unscale(polar_action[: self.num_legs], self.min_angle_delta, self.max_angle_delta)
        radius_deltas = self.unscale(polar_action[self.num_legs :], self.min_radius_delta, self.max_radius_delta)
        self.current_angles = self.angles_at_rest + angle_deltas
        self.current_radii = self.radii_at_rest + radius_deltas
        return inverse_polar(self.get_polar_coord(), self.motors_at_rest)

    def step(self, action: np.ndarray):
        # Convert to desired motor pos
        if self.use_polar_coords:
            scaled_action = action.copy()
            action = self.inverse_polar_coords(action)

        return super().step(action, scaled_action=scaled_action)

    def compute_reward(self, scaled_action, done: bool) -> float:

        deviation_cost = self.weight_center_deviation * self._y_deviation_cost()
        angular_velocity_cost = self.weight_angular_velocity * self._angular_velocity_cost()
        energy_cost = self.weight_energy_consumption * self._energy_consumption_cost()

        continuity_cost = self.weight_continuity * self._continuity_cost(scaled_action)
        heading_cost, is_headed = self._heading_cost()
        heading_cost *= self.weight_heading_deviation
        # Desired delta in distance when weight_linear_speed > 0
        desired_delta = self.desired_linear_speed * self.wanted_dt

        linear_speed_cost = (desired_delta - self.delta_world_position.x) ** 2 / desired_delta**2
        linear_speed_cost = self.weight_linear_speed * linear_speed_cost

        distance_traveled = self.delta_world_position.x
        # Clip to be at most desired_delta
        if self.weight_linear_speed > 0.0:
            distance_traveled = np.clip(distance_traveled, -desired_delta, desired_delta)

        # use delta in x direction as distance that was travelled
        distance_reward = self.weight_distance_traveled * distance_traveled

        if self.verbose > 1:
            print(f"Distance Reward: {distance_reward:.5f}", f"Continuity Cost: {continuity_cost:5f}")
            print(f"Deviation cost: {deviation_cost:.2f}")
            print(f"Heading cost: {heading_cost:.2f}")
            print(f"Energy cost: {energy_cost:.2f}")

        # Do not reward agent if it has terminated due to fall/not headed/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            distance_reward = 0.0

        reward = (
            -(linear_speed_cost + deviation_cost + heading_cost + continuity_cost + angular_velocity_cost + energy_cost)
            + distance_reward
        )
        if done:
            # give negative reward
            reward -= self.early_termination_penalty
        return reward

    def _y_deviation_cost(self) -> float:
        """
        Cost for deviating from the center of the treadmill (y = 0)
        :return: normalized squared value for deviation from a straight line
        """
        deviation = self.world_position.y
        deviation = deviation / self.threshold_center_deviation
        return deviation**2

    def _heading_cost(self) -> Tuple[float, bool]:
        """
        Computes the deviation from the expected heading
        :return: Normalized (0 to 1) squared deviation from expected heading and bool if it is still headed correctly
        """
        # assume heading and expected_heading is given in radians
        heading_offset = normalize_angle(self.heading - self.robot.start_heading)
        heading_deviation = np.abs(heading_offset)
        heading_deviation = heading_deviation / self.heading_deviation_threshold_radians
        return heading_deviation**2, bool(heading_deviation < 1)

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        has_fallen = self.has_fallen()
        is_centered = math.fabs(self.world_position.y) < self.threshold_center_deviation
        # Deactivate crawling detecting for sim (negative height at the beginning)
        is_crawling = self.is_crawling() and self.is_real_robot
        _, is_headed = self._heading_cost()
        return has_fallen or not is_centered or not is_headed or is_crawling
