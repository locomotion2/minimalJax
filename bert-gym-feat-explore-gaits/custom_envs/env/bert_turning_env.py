import math

import numpy as np
from bert_utils.geometry import Box2D, normalize_angle

from custom_envs.env.bert_base_env import BaseBertEnv


class TurningBertEnv(BaseBertEnv):
    """
    The Gym interface for learning a turning gait.

    :param threshold_center_deviation: how far the robot may deviate from the center until the episode is stopped
    :param weight_center_deviation: weight for the off center derivation in any axis
    :param weight_turning_angle: weight for the turned angle
    :param weight_continuity: weight for the enforcing continuity/smoothness in the command
    :param weight_angular_velocity: weight for angular velocity that is not around yaw
    :param verbose:
    :param is_real_robot:
    """

    def __init__(
        self,
        threshold_center_deviation: float = 0.4,
        weight_center_deviation: float = 1,
        weight_turning_angle: float = 5,
        weight_continuity: float = 0.1,
        weight_angular_velocity: float = 1.0,
        verbose: int = 1,
        control_frequency: float = 20,
        is_real_robot: bool = False,
        limit_action_space_factor: float = 1.0,
    ):
        super().__init__(verbose, control_frequency, is_real_robot, limit_action_space_factor)

        # Limit to consider the robot has fallen
        self.roll_over_limit = np.deg2rad(40)

        self.weight_continuity = weight_continuity
        self.weight_center_deviation = weight_center_deviation
        self.weight_turning_angle = weight_turning_angle
        self.weight_angular_velocity = weight_angular_velocity
        self.threshold_center_deviation = threshold_center_deviation

        # definitions for cost functions
        self.early_termination_penalty = 2

        # change box dimensions because we only want to turn on the spot
        self.box_limits = Box2D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2)

    def compute_reward(self, scaled_action, done: bool) -> float:

        deviation_cost = self.weight_center_deviation * self._xy_deviation_cost()
        angular_velocity_cost = self.weight_angular_velocity * self._masked_angular_velocity_cost()

        continuity_cost = self.weight_continuity * self._continuity_cost(scaled_action)

        # use delta in orientation as primary reward
        # the sign of the desired delta make the robot rotate clockwise or anti-clockwise
        turning_reward = np.rad2deg(self._heading_delta()) * self.weight_turning_angle

        if self.verbose > 1:
            print(f"Turning Reward: {turning_reward:.5f}", f"Continuity Cost: {continuity_cost:5f}")
            print(f"Deviation cost: {deviation_cost:.5f}")
            print(f"Angular velocity cost: {angular_velocity_cost:.5f}")

        # Do not reward agent if it has terminated due to fall/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            turning_reward = 0.0

        reward = -(deviation_cost + angular_velocity_cost + continuity_cost) + turning_reward

        if done:
            # give negative reward
            reward -= self.early_termination_penalty
        return reward

    def _masked_angular_velocity_cost(self) -> float:
        """
        Cost for rotational velocities around x and y axis
        :return: squared normalized mean velocity
        """
        mean_angular_velocity = np.mean(np.abs(self.ang_vel[0:2]) / self.max_ang_vel)  # normalize
        return mean_angular_velocity**2

    def _xy_deviation_cost(self) -> float:
        """
        Cost for deviating from the center of the treadmill (y = 0)
        :return: normalized squared value for deviation from a straight line
        """
        deviation = self._center_deviation() / self.threshold_center_deviation
        return deviation**2

    def _center_deviation(self) -> float:
        return np.sqrt(self.world_position.x**2 + self.world_position.y**2)

    def _heading_delta(self) -> float:
        """
        Computes the deviation from the expected heading
        :return: Normalized (-pi to pi) delta in heading
        """
        # assume heading and expected_heading is given in radians
        heading_offset = self.heading - self.last_heading
        self.last_heading = self.heading
        return normalize_angle(heading_offset)

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        has_fallen = self.has_fallen()
        is_centered = math.fabs(self._center_deviation()) < self.threshold_center_deviation
        is_crawling = self.is_crawling() and self.is_real_robot
        return has_fallen or not is_centered or is_crawling
