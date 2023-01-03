import math
from typing import Dict

import numpy as np
from bert_utils.geometry import normalize_angle
from gym import spaces

from custom_envs.env.bert_turning_env import TurningBertEnv


class FixedTurningBertEnv(TurningBertEnv):
    """
    The Gym interface for learning a turning gait.

    :param threshold_center_deviation: how far the robot may deviate from the center until the episode is stopped
    :param weight_center_deviation: weight for the off center derivation in any axis
    :param weight_turning_angle: weight for the turned angle
    :param weight_continuity: weight for the enforcing continuity/smoothness in the command
    :param weight_angular_velocity: weight for angular velocity that is not around yaw
    :param turning_angle: angle the robot should turn in degrees
    :param steps_at_target: number of update steps the robot has to be at the target before termination
    :param heading_target_threshold: minimum deviation from target heading which is counted as being at the target in deg
    :param goal_reward: reward which is gained if the target position is successfully reached
    :param verbose:
    :param is_real_robot:
    """

    def __init__(
        self,
        threshold_center_deviation: float = 0.4,
        weight_center_deviation: float = 1,
        weight_turning_angle: float = 4,
        weight_continuity: float = 0.1,
        weight_angular_velocity: float = 1.0,
        turning_angle: float = 45,
        steps_at_target: int = 20,  # TODO: replace with time to be independent of the control freq
        heading_target_threshold: float = 5.0,
        goal_reward: float = 100.0,
        verbose: int = 1,
        control_frequency: float = 20,
        max_torque: float = 1.0,
        is_real_robot: bool = False,
        randomize_target: bool = False,
        limit_action_space_factor: float = 1.0,
    ):
        super().__init__(
            threshold_center_deviation,
            weight_center_deviation,
            weight_turning_angle,
            weight_continuity,
            weight_angular_velocity,
            verbose,
            control_frequency,
            is_real_robot,
            limit_action_space_factor,
        )
        self.turning_angle_deg = turning_angle
        self.turning_angle_rad = np.deg2rad(turning_angle)
        self.expected_heading_rad = self.turning_angle_rad
        self.current_steps_at_target = 0
        self.heading_target_threshold = heading_target_threshold
        self.steps_at_target = steps_at_target
        self.goal_reward = goal_reward
        self._is_success = False
        self.early_termination_penalty = 100
        self.randomize_target = randomize_target
        self.crawling_height_limit = 0.07
        self.expected_heading_imu = 0.0

        control_dimension = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dimension + control_dimension,))

    def reset(self) -> np.ndarray:
        self.current_steps_at_target = 0

        if self.randomize_target:
            desired_angle = np.sign(np.random.rand() - 0.5) * np.random.uniform(
                low=self.heading_target_threshold + 1, high=self.turning_angle_deg
            )
            self.desired_angle_rad = np.deg2rad(desired_angle)
        else:
            self.desired_angle_rad = self.turning_angle_rad if np.random.rand() > 0.5 else -self.turning_angle_rad

        _ = super().reset()
        self.expected_heading_rad = normalize_angle(self.robot.start_heading + self.desired_angle_rad)

        if self.is_real_robot:
            # Should be the same as `expected_heading_rad`
            self.expected_heading_imu = normalize_angle(self.start_imu_orientation[2] + self.desired_angle_rad)
        # UPDATE the observation with the desired heading!
        observation = self.extract_observation(self.data)

        self._is_success = False
        return observation

    def _additional_infos(self):
        return dict(is_success=self._is_success)

    def extract_observation(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        lin_acc = np.array(data["lin_acc"])
        joint_torque = np.array(data["joint_torques"])
        joint_positions = np.array(data["joint_positions"])
        joint_velocities = np.array(data["joint_velocities"])

        heading_deviation = normalize_angle(self.heading - self.expected_heading_rad)

        if self.is_real_robot:
            heading_deviation = normalize_angle(self.imu_orientation[2] - self.expected_heading_imu)

        observation = np.concatenate(
            (
                self.current_rot,
                joint_torque,
                joint_positions,
                joint_velocities,
                self.ang_vel,
                lin_acc,
                np.array([heading_deviation, self.desired_angle_rad]),
            )
        )
        return observation

    def _on_step(self):
        super()._on_step()
        if self._at_target():
            self.current_steps_at_target += 1
        else:
            self.current_steps_at_target = 0

    def compute_reward(self, scaled_action, done: bool) -> float:

        deviation_cost = self.weight_center_deviation * self._xy_deviation_cost()
        angular_velocity_cost = self.weight_angular_velocity * self._masked_angular_velocity_cost()

        continuity_cost = self.weight_continuity * self._continuity_cost(scaled_action)

        max_distance = 90  # 90 degrees, for normalization
        distance_cost = np.rad2deg(self._heading_offset_to_target(self.expected_heading_rad)) / max_distance
        distance_cost = self.weight_turning_angle * distance_cost**2

        if self.verbose > 1:
            print(f"Distance cost: {distance_cost:.5f}", f"Continuity Cost: {continuity_cost:5f}")
            print(f"Deviation cost: {deviation_cost:.5f}")
            print(f"Angular velocity cost: {angular_velocity_cost:.5f}")

        reward = -(deviation_cost + angular_velocity_cost + continuity_cost + distance_cost)

        if done:
            # TODO: "fail" in self.termination_reason
            # is probably not needed anymore on the real robot
            # as the termination check is fixed
            if self._early_termination() or "fail" in self.termination_reason:
                # give negative reward
                reward -= self.early_termination_penalty
            elif self._at_target():
                self._is_success = True
                # give positive reward for reaching the goal without tipping over
                reward += self.goal_reward
                self.termination_reason = "success"

        return reward

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        # check if the robot has reached the target for long enough
        if (self.current_steps_at_target >= self.steps_at_target and self._at_target()) or self._early_termination():
            if self.termination_reason == "":
                self.termination_reason = f"num_steps_at_target:{self.current_steps_at_target}| _at_target={self._at_target()}"
            return True
        return False

    def _early_termination(self) -> bool:
        has_fallen = self.has_fallen()
        is_centered = math.fabs(self._center_deviation()) < self.threshold_center_deviation
        is_crawling = self.is_crawling() and self.is_real_robot
        if has_fallen:
            self.termination_reason = "has_fallen_fail"
        if not is_centered:
            self.termination_reason = "is_centered_fail"
        if is_crawling:
            self.termination_reason = "is_crawling_fail"

        return has_fallen or not is_centered or is_crawling

    def _heading_offset_to_target(self, expected_heading: float) -> float:
        """
        Computes the deviation from the expected heading
        :return: Normalized (-pi to pi) delta in heading
        """
        # assume heading and expected_heading is given in radians
        heading_offset = self.heading - expected_heading
        return normalize_angle(heading_offset)

    def _at_target(self) -> bool:
        heading_offset = np.abs(np.rad2deg(self._heading_offset_to_target(self.expected_heading_rad)))
        if heading_offset < self.heading_target_threshold:
            return True
        else:
            return False
