from enum import Enum
from typing import List

import numpy as np
import optuna
from bert_utils import inverse_polar, joint_angles_to_polar

from custom_envs.constants import RESET_MOTOR_PARAM

# Raise error instead of numpy warnings
# np.seterr(all="raise")

GAZEBO_SIM = False

# TODO: fix real robot convention
if GAZEBO_SIM:

    class LEG(Enum):
        FRONT_RIGHT = 0
        FRONT_LEFT = 1
        HIND_LEFT = 2
        HIND_RIGHT = 3

    class LegIndices(Enum):
        FRONT_RIGHT = [0, 1]
        FRONT_LEFT = [2, 3]
        HIND_LEFT = [4, 5]
        HIND_RIGHT = [6, 7]

else:
    # Real robot
    class LEG(Enum):
        FRONT_LEFT = 0
        FRONT_RIGHT = 1
        HIND_RIGHT = 2
        HIND_LEFT = 3

    class LegIndices(Enum):
        FRONT_LEFT = [0, 1]
        FRONT_RIGHT = [2, 3]
        HIND_RIGHT = [4, 5]
        HIND_LEFT = [6, 7]


# Indices of the motors for each diagonal
LEFT_DIAGONAL = [LEG.FRONT_LEFT.value, LEG.HIND_RIGHT.value]
RIGHT_DIAGONAL = [LEG.FRONT_RIGHT.value, LEG.HIND_LEFT.value]

LEG_TO_INDICES = {
    LEG.FRONT_RIGHT: LegIndices.FRONT_RIGHT.value,
    LEG.FRONT_LEFT: LegIndices.FRONT_LEFT.value,
    LEG.HIND_LEFT: LegIndices.HIND_LEFT.value,
    LEG.HIND_RIGHT: LegIndices.HIND_RIGHT.value,
}

# Pre-compute leg indices
LEG_INDICES = [np.array(LEG_TO_INDICES[leg]) for leg in LEG]


class State(Enum):
    RIGHT_DIAGONAL_PULL_UP = 0
    LEFT_DIAGONAL_POLAR_PUSH = 1  # torque > threshold take down
    LEFT_DIAGONAL_RADIAL_PUSH = 2  # torque < threshold take off
    LEFT_DIAGONAL_PULL_UP = 3
    RIGHT_DIAGONAL_POLAR_PUSH = 4  # torque > threshold take down
    RIGHT_DIAGONAL_RADIAL_PUSH = 5  # torque < threshold take off


STATES = list(State)

# Set next states
for idx in range(len(STATES)):
    STATES[idx].next = STATES[(idx + 1) % len(STATES)]


class ReflexStateMachine:
    def __init__(self):
        self.num_legs = 4
        self.num_joints_per_leg = 2
        self.num_motors = self.num_legs * self.num_joints_per_leg
        # Input: joint torques Output: 1D signal (in the manifold)
        # Initial weights: same normalized weights for each input
        self.init_weights = 1 / np.sqrt(self.num_joints_per_leg)
        self.encoder_weights = self.init_weights * np.ones((self.num_joints_per_leg, 1))
        # Tied weights for now
        self.decoder_weights = self.encoder_weights.T
        # Values at rest to normalized
        self.torques_at_rest = np.zeros((self.num_motors,))
        self.motors_at_rest = np.zeros((self.num_motors,))
        # Will be changed on reset
        self.q_init = self.motors_at_rest.copy()

        # Will be changed on reset
        self.radius_at_rest = 0.15  # in m?
        self.angle_at_rest = 0.0  # in rad
        self.current_radii = self.radius_at_rest * np.ones((self.num_legs,))
        self.current_angles = self.angle_at_rest * np.ones((self.num_legs,))
        self.measured_radii = self.radius_at_rest * np.ones((self.num_legs,))
        self.measured_angles = self.angle_at_rest * np.ones((self.num_legs,))

        self.radii_at_rest = self.current_radii.copy()
        self.angles_at_rest = self.current_angles.copy()

        self.current_motor_pos = self.motors_at_rest.copy()
        # Delta theta in simulink
        self.delta_motor_pos = np.zeros_like(self.current_motor_pos)

        # 2 params per leg (amplitude and threshold, for now using symmetric threshold)
        initial_threshold = 0.05
        initial_amplitude_angle = np.deg2rad(17)  # 17 degrees
        initial_amplitude_radius = 0.1
        initial_ground_clearance = 0.015  # foot swing height

        self.radius_offset = initial_ground_clearance * np.ones(
            (self.num_legs,)
        )  # two legs in contact offset so the two others are not

        self.ground_penetration = 0.01  # foot stance penetration into ground
        # Additional ground clearance for rear legs
        self.hind_leg_ground_clearance = 0.0  # 0.01
        # TODO: update
        # self.robot_height = 0.25  # in nominal case (standing)
        # self.des_step_len=0.04,  # desired step length

        self.bang_bang_amplitude_radii = initial_amplitude_radius * np.ones((self.num_legs,))
        self.bang_bang_amplitude_angles = initial_amplitude_angle * np.ones((self.num_legs,))
        self.bang_bang_thresholds = initial_threshold * np.ones((self.num_legs,))

        self.current_state = State.RIGHT_DIAGONAL_PULL_UP

        self.one_param_per_leg = False

    def sample_params(self, trial: optuna.Trial) -> None:
        """
        Sampler for Optuna.

        :param trial: Optuna trial object.
        """
        # One set of parameters for all legs
        if not self.one_param_per_leg:
            # Ground clearance
            self.radius_offset = trial.suggest_float("radius_offset", 0, 0.05) * np.ones((self.num_legs,))
            self.bang_bang_thresholds = trial.suggest_float("threshold", 0, 0.3) * np.ones((self.num_legs,))
            angle_amplitude = trial.suggest_float("angle_amplitude", 5, 20) * np.ones((self.num_legs,))
            self.bang_bang_amplitude_angles = np.deg2rad(angle_amplitude)
            self.bang_bang_amplitude_radii = trial.suggest_float("radius_amplitude", 0, 0.15) * np.ones((self.num_legs,))
            return

        # One parameter per leg
        for leg_idx in range(self.num_legs):
            self.radius_offset[leg_idx] = trial.suggest_float(f"radius_offset_{leg_idx}", 0, 0.05)
            self.bang_bang_thresholds[leg_idx] = trial.suggest_float(f"threshold_{leg_idx}", 0, 0.3)
            angle_amplitude = trial.suggest_float(f"angle_amplitude_{leg_idx}", 5, 20)
            self.bang_bang_amplitude_angles[leg_idx] = np.deg2rad(angle_amplitude)
            self.bang_bang_amplitude_radii[leg_idx] = trial.suggest_float(f"radius_amplitude_{leg_idx}", 0, 0.15)

    def reset(self, joint_torques: np.ndarray, joint_positions: np.ndarray):
        # Reset initial state
        self.current_state = State.RIGHT_DIAGONAL_PULL_UP
        # Initial motor pos (default to "backward")
        self.motors_at_rest = RESET_MOTOR_PARAM.copy()

        self.torques_at_rest = joint_torques
        # TODO(antonin): check that we can use RESET_MOTOR_PARAM here
        # self.set_polar_coord(joint_angles_to_polar(np.array(self.data["motor_positions"])))
        self.set_polar_coord(joint_angles_to_polar(self.motors_at_rest.copy()))

        self.radii_at_rest = self.current_radii.copy()
        self.angles_at_rest = self.current_angles.copy()
        # We can also use the motors at rest for disambiguation of the inverse
        self.q_init = joint_positions
        self.current_motor_pos = inverse_polar(self.get_polar_coord(), self.motors_at_rest)

        polar_coords = joint_angles_to_polar(joint_positions)
        self.measured_angles = polar_coords[::2]
        self.measured_radii = polar_coords[1::2]

    def update(self, joint_torques: np.ndarray) -> np.ndarray:  # -> Tuple[np.ndarray, np.ndarray]:
        self._handle_current_state(joint_torques)

        self.current_motor_pos = inverse_polar(self.get_polar_coord(), self.motors_at_rest) + self.delta_motor_pos

        # x = -self._desired_step_len * r * np.cos(theta)
        #
        # ground_clearance = self._ground_clearance * np.sin(theta)
        # ground_penetration = self._ground_penetration * np.sin(theta)
        # above_ground = np.sin(theta) > 0
        # ground_offset = above_ground * ground_clearance + (1.0 - above_ground) * ground_penetration
        # z = -self._robot_height + ground_offset
        #
        # return x, z
        return self.current_motor_pos

    def next_state(self):
        self.current_state = self.current_state.next

    def _handle_half_cycle(self, first_diagonal: List[int], second_diagonal: List[int], joint_torques: np.ndarray):
        if self.current_state in [State.LEFT_DIAGONAL_PULL_UP, State.RIGHT_DIAGONAL_PULL_UP]:
            # Initial motor pos for the first diagonal
            self.reset_radius(first_diagonal)
            # Retract legs from the second diagonal
            self.retract_legs(second_diagonal)

            # Check for landing
            if self.is_latent_above_threshold(first_diagonal, joint_torques):
                self.next_state()

        elif self.current_state in [State.LEFT_DIAGONAL_POLAR_PUSH, State.RIGHT_DIAGONAL_POLAR_PUSH]:
            # Retract legs from the first diagonal
            self.retract_legs(first_diagonal)
            # Extend back the legs from the second diagonal
            self.reset_legs(second_diagonal)

            self.next_state()

        elif self.current_state in [State.LEFT_DIAGONAL_RADIAL_PUSH, State.RIGHT_DIAGONAL_RADIAL_PUSH]:
            # Move legs from the first diagonal forward and extend legs
            self.move_forward_and_extend(first_diagonal)
            # Move legs from the second diagonal backward
            self.move_legs_backward(second_diagonal)

            # Note: Missing delay leg extend
            # Check for take off
            if self.is_latent_below_threshold(first_diagonal, joint_torques):
                self.next_state()

    def _handle_current_state(self, joint_torques: np.ndarray) -> None:
        """
        State machine for trotting.
        The idea is to keep each diagonal shifted by half a cycle.

        :param joint_torques: Measured joint torques, used to determine take off/landing
        """

        self.delta_motor_pos = np.zeros_like(self.current_motor_pos)

        if self.current_state in [
            State.RIGHT_DIAGONAL_PULL_UP,
            State.LEFT_DIAGONAL_POLAR_PUSH,
            State.LEFT_DIAGONAL_RADIAL_PUSH,
        ]:
            self._handle_half_cycle(LEFT_DIAGONAL, RIGHT_DIAGONAL, joint_torques)

        elif self.current_state in [
            State.LEFT_DIAGONAL_PULL_UP,
            State.RIGHT_DIAGONAL_POLAR_PUSH,
            State.RIGHT_DIAGONAL_RADIAL_PUSH,
        ]:
            self._handle_half_cycle(RIGHT_DIAGONAL, LEFT_DIAGONAL, joint_torques)
        else:
            raise ValueError(f"Unknown state: {self.current_state}")

    def set_polar_coord(self, polar_coords: np.ndarray) -> None:
        self.current_angles = polar_coords[::2]
        self.current_radii = polar_coords[1::2]

    def get_polar_coord(self) -> np.ndarray:
        vector = np.zeros((self.num_legs * 2,))
        vector[::2] = self.current_angles
        vector[1::2] = self.current_radii
        return vector

    def compute_weights(self, leg_idx: int) -> np.ndarray:
        """
        Compute the weights for the linear auto-encoder.
        Currently fixed (same weight for each joint), could be updated with gradient descent.

        :param leg_idx: Index of the leg to compute the weight for.
        :return: encoder weights
        """
        encoder_weights = self.init_weights * np.ones((self.num_joints_per_leg, 1))

        if self.q_init[LEG_INDICES[leg_idx][1]] < self.q_init[LEG_INDICES[leg_idx][0]]:
            encoder_weights *= -1

        return encoder_weights

    def leg_encode(self, leg_idx: int, joint_torques: np.ndarray) -> float:
        """
        Compute latent value for a given leg.

        :param leg_idx: Index of the leg (convertion is done using Enum)
        :param joint_torques: Measured joint torques
        :return: latent value
        """
        # Re-Compute weights
        encoder_weights = self.compute_weights(leg_idx)
        return self.encode(joint_torques[LEG_INDICES[leg_idx]] - self.torques_at_rest[LEG_INDICES[leg_idx]], encoder_weights)

    def leg_decode(self, leg_idx: int, latent_value: float) -> np.ndarray:
        # Re-Compute weights
        decoder_weights = self.compute_weights(leg_idx).T
        return self.decode(latent_value, decoder_weights)

    def get_latent_values(self, legs: List[int], joint_torques: np.ndarray) -> np.ndarray:
        return np.array([self.leg_encode(leg, joint_torques) for leg in legs])

    @staticmethod
    def encode(input_vector: np.ndarray, encoder_weights: np.ndarray) -> float:
        """
        Forward pass for the encoder part of the auto-encoder.

        :param input_vector: shape (1, 2)
        :param encoder_weights: shape (2, 1)
        :return: latent value
        """
        return (input_vector @ encoder_weights).flatten()[0]

    @staticmethod
    def decode(latent_value: float, decoder_weights: np.ndarray) -> np.ndarray:
        """
        Forward pass for the decoder part of the auto-encoder.
        :param latent_value:
        :param decoder_weights: shape (2, 1)
        :return: decoded value (shape: (1, 2))
        """
        return (latent_value * np.ones((1,))) @ decoder_weights

    def retract_legs(self, legs: List[int]) -> None:
        for leg in legs:
            self.current_radii[leg] = self.radii_at_rest[leg] - self.radius_offset[leg]
            # Additional ground clearance for rear legs
            if leg in [LEG.HIND_LEFT.value, LEG.HIND_RIGHT.value]:
                self.current_radii[leg] -= self.hind_leg_ground_clearance
            # self.current_angles[leg] = self.angles_at_rest[leg]

    def reset_radius(self, legs: List[int]) -> None:
        for leg in legs:
            self.current_radii[leg] = self.radii_at_rest[leg]

    def reset_angle(self, legs: List[int]) -> None:
        for leg in legs:
            self.current_angles[leg] = self.angles_at_rest[leg]

    def reset_legs(self, legs: List[int]) -> None:
        for leg in legs:
            self.current_radii[leg] = self.radii_at_rest[leg]
            self.current_angles[leg] = self.angles_at_rest[leg]

    def move_legs_backward(self, legs: List[int]) -> None:
        for leg in legs:
            self.current_radii[leg] = self.radii_at_rest[leg]
            self.current_angles[leg] = self.angles_at_rest[leg] - self.bang_bang_amplitude_angles[leg]

    def move_forward_and_extend(self, legs: List[int]) -> None:
        for leg in legs:
            self.current_angles[leg] = self.angles_at_rest[leg] + self.bang_bang_amplitude_angles[leg]
            self.delta_motor_pos[LEG_INDICES[leg]] = self.leg_decode(leg, self.bang_bang_amplitude_radii[leg])

    def is_latent_above_threshold(self, legs: List[int], joint_torques: np.ndarray) -> bool:
        """
        Check if the latent value for the torque is above a given threshold
        (used to detect landing) for a given set of legs.

        :param legs: indices of the legs to be checked
        :param joint_torques: measured joint torques
        :return: True if the latent value is above that threshold for all legs
        """
        return np.all(self.get_latent_values(legs, joint_torques) > self.bang_bang_thresholds[legs])

    def is_latent_below_threshold(self, legs: List[int], joint_torques: np.ndarray) -> bool:
        return np.all(self.get_latent_values(legs, joint_torques) < self.bang_bang_thresholds[legs])
