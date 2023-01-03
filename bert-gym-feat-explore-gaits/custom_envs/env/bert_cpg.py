import os
import time
from collections import defaultdict
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Tuple

import cpg_gamepad
import numpy as np
import optuna
import yaml
from bert_utils.geometry import Box2D
from cpg_gamepad.cpg import COUPLING_MATRICES, GAITS, CPGController, skew_sim_matrix
from gym import spaces

from custom_envs.constants import COMMANDS, RESET_MOTOR_PARAM
from custom_envs.env.bert_base_env import BaseBertEnv

# Match reset pos
cpg_gamepad.cpg.RESET_MOTOR_PARAM = RESET_MOTOR_PARAM


class Task(Enum):
    SPEED = "speed"
    BALANCE = "balance"
    TURN = "turn"


class Symmetry(Enum):
    NONE = None
    TROT = "trot"
    BOUND = "bound"
    PACE = "pace"


class OffsetsAxes(Enum):
    BOTH = "both"
    ONLY_X = "only_x"
    ONLY_Z = "only_z"


class CPGBertEnv(BaseBertEnv):
    """
    The Gym interface for learning a trot gait using modes.

    :param verbose:
    :param control_frequency: limit control frequency (in Hz)
    :param is_real_robot:
    :param one_param_per_leg: Optimize parameters of each leg separately
    """

    def __init__(
        self,
        verbose: int = 1,
        control_frequency: float = 60,
        is_real_robot: bool = True,
        one_param_per_leg: bool = True,
        gait: str = "trot",
        backward: bool = False,
        enable_rl_offsets: bool = False,
        max_offset: float = 0.01,  # in m, offset range is [-max, max]
        task: str = "speed",
        symmetry: Optional[str] = None,
        offset_axes: str = "both",
        action_repeat: int = 1,
        explore_gaits: bool = False,
        optimize_cpg_params: bool = True,
        optimize_only_omega: bool = False,
        cpg_parameters: Optional[Dict[str, float]] = None,
    ):
        super().__init__(verbose, control_frequency, is_real_robot)

        # Limit to consider the robot has fallen
        self.roll_over_limit = np.deg2rad(45) if self.is_real_robot else np.deg2rad(70)

        # Default params
        self.current_gait = GAITS(gait)
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Load config file
        with open(os.path.join(base_path, "../cpg/config.yml")) as f:
            parameters: Dict[str, float] = yaml.safe_load(f)[self.current_gait.value]

        if cpg_parameters is not None:
            # Override params
            print("Overriding CPG params")
            pprint(cpg_parameters)
            parameters.update(cpg_parameters)

        self.omega_swing = parameters["omega_swing"]
        self.omega_stance = parameters["omega_stance"]
        self.desired_step_len = parameters["desired_step_len"]
        self.ground_clearance = parameters["ground_clearance"]
        self.ground_penetration = parameters["ground_penetration"]

        self.one_param_per_leg = one_param_per_leg
        self.explore_gaits = explore_gaits
        self.optimize_cpg_params = optimize_cpg_params
        self.optimize_only_omega = optimize_only_omega

        self.enable_rl_offsets = enable_rl_offsets
        self.max_offset = max_offset
        self.task = Task(task)
        self.symmetry = Symmetry(symmetry)
        self.offset_axes = OffsetsAxes(offset_axes)
        self.action_repeat = action_repeat

        # x,z coordinate of the leg
        cpg_dimensions = 2 * self.num_legs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dimension + cpg_dimensions,))
        # x, z delta for each leg
        n_actions = 2 * self.num_legs
        if self.symmetry != Symmetry.NONE:
            # Reduce action space by two
            n_actions /= 2

        if self.offset_axes != OffsetsAxes.BOTH:
            # Reduce action space by two
            # only offset in x or z
            n_actions /= 2

        self.action_space = spaces.Box(low=-1, high=1, shape=(int(n_actions),))

        self.backward = backward
        # Check for space behind (when using real robot, backward mode)
        if self.backward:
            self.box_limits = Box2D(x_min=-0.4, x_max=0.0, y_min=-0.10, y_max=0.10)
        else:
            # Forward mode:
            self.box_limits = Box2D(x_min=0.0, x_max=0.4, y_min=-0.10, y_max=0.10)

        self.cpg_controller = CPGController(self.current_gait, backward=self.backward, verbose=0)
        self.last_rotation = np.zeros(3)

        self.coupling_matrix = COUPLING_MATRICES[self.current_gait]

        self.reward_hist = defaultdict(list)

        print()
        print("Task", self.task)
        print("Gait", self.current_gait)
        print("Symmetry", self.symmetry)
        print("Max offset", self.max_offset)
        print("Offset Axes", self.offset_axes)
        print("Action repeat", self.action_repeat)
        print("RESET_MOTOR_PARAM", RESET_MOTOR_PARAM)
        print()

    def get_parameters(self) -> Dict[str, float]:
        return {
            "omega_swing": self.omega_swing,
            "omega_stance": self.omega_stance,
            "desired_step_len": self.desired_step_len,
            "ground_clearance": self.ground_clearance,
            "ground_penetration": self.ground_penetration,
        }

    def sample_params(self, trial: optuna.Trial) -> None:
        """
        Sampler for Optuna.

        :param trial: Optuna trial object.
        """
        if self.explore_gaits:
            self.leg_phase_shifts = {
                "front_right": 0,
                "front_left": 0,
                "rear_right": 0,
                "rear_left": 0,
            }
            self.leg_phase_shifts = {key: trial.suggest_float(f"{key}", 0, 1.0) for key in self.leg_phase_shifts}
            self.coupling_matrix = 2 * np.pi * skew_sim_matrix(**self.leg_phase_shifts)

        # One set of parameters for all legs
        if not self.one_param_per_leg and self.optimize_cpg_params:
            self.omega_swing = trial.suggest_float("omega_swing", 0.5, 14)  # * np.ones((self.num_legs,))
            self.omega_stance = trial.suggest_float("omega_stance", 0.5, 14)

            if self.optimize_only_omega:
                return

            if self.current_gait == GAITS.PRONK:
                self.desired_step_len = 0.0
            else:
                self.desired_step_len = trial.suggest_float("desired_step_len", 0.02, 0.06)
                self.ground_clearance = trial.suggest_float("ground_clearance", 0.02, 0.06)
                self.ground_penetration = trial.suggest_float("ground_penetration", 0.0, 0.03)
            return

        # One parameter per leg
        # for leg_idx in range(self.num_legs):
        #     self.ground_clearance[leg_idx] = trial.suggest_float(f"ground_clearance_{leg_idx}", 0, 0.05)
        #     self.ground_penetration[leg_idx] = trial.suggest_float(f"ground_penetration_{leg_idx}", 0, 0.05)
        #     self.omega_swing[leg_idx] = trial.suggest_float(f"omega_swing_{leg_idx}", 2, 30)
        #     self.omega_stance[leg_idx] = trial.suggest_float(f"omega_stance_{leg_idx}", 2, 30)
        #     self.desired_step_len[leg_idx] = trial.suggest_float(f"desired_step_len_{leg_idx}", 2, 30)

    def extract_observation(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        obs = super().extract_observation(data)
        return np.concatenate((obs, self.cpg_controller.desired_coords))

    def reset(self) -> np.ndarray:
        obs = super().reset()
        # Hack to force reset
        self.cpg_controller.cpg = None

        self.cpg_controller.switch_gait(self.current_gait, backward=self.backward, parameters=self.get_parameters())
        self.cpg_controller.cpg.coupling_matrix = self.coupling_matrix
        # Update initial phase
        self.cpg_controller.cpg.X[1, :] = (self.coupling_matrix[0, :] + np.pi / 2) % (2 * np.pi)
        # Reset motor pos
        _ = self.server_step(COMMANDS.STEP, RESET_MOTOR_PARAM)

        if self.current_gait == GAITS.PRONK:
            time.sleep(1)
        elif self.current_gait in {GAITS.TROT, GAITS.FAST_TROT}:
            time.sleep(0.5)

        time.sleep(0.2)
        # Update observation
        obs = self.server_step(COMMANDS.STEP, RESET_MOTOR_PARAM)
        if self.tracking_enabled:
            self.last_rotation = self.current_rot.copy()

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

        angular_velocity_cost = self._angular_velocity_cost()
        center_deviation_cost = self._center_deviation_cost()

        # Do not reward agent if it has terminated due to fall/not headed/crawling/...
        # to avoid encouraging aggressive behavior
        if done:
            distance_reward = 0.0

        if self.task == Task.SPEED:
            reward = 20 * distance_reward - 0.01 * angular_velocity_cost

        elif self.task == Task.BALANCE:
            # Reward for velocity on the z axis
            v_speed_norm = 0.1  # 10 cm / s
            vertical_speed = (np.abs(self.delta_world_position.z) / self.dt) / v_speed_norm
            # TODO: try exponential cost,
            # normalized by design
            reward = (1 - angular_velocity_cost) + (0.2 - 5 * center_deviation_cost) + vertical_speed
        elif self.task == Task.TURN:

            desired_angular_speed = -np.deg2rad(40)  # in rad/s
            yaw_speed = (self.current_rot[2] - self.last_rotation[2]) / self.dt
            # Equivalent to
            # yaw_speed = self.ang_vel[2] # opposite sign :/

            current_speed = self.delta_world_position.x / self.dt
            angular_speed_margin = 0.8
            angular_speed_reward = np.exp(-((desired_angular_speed - yaw_speed) ** 2) / angular_speed_margin**2)

            desired_forward_speed = 0.20  # m/s
            speed_margin = 0.1
            forward_reward = np.exp(-((desired_forward_speed - current_speed) ** 2) / speed_margin**2)

            reward = 0.0 * forward_reward + 1.0 * angular_speed_reward
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return reward

    def _center_deviation_cost(self) -> float:
        max_center_deviation = 0.025  # for normalization
        return np.sqrt(self.delta_world_position.x**2 + self.delta_world_position.y**2) / max_center_deviation

    def _angular_velocity_cost(self) -> float:
        """
        Cost for rotational velocities around all axis except pitch
        :return: squared normalized mean velocity
        """
        indices = np.array([0, 2])  # do not penalize pitch
        mean_angular_velocity = np.mean(np.abs(self.ang_vel[indices]) / self.max_ang_vel)  # normalize
        return mean_angular_velocity**2

    def handle_symmetry(self, offsets_x: np.ndarray, offsets_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.symmetry == Symmetry.NONE:
            return offsets_x, offsets_z

        # output is (front left, front right, rear left, rear right)
        if self.symmetry == Symmetry.TROT:
            # input is (front left, front right)
            # rear right same as front left
            # rear left same as front right
            offsets_x = np.concatenate((offsets_x, offsets_x[::-1]))
            offsets_z = np.concatenate((offsets_z, offsets_z[::-1]))
        elif self.symmetry == Symmetry.BOUND:
            # input is (front left, rear left)
            # front right same as front left
            # rear right same as rear left
            offsets_x = np.array((offsets_x[0], offsets_x[0], offsets_x[1], offsets_x[1]))
            offsets_z = np.array((offsets_z[0], offsets_z[0], offsets_z[1], offsets_z[1]))

        elif self.symmetry == Symmetry.PACE:
            # input is (front left, front right)
            # rear right same as front right
            # rear left same as front left
            n_repeat = 2
            offsets_x = np.tile(offsets_x, n_repeat)
            offsets_z = np.tile(offsets_z, n_repeat)
        else:
            raise NotImplementedError(f"{self.symmetry} not implemented")
        return offsets_x, offsets_z

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        :param action:
        :return:
        """
        if self.enable_rl_offsets:
            offsets = action * self.max_offset
            if self.offset_axes == OffsetsAxes.BOTH:
                offsets_x, offsets_z = offsets[::2], offsets[1::2]
            elif self.offset_axes == OffsetsAxes.ONLY_X:
                offsets_x = offsets
                offsets_z = np.zeros_like(offsets_x)
            elif self.offset_axes == OffsetsAxes.ONLY_Z:
                offsets_z = offsets
                offsets_x = np.zeros_like(offsets_z)
            else:
                raise NotImplementedError(f"{self.offset_axes} support not implemented")

            offsets_x, offsets_z = self.handle_symmetry(offsets_x, offsets_z)
        else:
            offsets_x, offsets_z = None, None

        for _ in range(self.action_repeat):
            # Account for the difference between control frequency and dt
            # used to compute CPG target
            env_dt = 1 / self.control_frequency
            n_cpg_step_per_env_step = max(int(env_dt / self.cpg_controller.dt), 1)
            for _ in range(n_cpg_step_per_env_step):
                desired_motor_pos = self.cpg_controller.update(offsets_x=offsets_x, offsets_z=offsets_z)

            obs = self.server_step(COMMANDS.STEP, self.cpg_controller.direction * desired_motor_pos)

        # (for instance n steps at targets, that should be decoupled from compute reward)
        self._on_step()
        done = self.is_terminal_state()

        if self.tracking_enabled:
            reward = self.compute_reward(action, done)
            self.last_rotation = self.current_rot.copy()
        else:
            reward = 0.0
        info = {}
        info.update(self._additional_infos())

        if self.is_real_robot:
            # check if robot has left the boundaries or it lost tracking, should only happen when it moved too far
            if self.reset_needed():
                # Timeout when leaving tracking
                info["TimeLimit.truncated"] = not done and self.out_of_bounds()
                if info["TimeLimit.truncated"] is True:
                    print("out of bound truncation")
                done = True

        return obs.astype(np.float32), reward, done, info

    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        if not self.tracking_enabled:
            return False
        has_fallen = self.has_fallen()
        return has_fallen
