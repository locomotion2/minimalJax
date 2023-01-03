"""
Adapted from https://rmc-github.robotic.dlr.de/raff-an/neck_control
"""
import abc
import math
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
from bert_utils.geometry import Box2D, Point3D, Robot, normalize_angle, out_of_bounds
from bert_utils.rate_limiter import RateLimiter
from gym import spaces

from custom_envs.constants import (
    COMMANDS,
    RESET_MOTOR_PARAM,
    RESET_TYPE,
    TRACKING_DISABLED,
    TRACKING_LIMITS,
    TREADMILL_ENABLED,
    TREADMILL_MIN_POS,
)
from custom_envs.ln_servers.ln_grey_real import LNRealServer
from custom_envs.ln_servers.ln_grey_sim import LNSimServer


class BaseBertEnv(gym.Env, abc.ABC):
    """
    The Gym interface for controlling bert.

    :param verbose:
    :param control_frequency: limit control frequency (in Hz)
    :param is_real_robot:
    :param limit_action_space_factor: Reduce the action space (motor limits) by a given factor (between 0 and 1)
        This makes learning faster but reduces the possibilities.
    """

    def __init__(
        self,
        verbose: int = 1,
        control_frequency: float = 20.0,
        is_real_robot: bool = False,
        limit_action_space_factor: float = 1.0,
    ):

        super().__init__()

        self.verbose = verbose
        self.is_real_robot = is_real_robot
        self.tracking_enabled = not TRACKING_DISABLED

        self.control_frequency = control_frequency
        self.wanted_dt = 1.0 / self.control_frequency

        self.rate_limiter = RateLimiter(wanted_dt=self.wanted_dt, max_dt=2.0, verbose=self.verbose)

        self.num_steps = 0
        self.start_time = time.time()

        # Limit to consider the robot has fallen
        self.roll_over_limit = np.deg2rad(20) if self.is_real_robot else np.deg2rad(70)
        # Limit for trying to stand up from fallen position
        self.stand_up_roll_limit = np.deg2rad(40)
        # Height limit to assume that the robot is crawling
        self.crawling_height_limit = 0.08
        # Normalizing factor when penalizing large angular velocities
        self.max_ang_vel = np.deg2rad(35)  # empirical value

        # holds all the necessary information
        self.current_state = None
        self.robot_position = Point3D(np.zeros(3))  # x,y,z tracking position (without transform)
        self.current_rot = np.zeros(3)
        self.last_rotation_tmp = np.zeros(3)
        self.imu_orientation = np.zeros(3)
        self.start_imu_orientation = np.zeros(3)
        self.last_heading = 0

        self.lost_tracking = False
        self.ang_vel = np.zeros(3)
        self.last_action = None
        self.last_obs = None

        # For debug
        self.reset_counter = 0
        self.termination_reason = ""
        self.dt = 1 / control_frequency
        self.data = {}

        num_legs = 4
        num_joints_per_leg = 2
        num_var_per_joint = 3  # position,velocity, torque
        imu_input_size = 3 * 2  # 3D accel, gyro, magneto ?
        dim_current_rotation = 3
        dim_heading = 1  # deviation to desired heading
        # 4 legs with 2 joints and each has position, velocity, torque and pose
        # + current rotation + heading
        dim_additional = imu_input_size + dim_current_rotation + dim_heading

        # desired_positions of 4 legs with 2/3 joints
        self.action_dimensions = num_legs * num_joints_per_leg
        self.input_dimension = num_legs * num_joints_per_leg * num_var_per_joint + dim_additional
        self.num_legs = num_legs
        self.num_joints_per_leg = num_joints_per_leg

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dimension,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dimensions,))

        # Motor positions limits (grey bert)
        self.action_upper_limits = np.array([0.6, 0.5, 0.6, 0.5, 0.75, 0.5, 0.75, 0.5]) * limit_action_space_factor
        self.action_lower_limits = np.array([-0.75, -0.5, -0.75, -0.5, -0.6, -0.5, -0.6, -0.5]) * limit_action_space_factor
        self.motor_offsets = np.zeros_like(self.action_upper_limits)
        self.last_motor_pos = np.zeros_like(self.action_upper_limits)

        # Bounds when using the real robot (in m)
        self.tracking_limits = Box2D(
            x_min=TRACKING_LIMITS["x_min"],
            x_max=TRACKING_LIMITS["x_max"],
            y_min=TRACKING_LIMITS["y_min"],
            y_max=TRACKING_LIMITS["y_max"],
        )
        # Bounds for checking that the robot has space in front of it
        self.box_limits = Box2D(x_min=0.0, x_max=0.75, y_min=-0.15, y_max=0.15)

        self.robot = Robot()

        if self.verbose > 0:
            print(f"Roll over threshold: {np.rad2deg(self.roll_over_limit):.2f}")

        # Connect to the ln server:
        self.ln_server = LNRealServer() if self.is_real_robot else LNSimServer()
        self.ln_server.init()

    @property
    def world_position(self) -> Point3D:
        return self.robot.world_position

    @property
    def delta_world_position(self) -> Point3D:
        return self.robot.delta_world_position

    def reset(self) -> np.ndarray:
        self.last_action = np.zeros(self.action_dimensions)
        self.rate_limiter.reset_last_time()
        if self.is_real_robot and not self.tracking_enabled or bool(os.environ.get("FORCE_RESET", False)):
            # Tracking disabled
            self.manual_reset()
        elif self.is_real_robot and self.tracking_enabled:
            # only do a human reset if needed
            while self.reset_needed(first_step=True):
                if self.reset_counter == 0 or self.lost_tracking:
                    self.manual_reset()
                else:
                    # first try to get robot in standing position
                    if not self.try_standing_up():
                        self.manual_reset()
                        # TO CHECK: no reset_transform here? its not necessary but shouldn't be wrong either
                        continue
                    # try to move robot back via treadmill
                    if not self.can_fit_box() or self.out_of_bounds():
                        if self.treadmill_recovery_available():
                            self.server_step(COMMANDS.RESET, RESET_TYPE.TREADMILL)
                        else:
                            # TODO(antonin): this can be removed for walking (autoreset)
                            self.manual_reset()
                    else:
                        self.manual_reset()
                self.reset_transform()
        else:
            self.last_obs = self.server_step(COMMANDS.RESET)

        self.reset_transform()
        self.num_steps = 0
        self.start_time = time.time()
        self.termination_reason = ""
        self.last_heading = self.robot.start_heading
        return self.last_obs

    def manual_reset(self) -> None:
        if self.verbose > 0:
            print(f"==== Manual reset {self.reset_counter}===")
            # self.print_reset_debug_info()
        self.reset_counter += 1
        self.last_obs = self.server_step(COMMANDS.RESET, RESET_TYPE.MANUAL)

    def server_step(self, command: COMMANDS, action: Union[np.ndarray, RESET_TYPE, None] = None) -> np.ndarray:
        """
        :param command:
        :param action:
        :return:
        """
        # Offset will be added in the simulation
        if action is not None and not isinstance(action, RESET_TYPE):
            real_action = action - self.motor_offsets
        else:
            real_action = action

        # Save previous motor pos
        if "motor_positions" in self.data:
            self.last_motor_pos = np.array(self.data["motor_positions"]).copy()

        self.send_action(command, real_action)

        # self.data was update in send_action()
        data = self.data

        # For the sim, no motor dynamics, current motor pos is the desired pos
        # this changes also self.data
        if not self.is_real_robot and isinstance(real_action, np.ndarray):
            data["motor_positions"] = real_action.copy()

        if self.tracking_enabled:
            self.robot_position = Point3D(np.array(data["position_3d"]))
            self.current_rot = np.array(data["rotation_3d"])

        # FIXME: rotation of grey bert is somehow wrong
        self.heading = -normalize_angle(self.current_rot[2])  # extract yaw
        self.ang_vel = np.array(data["ang_vel"])

        # HACK: gyro seems buggy
        # if np.allclose(self.ang_vel, np.zeros(3)):
        #     # opposite sign :/
        #     self.ang_vel = -(self.current_rot - self.last_rotation_tmp) / self.dt
        #     self.last_rotation_tmp = self.current_rot.copy()

        if self.is_real_robot:
            self.lost_tracking = not data["tracking_active"] and self.tracking_enabled
            self.imu_orientation = np.array(data["imu_orientation"])

        self.update_world_position()

        is_init = command is None or command == COMMANDS.RESET
        self.dt = self.rate_limiter.update_control_frequency(init=is_init)

        observation = self.extract_observation(data)

        # Save last observation in case no manual reset is needed
        self.last_obs = observation.copy()
        return observation

    def extract_observation(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        lin_acc = np.array(data["lin_acc"])
        joint_torque = np.array(data["joint_torques"])
        joint_positions = np.array(data["joint_positions"])
        joint_velocities = np.array(data["joint_velocities"])
        # TODO(antonin): add motor pos (and velocity)
        # TODO(antonin): add polar coords or cartesian coords

        heading_deviation = normalize_angle(self.heading - self.robot.start_heading)

        if self.is_real_robot:
            heading_deviation = normalize_angle(self.imu_orientation[2] - self.start_imu_orientation[2])

        observation = np.concatenate(
            (
                self.current_rot,  # TODO(antonin): remove if not state estimation
                joint_torque,
                joint_positions,
                joint_velocities,
                self.ang_vel,
                lin_acc,
                # TODO(antonin): remove for cpg env? (or clip it)
                np.array([0.0 * heading_deviation]),
            )
        )
        return observation

    def send_action(self, command: COMMANDS, action: Union[np.ndarray, RESET_TYPE, None] = None) -> None:
        self.data = self.ln_server.step(command, action)
        # Debug message
        # if self.is_real_robot:
        #     debug = dict(
        #         out_of_bounds=self.out_of_bounds(),
        #         lost_tracking=self.lost_tracking,
        #         has_fallen=self.has_fallen(),
        #         can_fit_box=self.can_fit_box(),
        #         is_crawling=self.is_crawling(),
        #         last_obs_valid=self.last_obs is not None,
        #         transform=self.rotation_matrix.tolist(),
        #         translation=self.translation.tolist(),
        #         world_position=self.world_position.tolist(),
        #         heading=self.heading,
        #         start_heading=self.start_heading,
        #     )
        #     msg = dict(command=command.value, action=action, debug=debug)
        # else:
        #     msg = dict(command=command.value, action=action)
        # self.socket.send_json(msg)

    def update_world_position(self):
        self.robot.update_position(self.robot_position, self.heading, rotation=self.current_rot)

    def reset_transform(self) -> None:
        self.start_imu_orientation = self.imu_orientation
        self.robot.reset_local_frame(self.robot_position, self.heading, degrees=False, rotation=self.current_rot)

    def _additional_infos(self) -> Dict[str, Any]:
        return {}

    def _on_step(self) -> None:
        self.num_steps += 1
        # print every 2 seconds
        print_freq = 2 * self.control_frequency
        if self.num_steps % print_freq == 0:
            dt = time.time() - self.start_time
            print(f"{self.num_steps/dt:.2f} FPS")

    def step(self, action: np.ndarray, scaled_action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # Unscale to real world action (positions of the joints)
        if scaled_action is None:
            scaled_action = action
            action = self.unscale_action(action)

        if self.verbose > 1:
            print(f"action : {action}")

        # Safety clip (even though boundaries should be ensured by the agent)
        # action = np.clip(action, self.action_lower_limits, self.action_upper_limits)

        # actual step in real world
        observation = self.server_step(COMMANDS.STEP, action)
        # Update internal state if needed
        # (for instance n steps at targets, that should be decoupled from compute reward)
        self._on_step()
        done = self.is_terminal_state()
        if self.tracking_enabled:
            reward = self.compute_reward(scaled_action, done)
        else:
            reward = 0.0

        info = dict(
            current_state=self.current_state,
            world_position=self.world_position,
        )

        info.update(self._additional_infos())

        if self.is_real_robot:
            # check if robot has left the boundaries or it lost tracking, should only happen when it moved too far
            if self.reset_needed():
                info["TimeLimit.truncated"] = not done and self.out_of_bounds()
                done = True
        if done and self.termination_reason != "":
            print(f"Termination reason: {self.termination_reason}")

        return observation, reward, done, info

    @abc.abstractmethod
    def is_terminal_state(self) -> bool:
        """
        :return: True if the robot is in a terminal state (episode should end)
        """
        pass

    @abc.abstractmethod
    def compute_reward(self, scaled_action, done: bool) -> float:
        """
        this function should compute the reward in each time step
        :param scaled_action: the action from the agent
        :param done: true if it is the last step
        :return: reward
        """
        pass

    def _energy_consumption_cost(self) -> float:
        """
        See https://ashish-kmr.github.io/rma-legged-robots/.
        Approximating energy consumed.

        Option 1: (discard negative power)
        https://github.com/duburcqa/jiminy/blob/master/python/gym_jiminy/
        common/gym_jiminy/common/envs/env_locomotion.py#L394

        Option 2:
        https://github.com/robotlearn/pyrobolearn/blob/master/
        pyrobolearn/rewards/joint_cost.py#L463

        """
        torques = np.array(self.data["joint_torques"])
        # Revert hack
        # if self.is_real_robot:
        #     torques[1::2] = -torques[1::2]

        # Approximate motor velocity
        if "motor_velocities" not in self.data:
            motor_velocities = np.array(self.data["motor_positions"]) - self.last_motor_pos
            motor_velocities /= self.wanted_dt
        else:
            motor_velocities = self.data["motor_velocities"]
        # Option 1
        power_consumption = np.sum(np.maximum(torques * motor_velocities, 0.0))

        # Option 2
        # np.abs(np.dot(torques, motor_velocities)) * self.dt

        # torque is ~1 N.m and motor velocity is in [0, 10] rad/s
        # TODO(antonin): check normalization
        target_max_vel = 1.0  # rad/s
        target_max_torque = 1.0  # N.m
        # action_dimensions is num_legs * num_joints_per_leg
        normalization_factor = target_max_vel * target_max_torque * self.action_dimensions
        return power_consumption / normalization_factor

    def can_fit_box(self) -> bool:
        return self.robot.can_fit_box(self.box_limits, self.tracking_limits)

    def out_of_bounds(self) -> bool:
        """
        :return: True if the robot is not inside the tracking_limits
        """
        return out_of_bounds(self.robot_position, self.tracking_limits)

    def is_crawling(self) -> bool:
        """
        :return True if the robot is too low
        """
        return bool(self.robot_position.z < self.crawling_height_limit)

    def has_fallen(self) -> bool:
        """
        :return True if the robot has fallen
        """
        return bool(
            math.fabs(self.current_rot[0]) > self.roll_over_limit or math.fabs(self.current_rot[1]) > self.roll_over_limit
        )

    def try_standing_up(self) -> bool:
        """
        If the conditions are right, the robot tries to regain an upright position
        :return: True if the recovery was successful
        """
        if (
            math.fabs(self.current_rot[0]) > self.stand_up_roll_limit
            or math.fabs(self.current_rot[1]) > self.stand_up_roll_limit
        ):
            return False  # don't attempt a auto recovery after a certain angle

        self.server_step(COMMANDS.STEP, RESET_MOTOR_PARAM)
        time.sleep(0.5)
        self.server_step(COMMANDS.STEP, RESET_MOTOR_PARAM)  # make another server_step to update telemetry
        # return true if recovery was successful
        return not self.has_fallen() and not self.is_crawling()

    def treadmill_recovery_available(self) -> bool:
        """
        check if the robots position allows for an automatic treadmill reset
        """
        if not TREADMILL_ENABLED:
            return False

        # can not recover if out of left or right bound
        if self.robot_position.y < self.tracking_limits.y_min or self.robot_position.y > self.tracking_limits.y_max:
            return False
        # if the robot is already at the end, nothing can be done (will be checked by the ln_servers anyway)
        if self.robot_position.x < TREADMILL_MIN_POS:
            return False

        # safety checks done, treadmill can be used
        return True

    def reset_needed(self, first_step: bool = False) -> bool:
        """
        Check if a reset is needed for the real robot.

        :param first_step: true, if this is the first step of an episode
            We check if there is enough space in front of the robot.
        :return: Whether a reset is needed or not.
        """
        if not self.tracking_enabled:
            return False

        # This should not be called at every steps
        if first_step and not self.can_fit_box():
            self.termination_reason = "first_step"
            return True

        if self.out_of_bounds() or self.lost_tracking or self.last_obs is None or self.is_terminal_state():  # first reset
            if self.out_of_bounds():
                self.termination_reason = "out_of_bounds"
            elif self.lost_tracking:
                self.termination_reason = "lost_tracking"

            if self.last_obs is None:
                self.termination_reason = "self.last_obs is None"

            return True
        return False

    def print_reset_debug_info(self) -> None:
        print(f"Has Fallen: {self.has_fallen()}")
        print(f"Can fit box: {self.can_fit_box()}")
        print(f"Out of bounds: {self.out_of_bounds()}")
        print(f"Lost tracking: {self.lost_tracking}")
        print(f"Is Crawling: {self.is_crawling()}")

    def _angular_velocity_cost(self) -> float:
        """
        Cost for rotational velocities around all axis
        :return: squared normalized mean velocity
        """
        mean_angular_velocity = np.mean(np.abs(self.ang_vel) / self.max_ang_vel)  # normalize
        return mean_angular_velocity**2

    def _continuity_cost(self, action: np.ndarray) -> float:
        """
        Computes the continuity cost (how far the new actions is from the last)
        :param action: Scaled action in [-1,1]
        """
        if self.last_action is not None:
            max_delta = 2.0  # for the action space: high - low = 1 - (-1) = 2
            continuity_cost = np.mean((action - self.last_action) ** 2 / max_delta**2)
            self.last_action = action.copy()
        else:
            continuity_cost = 0.0
            self.last_action = action.copy()
        return continuity_cost

    @staticmethod
    def linear_scale(value: np.ndarray, min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
        """
        Rescale the value from [min_value, max_value] to [-1, 1]
        """
        return 2.0 * ((value - min_value) / (max_value - min_value)) - 1.0

    @staticmethod
    def unscale(scaled_value: np.ndarray, min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
        """
        Unscale the value from [-1, 1] to [min_value, max_value]
        """
        return min_value + (0.5 * (scaled_value + 1.0) * (max_value - min_value))

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        """
        return self.linear_scale(action, self.action_lower_limits, self.action_upper_limits)

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return self.unscale(scaled_action, self.action_lower_limits, self.action_upper_limits)

    def render(self, mode="human", close=False) -> None:
        pass

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
