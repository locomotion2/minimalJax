import os
import select
import sys
import time
from typing import Dict, Union

import links_and_nodes as ln  # pytype: disable=import-error
import numpy as np
from bert_utils.controllers import BertController, ControllerF710, TreadmillController
from scipy.spatial.transform import Rotation as R

from custom_envs.constants import (
    COMMANDS,
    MAX_VELOCITY,
    RESET_MOTOR_PARAM,
    RESET_TYPE,
    ROBOT_NUM,
    TRACKING_DISABLED,
    TREADMILL_ENABLED,
    TREADMILL_MIN_POS,
)


class LNRealServer:
    """
    Class to connect to LN with real greybert robot.

    :param ln_manager:
    """

    def __init__(self, ln_manager: str = os.environ.get("LN_MANAGER", f"localhost:541{ROBOT_NUM}")):  # noqa: B008
        super().__init__()
        self.ln_manager = ln_manager
        self.bert_telemetry_port = None
        self.bert_control_port = None
        self.tracking_enabled = not TRACKING_DISABLED

        self.gamepad_controller = None
        self.bert_controller = None

        self.treadmill_controller = None
        self.treadmill_enabled = TREADMILL_ENABLED

    def init(self) -> None:
        print("Connecting to LN Manager")
        print(self.ln_manager)
        self.ln_client = ln.client("ln_bert_rl", self.ln_manager)
        print("Connected to manager")

        # setup gamepad controller
        print("Starting Gamepad Controller")
        self.gamepad_controller = ControllerF710(self.ln_client)

        # setup bert controller
        print("Starting Bert Controller")
        self.bert_controller = BertController(self.ln_client, tracking_enabled=self.tracking_enabled)

        # setup treadmill controller
        if self.treadmill_enabled:
            assert self.tracking_enabled, "Tracking must be enabled with treadmill"
            print("Starting Treadmill Controller")
            self.treadmill_controller = TreadmillController(self.ln_client)
            self.treadmill_controller.enable()
            self.treadmill_controller.set_speed(0.0)

        # Debug topic
        # Add message definition if needed
        # msg_def_path = os.path.join(os.path.dirname(__file__), "../ln_topics/rl_debug")
        # with open(msg_def_path, "r") as f:
        #     raw_string = f.read()
        # self.ln_client.put_message_definition("rl_debug", raw_string)
        # self.rl_debug = self.ln_client.publish("rl.debug", "rl_debug")

        # enable bert: enable motors and limit max velocity
        self.bert_controller.setup(MAX_VELOCITY)
        print("Setup done, robot ready, enjoy!")

    def step(self, command: COMMANDS, action: Union[RESET_TYPE, np.ndarray, None]) -> Dict[str, np.ndarray]:

        # Publish on the debug topic
        # self.publish_debug(msg.get("debug"))

        if command == COMMANDS.RESET:
            assert isinstance(action, RESET_TYPE)
            reset_type = action
            # Ask the user for a manual reset
            if reset_type == RESET_TYPE.MANUAL:
                self.manual_reset()
            else:
                self.treadmill_reset()
        elif command == COMMANDS.STEP:
            assert action is not None
            # set desired motor positions
            self.bert_controller.set_motor_positions(np.array(action))
        elif command == COMMANDS.EXIT:
            pass
        else:
            raise ValueError(f"Unknown command: {command}")

        telemetry_data = self.bert_controller.get_telemetry()
        if self.tracking_enabled:
            tracking_data = self.bert_controller.get_camera_tracking()

        # Retrieve only 8 values
        # instead of the 12d vector (q1, q2, q3=q2 - q1)
        # TODO(antonin): double check
        correct_indices = np.ravel([[i, i + 1] for i in range(0, 12, 3)])

        joint_positions = telemetry_data.q.copy()[correct_indices]
        joint_velocities = telemetry_data.dq.copy()[correct_indices]
        # The last 3 torques are zeros
        joint_torques = telemetry_data.tau.copy()[:8]
        # Hack to match sim
        # TODO(antonin): investigate and match sim with real robot?
        joint_torques[1::2] = -joint_torques[1::2]

        motor_positions = telemetry_data.theta.copy()
        motor_velocities = telemetry_data.dtheta.copy()

        # Angular velocity
        ang_vel = telemetry_data.dx[3:].copy()
        # Linear acceleration
        lin_acc = telemetry_data.ddx[:3].copy()
        imu_orientation = telemetry_data.x[3:].copy()
        # 3D Position
        if self.tracking_enabled:
            # position from tracking system
            position_3d = tracking_data.loc.copy()
            # convert from matrix to euler
            rot_mat = tracking_data.rot.copy()
            rotation_3d = R.from_matrix(rot_mat.reshape((3, 3))).as_euler("xyz", degrees=False)

            tracking_active = tracking_data.quality > 0
        else:
            tracking_active = False
            rotation_3d = None
            position_3d = None

        current_bridge = telemetry_data.current_bridge.copy()
        voltage_bridge = telemetry_data.voltage_bridge.copy()

        new_infos = dict(
            joint_torques=joint_torques,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            motor_positions=motor_positions,
            motor_velocities=motor_velocities,
            ang_vel=ang_vel,
            lin_acc=lin_acc,
            position_3d=position_3d,
            rotation_3d=rotation_3d,
            imu_orientation=imu_orientation,
            tracking_active=tracking_active,
            current_bridge=current_bridge,
            voltage_bridge=voltage_bridge,
        )
        return new_infos  # pytype: disable=bad-return-type

    def close(self) -> None:
        pass

    def manual_reset(self) -> None:
        print("======================= Reset Requested =======================")
        print("Please lift to reset legs")
        self.wait_user_input()
        self.bert_controller.set_motor_positions(RESET_MOTOR_PARAM)
        time.sleep(0.2)

    @staticmethod
    def input_available():
        # TODO(antonin): add return type (apparently a list)
        """
        checks if stdin is ready for read (if data is available)
        """
        return select.select([sys.stdin.fileno()], [], [], 0.0)[0]

    def wait_user_input(self) -> None:
        # empty stdin
        while self.input_available():
            os.read(sys.stdin.fileno(), 4096)  # read all the input from stdin, 4096 is chosen at random

        print("Hit Enter, or controller A to continue")

        self.gamepad_controller.read()
        while True:
            if self.input_available():
                print("Key pressed!\n")
                break

            gamepad = self.gamepad_controller.read()

            if not gamepad:
                continue

            if gamepad.button_A == 1:
                print("Button A pressed!\n")
                break

            time.sleep(0.2)

    def treadmill_reset(self) -> None:
        print("======================= treadmill reset requested =======================")
        if not self.treadmill_enabled:
            print("Treadmill reset not enabled in config!")
            return
        tracking_data = self.bert_controller.get_camera_tracking()
        position_3d = tracking_data.loc.copy()

        if position_3d[0] < TREADMILL_MIN_POS:
            print("Treadmill reset requested but position too small: x = ", position_3d[0])
            return

        # start treadmill
        self.treadmill_controller.set_speed(2.0)
        while position_3d[0] > TREADMILL_MIN_POS:  # 1m safety distance for stop
            tracking_data = self.bert_controller.get_camera_tracking(blocking=True)
            # if tracking_data is None:
            #     print("Tracking lost! (no data received)")
            #     break
            position_3d = tracking_data.loc.copy()
            if tracking_data.quality < 1:
                print("Tracking lost! (bad tracking quality)")
                break
            time.sleep(0.01)

        # stop treadmill
        self.treadmill_controller.set_speed(0.0)
        print("Treadmill stopped")
        time.sleep(2)
