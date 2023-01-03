import os
import time
from typing import Dict, Optional

import links_and_nodes as ln  # pytype: disable=import-error
import numpy as np

from custom_envs.constants import COMMANDS, RESET_MOTOR_PARAM


class LNSimServer:
    """
    Class to connect to LN with greybert simulation.

    :param ln_manager:
    """

    def __init__(self, ln_manager: str = os.environ.get("LN_MANAGER", "localhost:54133")):  # noqa: B008
        super().__init__()
        self.ln_manager = ln_manager
        self.bert_move_port = None
        self.bert_telemetry_port = None
        self.bert_control_port = None

    def init(self) -> None:
        print("Connecting to LN Manager")
        print(self.ln_manager)
        self.ln_client = ln.client("ln_bert_rl", self.ln_manager)
        print("Connected to manager")
        self.bert_telemetry_port = self.ln_client.subscribe(topic_name="bert_telemetry")  # subscribe to telemetry
        self.bert_control_port = self.ln_client.publish(
            "bert_motor", message_definition_name="greybert/motor_packet", buffers=1
        )
        self.bert_move_port = self.ln_client.publish("bert_move", message_definition_name="move_bert_packet", buffers=1)

    def step(self, command: COMMANDS, action: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:

        if command == COMMANDS.RESET:
            self.reset()
        elif command == COMMANDS.STEP:
            assert action is not None
            self.send_command(np.array(action))
        elif command == COMMANDS.EXIT:
            pass
        else:
            raise ValueError(f"Unknown command: {command}")

        topic_data = self.bert_telemetry_port.read(blocking_or_timeout=True)
        joint_torques = topic_data.joint_torque.copy()
        joint_positions = topic_data.joint_position.copy()
        joint_velocities = topic_data.joint_velocity.copy()
        motor_positions = topic_data.motor_position.copy()
        # Angular velocity
        ang_vel = topic_data.imu_ang_vel.copy()
        # Linear acceleration
        lin_acc = topic_data.imu_lin_acc.copy()
        # 3D Position
        pose = topic_data.pose.copy()
        position_3d = pose[:3]
        rotation_3d = pose[3:]
        new_infos = dict(
            joint_torques=joint_torques,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            motor_positions=motor_positions,
            ang_vel=ang_vel,
            lin_acc=lin_acc,
            position_3d=position_3d,
            rotation_3d=rotation_3d,
        )
        return new_infos

    def send_command(self, motor_position: np.ndarray) -> None:
        """
        :param motor_position: desired positions for the motors
        """
        self.bert_control_port.packet.motor_position = motor_position
        self.bert_control_port.write()

    def reset(self) -> None:
        # reset motor position
        self.bert_control_port.packet.motor_position = np.zeros(8)
        self.bert_control_port.write()
        time.sleep(0.25)
        # move robot back to start
        self.bert_move_port.packet.position = np.zeros(2)
        self.bert_move_port.packet.heading = 0
        self.bert_move_port.write()
        time.sleep(0.5)
        # Initial pose
        self.bert_control_port.packet.motor_position = RESET_MOTOR_PARAM
        self.bert_control_port.write()
        time.sleep(0.25)

    def close(self) -> None:
        pass
