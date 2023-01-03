import os

import links_and_nodes as ln  # pytype: disable=import-error
import numpy as np

ROBOT_NUM = 21

LN_MANAGER = os.environ.get("LN_MANAGER", f"localhost:541{ROBOT_NUM}")

ln_client = ln.client("test_client", LN_MANAGER)
parameter = ln_client.get_parameters("motor_pos_test.position")
velocity = ln_client.get_parameters("motor_pos_test.dtheta_d")
velocity.set_override("value", 2.0)

action_upper_limits = np.array([0.6, 0.5, 0.5, 0.6, 0.75, 0.5, 0.75, 0.5])
action_lower_limits = np.array([-0.75, -0.5, -0.5, -0.75, -0.6, -0.5, -0.6, -0.5])


n_motors = 8
motor_params = np.array([-0.38, 0.38, -0.38, 0.38, -0.38, 0.38, -0.38, 0.38])
parameter.set_override("value", motor_params)
print("Ready for testing")
raw = input()
motor_params = np.zeros(n_motors)
for i in range(len(action_upper_limits) // 2):
    motor_params[i * 2] = action_upper_limits[i * 2]
    motor_params[i * 2 + 1] = action_upper_limits[i * 2 + 1]
    parameter.set_override("value", motor_params)
    input()

    motor_params[i * 2 + 1] = action_lower_limits[i * 2 + 1]
    parameter.set_override("value", motor_params)
    input()
    motor_params[i * 2] = action_lower_limits[i * 2]
    parameter.set_override("value", motor_params)
    input()
    motor_params[i * 2 + 1] = action_upper_limits[i * 2 + 1]
    parameter.set_override("value", motor_params)
    input()
    motor_params[i * 2] = 0
    motor_params[i * 2 + 1] = 0
    parameter.set_override("value", motor_params)
    input("Test end")
