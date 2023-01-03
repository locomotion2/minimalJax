import os
from enum import Enum

import numpy as np

ROBOT_NUM = int(os.environ.get("ROBOT_NUM", 21))
MAX_VELOCITY = 4.0  # Radians per second - set to 2 for RL
# RESET_MOTOR_PARAM = np.array([-0.38, 0.38, -0.38, 0.38, -0.38, 0.38, -0.38, 0.38])

# All legs bended backward
# RESET_MOTOR_PARAM = 0.3 * np.array([1, -1, 1, -1, 1, -1, 1, -1])
# All legs bended forward
RESET_MOTOR_PARAM = 0.3 * np.array([-1, 1, -1, 1, -1, 1, -1, 1])
# Standard pose - bended towards the center (X)
# RESET_MOTOR_PARAM = 0.3 * np.array([1, -1, 1, -1, -1, 1, -1, 1])
# Opposite
# RESET_MOTOR_PARAM = 0.3 * np.array([-1, 1, -1, 1, 1, -1, 1, -1])


TRACKING_LIMITS = {"x_min": -0.15, "x_max": 2.1, "y_min": -0.5, "y_max": 0.5}
TREADMILL_MIN_POS = TRACKING_LIMITS["x_min"] + 1
TREADMILL_ENABLED = bool(eval(os.environ.get("TREADMILL_ENABLED", "False")))
TRACKING_DISABLED = bool(eval(os.environ.get("TRACKING_DISABLED", "False")))


class COMMANDS(Enum):
    """
    Commands for communicating with the LN server.
    """

    RESET = 0
    STEP = 1
    EXIT = 2


class RESET_TYPE(Enum):
    MANUAL = 0
    TREADMILL = 1
