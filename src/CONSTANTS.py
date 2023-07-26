import numpy as np

# TODO: Change the contents here into dictionaries and not import stars but settings

# Physical
g = -9.81

# Simulation
MIN_TIMESTEP = 0.05
FINAL_TIME = 5
ACTUAL_TIMESTEP = 0.05
MAX_TORQUE = 5

# Learning
INPUT_SIZE_PEND = 5
INPUT_SIZE_DPEND = 9
OUTPUT_SIZE_CPG = 2
OUTPUT_SIZE_DIRECT = 4
MAX_ENERGY = 20
MAX_SPEED = 40
ACTION_SCALE_CPG = [MAX_SPEED/6, 2*np.pi]  # Todo: search for a correct scale
ACTION_SCALE_DIRECT = [np.pi, MAX_SPEED/6]

# Curriculum
MIN_SCORE = 60
GROWTH_RATE = 0.05
MAX_GROWTH = 1

# Miscellaneous
FIG_SIZE = (19, 13)
FIG_COORDS = [2, 4]
VIZ_RATE = 2
LINE_DIST = 1000