import numpy as np

# Physical
g = -9.81

# Simulation
MIN_TIMESTEP = 0.05
FINAL_TIME = 5
ACTUAL_TIMESTEP = 0.05
MAX_TORQUE = 5

# Learning
INPUT_SIZE = 9
OUTPUT_SIZE = 2
MAX_ENERGY = 20
MAX_SPEED = 40
# ACTION_SCALE = [6, 2*np.pi]  # Todo: search for a correct scale
ACTION_SCALE = []

# Curriculum
MIN_SCORE = 60
GROWTH_RATE = 0.05
MAX_GROWTH = 1

# Miscellaneous
FIG_SIZE = (19, 10)
FIG_COORDS = [2, 4]
VIZ_RATE = 2
LINE_DIST = 1000


# Help functions
def gaus(value: float, width: float = 0.3):
    return float(np.exp(-(value / width) ** 2))


def debug_print(text: str = None, obj=None):
    if True:
        print(f'{text}: {obj}')