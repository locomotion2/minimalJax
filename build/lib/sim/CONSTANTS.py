# Physical
g = -9.81

# Simulation
MIN_TIMESTEP = 0.001
FINAL_TIME = 2
ACTUAL_TIMESTEP = 0.05
ACTUAL_FINAL_TIME = 5
MAX_TORQUE = 5

# Learning
INPUT_SIZE = 7
OUTPUT_SIZE = 2
MAX_ENERGY = 20
MAX_SPEED = 10
ACTION_SCALE = [5, 2]
MIN_SCORE = 60
GROWTH_RATE = 0.05
MAX_GROWTH = 1

# Miscellaneous
FIG_SIZE = (18, 9)
FIG_COORDS = [2, 3]
VIZ_RATE = 2
LINE_DIST = 1000


# Help functions
def default(dictionary, key, canon):
    try:
        if dictionary[key] is None:
            return canon
        else:
            return dictionary[key]
    except KeyError:
        return canon
