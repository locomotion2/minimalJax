from identification.systems import dpendulum_utils

from identification.hyperparams import settings

if __name__ == "__main__":
    num_points = 50
    num_points_start = 0

    dpendulum_utils.generate_random_data(settings,
                                         num_points,
                                         num_points_start)
