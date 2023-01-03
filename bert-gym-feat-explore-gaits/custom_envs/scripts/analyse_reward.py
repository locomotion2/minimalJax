import pickle
from collections import defaultdict

import numpy as np
import scipy.stats

filenames = [
    "logs/reward_hist_cpg_rl_repeat_2_60Hz_2.pkl",
    "logs/reward_hist_cpg_optimized_repeat_2_60Hz.pkl",
    "logs/reward_hist_cpg_daniel_repeat_2_60Hz.pkl",
]


def iqm(values: np.ndarray):
    """Computes the interquartile mean (IQM) for a serie of values.

    :param values: A 1D array.
    :return: IQM (25% trimmed mean) of scores.
    """
    return scipy.stats.trim_mean(values, proportiontocut=0.25)


def print_mean_std(values: np.ndarray, name: str):
    std_error = values.std() / (np.sqrt(len(values)))
    # print(f"{name} mean = {values.mean():.4f} +/- {values.std():.4f}")
    print(f"{name} mean = {values.mean():.4f} +/- {std_error:.4f}")
    # IQM and std error
    # print(f"{name} iqm = {iqm(values.flatten()):.4f} +/- {std_error:.4f}")


def compute_reward(
    vertical_speed: np.ndarray,
    drift_cost: np.ndarray,
    angular_vel_roll: np.ndarray,
    angular_vel_yaw: np.ndarray,
) -> float:
    v_speed_norm = 0.1  # 10 cm / s
    vertical_speed = vertical_speed / v_speed_norm

    max_center_deviation = 0.025
    center_deviation_cost = drift_cost / max_center_deviation

    max_ang_vel = np.deg2rad(35)
    mean_angular_velocity = (np.abs(angular_vel_roll) + np.abs(angular_vel_yaw)) / 2 * max_ang_vel
    angular_velocity_cost = mean_angular_velocity**2

    reward = (1 - angular_velocity_cost) + (0.2 - 5 * center_deviation_cost) + vertical_speed

    # if len(reward) < 300:
    #     reward[-1] = 0

    return reward.sum()


for filename in filenames:
    with open(filename, "rb") as f:
        reward_hist = pickle.load(f)

    n_episodes = len(reward_hist["drift_cost"])

    print("=" * 25)
    print(filename)

    totals = defaultdict(list)
    for episode in range(n_episodes):
        steps = len(reward_hist["drift_cost"][episode])
        # Skip unfinished trials
        if steps < 250:
            continue

        reward = compute_reward(
            np.array(reward_hist["vertical_speed"][episode]),
            np.array(reward_hist["drift_cost"][episode]),
            np.array(reward_hist["angular_vel_roll"][episode]),
            np.array(reward_hist["angular_vel_yaw"][episode]),
        )
        print(f"Episode {episode} - {steps} steps - Reward={reward:.2f}")

        totals["episode_reward"].append(reward)
        totals["episode_steps"].append(steps)
        for key in reward_hist.keys():
            if "roll" in key or "yaw" in key:
                continue
            # print(key, np.sum(reward_hist[key][episode]), np.mean(reward_hist[key][episode]))
            totals[key].append(np.sum(reward_hist[key][episode]))

    print()
    for key in totals.keys():
        # print(key, f"sum={np.sum(totals[key]):.2f} - mean={np.mean(totals[key]):.2f}")
        print_mean_std(np.array(totals[key]), key)
        if "steps" in key:
            print()

    print("=" * 25)
