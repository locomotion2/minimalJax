import argparse
import os
from typing import Dict, Optional

import cpg_gamepad
import matplotlib
import numpy as np
import seaborn
import yaml
from cpg_gamepad.cpg import GAITS, HopfNetwork, skew_sim_matrix  # COUPLING_MATRICES
from matplotlib import pyplot as plt
from rl_zoo3.utils import StoreDict

# Do not use type3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# Activate seaborn
seaborn.set()
# Seaborn style
seaborn.set(style="whitegrid")


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    # Wrap between [0, 2 * pi]
    return angle - 2 * np.pi * np.floor(angle / (2 * np.pi))


def compute_gait_stats(
    omega_swing: float,
    omega_stance: float,
    coupling_matrix: Optional[np.ndarray] = None,
    print_info: bool = False,
) -> Dict[str, float]:
    swing_duration = np.pi / omega_swing
    stance_duration = np.pi / omega_stance
    stride_duration = swing_duration + stance_duration
    cycle_duration = 2 * stride_duration
    duty_factor = stance_duration / stride_duration

    if print_info:
        print(f"Swing duration: {swing_duration:.4f}s")
        print(f"Stance duration: {stance_duration:.4f}s")
        print(f"Stride duration: {stride_duration:.4f}s")
        print(f"Cycle duration: {cycle_duration:.4f}s - {cycle_duration / dt:.2f} steps")
        print(f"Duty factor: {duty_factor:.4f}")
    stats = dict(
        swing_duration=swing_duration,
        stance_duration=stance_duration,
        stride_duration=stride_duration,
        cycle_duration=cycle_duration,
        duty_factor=duty_factor,
    )

    if coupling_matrix is not None:
        # normalize the coupling matrix
        coupling_matrix = wrap_angle(coupling_matrix)

        legs = ["front_right", "front_left", "rear_right", "rear_left"]
        legs = {name: idx for idx, name in enumerate(legs)}
        # See https://www.sciencedirect.com/science/article/pii/S1631069103001707
        # A new way of analysing symmetrical and asymmetrical gaits in quadrupeds
        f_lag = coupling_matrix[legs["front_left"], legs["front_right"]]
        h_lag = coupling_matrix[legs["rear_left"], legs["rear_right"]]
        p_lag = coupling_matrix[legs["front_right"], legs["rear_right"]]
        phase_stats = dict(f_lag=f_lag, h_lag=h_lag, p_lag=p_lag)
        for key, value in phase_stats.items():
            # TODO: clarify when lag=1 (inconsistent in the paper)
            new_value = (abs(value) / (2 * np.pi)) % 1.0
            phase_stats[key] = new_value
        phase_stats = {key: value for key, value in phase_stats.items()}

        stats.update(phase_stats)

    if coupling_matrix is not None and print_info:
        for key, value in phase_stats.items():
            print(f"{key}: {value:.4f}")

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gait", help="Desired gait", type=str, default="trot", choices=[gait.value for gait in GAITS])
    parser.add_argument("-t", "--max-time", help="Timeout (in s)", type=float, default=4)
    parser.add_argument("-plot", "--plot", action="store_true", default=False, help="Plot desired traj")
    parser.add_argument("-latex", "--latex", help="Enable latex support", action="store_true", default=False)
    parser.add_argument(
        "-gamepad", "--gamepad-params", help="User parameters from cpg_gamepad package", action="store_true", default=False
    )
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=float, default=[7, 3.5])
    parser.add_argument("--fontsize", help="Font size", type=int, default=16)
    parser.add_argument("-shift", "--leg-phase-shifts", type=str, nargs="+", action=StoreDict)
    parser.add_argument("-param", "--parameters", type=str, nargs="+", action=StoreDict)
    args = parser.parse_args()

    # Enable LaTeX support
    if args.latex:
        plt.rc("text", usetex=True)

    # Control frequency
    dt = 1 / 100
    max_time = args.max_time  # s
    max_steps = int(max_time / dt)
    num_legs = 4

    gait = GAITS(args.gait)

    # load config from cpg_gamepad
    if args.gamepad_params:
        base_path = os.path.dirname(cpg_gamepad.__file__)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    # Load config file
    with open(os.path.join(base_path, "config.yml")) as f:
        parameters = yaml.safe_load(f)[gait.value]
        if args.parameters is not None:
            parameters.update(args.parameters)
        omega_swing, omega_stance = np.pi * parameters["omega_swing"], np.pi * parameters["omega_stance"]

    phi_init = None
    # if gait in [GAITS.TROT, GAITS.FAST_TROT]:
    #     # start in stance for trot (coupling slightly different than the normal one,
    #     # will converge quickly)
    #     phi_init = 3 * COUPLING_MATRICES[gait][0, :] / 4 - 0.5 * np.pi / 3

    cpg = HopfNetwork(
        gait=gait,
        omega_swing=omega_swing,
        omega_stance=omega_stance,
        time_step=dt,
        mu=1,  # converge to sqrt(mu)
        coupling_strength=1,  # coefficient to multiply coupling matrix
        couple=True,  # should couple
        ground_clearance=parameters["ground_clearance"],  # foot swing height
        ground_penetration=parameters["ground_penetration"],  # foot stance penetration into ground
        robot_height=0.1,  # in nominal case (standing)
        desired_step_len=parameters["desired_step_len"],  # desired step length
        #  initial phase
        phi_init=phi_init,
    )

    if args.leg_phase_shifts is not None:
        cpg.coupling_matrix = 2 * np.pi * skew_sim_matrix(**args.leg_phase_shifts)
        # Update initial phase
        cpg.X[1, :] = (cpg.coupling_matrix[0, :] + np.pi / 2) % (2 * np.pi)
        # cpg.X[1, :] = cpg.coupling_matrix[0, :]

    # TODO: check the units, there might be a factor 2 missing
    stats = compute_gait_stats(omega_swing, omega_stance, cpg.coupling_matrix, print_info=True)

    in_stance = np.zeros((max_steps, num_legs))
    thetas = np.zeros((max_steps, num_legs))
    # For additional plots
    if args.plot:
        xs_list = np.zeros((max_steps, num_legs))
        zs_list = np.zeros((max_steps, num_legs))

    try:
        for step in range(max_steps):
            # get desired foot positions from CPG
            desired_x, desired_z = cpg.update()

            thetas[step, :] = cpg.theta
            in_stance[step, :] = np.sin(cpg.theta) < 0

            if args.plot:
                xs_list[step] = desired_x
                zs_list[step] = desired_z

    except KeyboardInterrupt:
        pass

    legs = np.array(["Front Right", "Front Left", "Rear Right", "Rear Left"])

    # plt.figure("Thetas")
    # for leg_idx, label in enumerate(legs):
    #     plt.plot(np.sin(thetas[:, leg_idx]), label=label)
    # plt.legend()

    legs_in_stance = in_stance.sum(axis=1)
    print(f"Mean number of legs in stance: {legs_in_stance.mean():.2f} +/- {legs_in_stance.std():.2f}")
    print(f"Min/Max number of legs in stance: {legs_in_stance.min()} / {legs_in_stance.max()}")
    print(f"Median number of legs in stance: {np.median(legs_in_stance)}")

    plt.figure("Num legs in stance", figsize=args.figsize)
    plt.plot(np.arange(max_steps) * dt, legs_in_stance)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylim(-0.2, 4.2)
    plt.tight_layout()

    plt.figure("Pattern", figsize=args.figsize)
    # plt.title("Walking Trot (Optimized)", fontsize=args.fontsize)
    plt.title(f"Pattern for {gait}", fontsize=args.fontsize)
    # plt.title("Pronking (Optimized)", fontsize=args.fontsize)

    for leg_idx in reversed(range(num_legs)):
        # Switch between swing and stance phases
        changes_indices = np.concatenate(
            (
                [0],
                np.where(in_stance[:-1, leg_idx] != in_stance[1:, leg_idx])[0] + 1,
                [max_steps],
            ),
            axis=0,
        )

        # Convert to bar chart
        y = changes_indices[1:] - changes_indices[:-1]

        color = {
            "Front Right": "#4683E0",
            "Front Left": "#104494",
            "Rear Right": "#F7BA40",
            "Rear Left": "#FFCD69",
        }[legs[leg_idx]]

        previous = 0
        # Duration of each phase
        # print(legs[leg_idx])
        # print(y *  dt)
        for stance_length in y:
            alpha = 1.0 if in_stance[previous, leg_idx] == 1 else 0.15
            plt.barh(legs[leg_idx], stance_length * dt, left=previous * dt, color=color, alpha=alpha, linewidth=0)
            previous += stance_length

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Time (s)", fontsize=14)
    plt.tight_layout()
    plt.show()

    if args.plot:
        # Limit data
        k = 1000
        plt.figure("Foot Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.plot(xs_list[:, 0], zs_list[:, 0], label="desired")
        plt.legend()
        plt.show()
