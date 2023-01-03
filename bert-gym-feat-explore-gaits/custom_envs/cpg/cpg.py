"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import argparse
import os
import time

import links_and_nodes as ln
import numpy as np
import yaml
from bert_utils import inverse_polar, joint_angles_to_polar
from bert_utils.controllers import BertController
from bert_utils.rate_limiter import RateLimiter
from cpg_gamepad.cpg import GAITS, HopfNetwork
from matplotlib import pyplot as plt

from custom_envs.constants import COMMANDS, MAX_VELOCITY, RESET_MOTOR_PARAM, TRACKING_DISABLED
from custom_envs.ln_servers.ln_grey_sim import LNSimServer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gait", help="Desired gait", type=str, default="trot", choices=[gait.value for gait in GAITS])
    parser.add_argument("--max-time", help="Timeout (in s)", type=int, default=4)
    parser.add_argument("-real", "--real-robot", action="store_true", default=False, help="Using real robot")
    parser.add_argument("-sim", "--gazebo-sim", action="store_true", default=False, help="Using Gazebo sim")
    parser.add_argument("-plot", "--plot", action="store_true", default=False, help="Plot desired traj")
    parser.add_argument("-b", "--backward", action="store_true", default=False, help="Go backward")
    parser.add_argument("-e", "--compute-energy", action="store_true", default=False, help="Compute cost of transport")
    args = parser.parse_args()

    # Control frequency
    dt = 1 / 1000
    max_time = args.max_time  # s
    max_steps = int(max_time / dt)
    num_legs = 4
    # leg_length = 0.08
    # max_height = 2 * leg_length
    polar_coords = joint_angles_to_polar(RESET_MOTOR_PARAM)
    # Take 90% of radius at rest, so leg are not too stretch
    robot_height = 0.9 * polar_coords[1]
    direction = -1 if args.backward else 1

    gait = GAITS(args.gait)
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Load config file
    with open(os.path.join(base_path, "config.yml")) as f:
        parameters = yaml.safe_load(f)[gait.value]
        omega_swing, omega_stance = np.pi * parameters["omega_swing"], np.pi * parameters["omega_stance"]

    # TODO: check the units, there might be a factor 2 missing
    d_swing = np.pi / omega_swing
    d_stance = np.pi / omega_stance
    # Note: fps is limited by LN com, those value should be rescaled by true FPS (~90)
    print(f"Swing duration: {d_swing:.4f}s")
    print(f"Stance duration: {d_stance:.4f}s")

    if args.real_robot:
        ln_client = ln.client("ln_bert_cpg", os.environ.get("LN_MANAGER", "localhost:4444"))
        bert_controller = BertController(ln_client, tracking_enabled=not TRACKING_DISABLED)
        if args.compute_energy:
            # Retrieve service
            energy_cost_service = ln_client.get_service("bert.energy_cost_service", "energy_cost_service")
            # Toggle recording
            energy_cost_service.call()

        # enable bert: enable motors and limit max velocity
        bert_controller.setup(MAX_VELOCITY)
        bert_controller.set_motor_positions(RESET_MOTOR_PARAM)
        time.sleep(0.1)

    elif args.gazebo_sim:
        sim_server = LNSimServer()
        sim_server.step(COMMANDS.RESET)

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
        robot_height=robot_height,  # in nominal case (standing)
        desired_step_len=parameters["desired_step_len"],  # desired step length
    )

    # For plotting
    if args.plot:
        radius = np.zeros((max_steps, num_legs))
        angles = np.zeros((max_steps, num_legs))

        xs_list = np.zeros((max_steps, num_legs))
        zs_list = np.zeros((max_steps, num_legs))

    start_time = time.time()
    rate_limiter = RateLimiter(wanted_dt=dt)

    try:
        for step in range(max_steps):
            # get desired foot positions from CPG
            desired_x, desired_z = cpg.update()

            # Hack convert to real robot convention (see bert modes)
            # switch hind left with hind right
            if args.real_robot:
                tmp_x, tmp_z = desired_x.copy(), desired_z.copy()
                desired_x[2], desired_z[2] = tmp_x[3], tmp_z[3]
                desired_x[3], desired_z[3] = tmp_x[2], tmp_z[2]

            # x = r * sin(theta)
            # z = - r * cos(theta)
            desired_radius = np.sqrt(desired_x**2 + desired_z**2)
            # desired_angle = np.arcsin(desired_x / r_desired)
            desired_angle = np.arctan2(desired_x, -desired_z)

            polar_coords = np.zeros((num_legs * 2,))
            polar_coords[::2] = desired_angle
            polar_coords[1::2] = desired_radius

            desired_motor_pos = inverse_polar(polar_coords, direction * RESET_MOTOR_PARAM.copy())

            if args.real_robot:
                bert_controller.set_motor_positions(direction * desired_motor_pos)

            elif args.gazebo_sim:
                sim_server.step(COMMANDS.STEP, desired_motor_pos)

            if args.plot:
                radius[step] = desired_radius
                angles[step] = desired_angle
                xs_list[step] = desired_x
                zs_list[step] = desired_z

            # Rate limiting, disabled for now
            rate_limiter.update_control_frequency(init=step == 0)
    except KeyboardInterrupt:
        pass

    end_time = time.time()
    print(f"{step / (end_time - start_time):.2f} FPS")

    if args.compute_energy:
        # Disable recording
        energy_cost_service.call()

        # Retrieve the response
        mean_cost_mecha = energy_cost_service.resp.mean_cost_mecha
        mean_cost_elec = energy_cost_service.resp.mean_cost_elec
        n_steps = energy_cost_service.resp.n_steps
        print(f"Mean cost of transport mecha = {mean_cost_mecha:.4f} - elec = {mean_cost_elec:.4f} - {n_steps} steps")

    if args.plot:
        k = 1000
        # Limit data
        radius, angles = radius[-k:, :], angles[-k:, :]
        plt.figure("Foot Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.plot(radius[:, 0] * np.sin(angles[:, 0]), -radius[:, 0] * np.cos(angles[:, 0]), label="desired")
        plt.legend()

        fig, axes = plt.subplots(2)
        fig.suptitle("Radius and angle")
        axes[0].plot(np.arange(len(radius[:, :2])), radius[:, :2], label="desired")
        axes[1].plot(np.arange(len(angles[:, :2])), np.rad2deg(angles[:, :2]), label="desired")
        plt.legend()

        # This figure should be the same as "Foot Trajectory"
        # fig = plt.figure(1, figsize=(16, 6), dpi=200, facecolor="w", edgecolor="k")
        # k = 1000
        # plt.plot(xs_list[-k:, 0], zs_list[-k:, 0], "b", label="x vs z (des)")
        # # plt.plot(x_list[-k:, 0], z_list[-k:, 0], "b--", label=r"x vs z")
        # plt.legend(loc="right")
        # plt.xlabel("x vs z")
        plt.show()
