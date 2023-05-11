from sim.CONSTANTS import *

import numpy as np


def default_func(state: dict):
    # Get inputs of function
    pos_model = state['Joint_pos']
    pos_gen = state['Pos_gen']
    vel_model = state['Joint_vel']
    vel_gen = state['Vel_gen']
    tau = state['Torque']
    dq_model = state['Joint_vel']
    E_d = state['Energy_des']
    E_k, E_p = state['Energies']

    # Best params thusfar
    # weights_gaussians = np.asarray([0.2, 0.25, 0.3])
    # weights_problem = np.asarray([0.7, 0.1, 0.2])

    # Define params
    weights_gaussians = np.asarray([0.2, 0.3, 0.3])
    weights_problem = np.asarray([0.7, 0.15, 0.15])
    weights_final = np.asarray([1.0, 0.0])

    # Energy rewards
    E_t = E_k + E_p
    cost_E_t = gaus(E_t - E_d, weights_gaussians[0])

    # Force punishment
    power_model = tau * dq_model
    cost_torque = gaus(np.linalg.norm(tau), weights_gaussians[1])

    # Joint positions reward
    dist_pos = np.linalg.norm(pos_model - pos_gen)
    dist_vel = np.linalg.norm(vel_model - vel_gen)
    cost_pos = gaus(dist_pos, weights_gaussians[2])
    cost_vel = gaus(dist_vel, weights_gaussians[2])
    cost_joints = (cost_pos + cost_vel) / 2

    # Problem critical rewards
    reward_problem = cost_E_t ** weights_problem[0] * cost_torque ** weights_problem[1] * cost_joints ** weights_problem[2]
    # reward_problem = cost_E_t ** weights_problem[0] * cost_torque ** weights_problem[1]

    # Desired optional rewards
    costs_desired = np.asarray([cost_pos])
    reward_desired = costs_desired @ np.asarray([1])

    # Total step costs
    rewards = np.asarray([reward_problem, reward_desired])
    reward_step = rewards @ weights_final
    reward_step = reward_problem
    costs_all = np.asarray([cost_E_t, cost_torque, cost_pos])

    return reward_step, costs_all
