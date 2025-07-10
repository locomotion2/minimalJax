import src.discovery_utils as sutils
import jax.numpy as jnp

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

    # Define params
    weights_gaussians = jnp.asarray([0.2, 0.3, 0.3])
    weights_problem = jnp.asarray([0.7, 0.15, 0.15])

    # Energy rewards
    E_t = E_k + E_p
    # Take the norm of the energy difference to get a scalar
    cost_E_t = sutils.gaus(jnp.linalg.norm(E_t - E_d), weights_gaussians[0])

    # Force punishment
    cost_torque = sutils.gaus(jnp.linalg.norm(tau), weights_gaussians[1])

    # Joint positions reward
    dist_pos = jnp.linalg.norm(pos_model - pos_gen)
    dist_vel = jnp.linalg.norm(vel_model - vel_gen)
    cost_pos = sutils.gaus(dist_pos, weights_gaussians[2])
    cost_vel = sutils.gaus(dist_vel, weights_gaussians[2])
    cost_joints = (cost_pos + cost_vel) / 2

    # Problem critical rewards
    reward_problem = cost_E_t ** weights_problem[0] *\
                     cost_torque ** weights_problem[1] *\
                     cost_joints ** weights_problem[2]

    # Total step costs
    reward_step = reward_problem
    costs_all = jnp.asarray([cost_E_t, cost_torque, cost_pos])

    return reward_step, costs_all