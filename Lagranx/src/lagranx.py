from copy import deepcopy as copy
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from aim import Run
from flax import linen as nn
from flax.training import train_state as ts
from jax.experimental.ode import odeint

import sqlite3

from Lagranx.src import dpend_model_arne as model


class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        # unpacking data
        # q_state = x[:4 * 10]
        # q_state, _ = jnp.split(x, 2)
        # full_state = x
        q_state = x

        net_size = 64 * 10
        # build kinetic net
        x_kin = self.layer(q_state, features=net_size)
        x_kin = nn.Dense(features=10)(x_kin)
        x_kin = nn.softplus(x_kin + 2) + 1

        # build potential net
        x_pot = self.layer(q_state, features=net_size)
        x_pot = nn.Dense(features=1)(x_pot)

        # # build friction net
        # x_f = self.layer(q_state, features=net_size)
        # x_f = nn.Dense(features=4)(x_f)
        # x_f = nn.tanh(x_f) + 1

        # return jnp.concatenate([jnp.array([x_full[0]]), x_pot, x_f])
        return jnp.concatenate([x_kin, x_pot])

    # @nn.compact
    # def __call__(self, x):
    #     net_size = 64 * 10
    #     x = self.layer(x, features=net_size)
    #     x = nn.Dense(features=2 + 4)(x)
    #
    #     return x

    def layer(self, x: jnp.array, features: int = 128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.softplus(x)
        return x


def split_state(state, buffer_length):
    @jax.jit
    def _split_state(state):
        # q = jnp.array([state[0],
        #                state[buffer_length + 1]])
        #
        # q1_buff = state[1: buffer_length]
        # q2_buff = state[buffer_length + 1: 2 * buffer_length]
        # q_buff = jnp.concatenate([q1_buff, q2_buff])
        #
        # dq = jnp.array([state[2 * buffer_length],
        #                 state[3 * buffer_length + 1]])
        # dq1_buff = state[2 * buffer_length + 1:
        #                  3 * buffer_length]
        # dq2_buff = state[3 * buffer_length + 1:
        #                  4 * buffer_length]
        # dq_buff = jnp.concatenate([dq1_buff, dq2_buff])
        #
        # tau = state[-2:]

        q = jnp.array([state[0 * buffer_length],
                       state[1 * buffer_length],
                       state[2 * buffer_length],
                       state[3 * buffer_length]])

        dq = jnp.array([state[4 * buffer_length],
                        state[5 * buffer_length],
                        state[6 * buffer_length],
                        state[7 * buffer_length]])

        q_buff = jnp.array([])
        dq_buff = jnp.array([])
        for index in range(4):
            q_temp = extract_from_sample_split(state,
                                               buffer_length,
                                               indices=(index, index + 1))

            dq_temp = extract_from_sample_split(state,
                                                buffer_length,
                                                indices=(4 + index, 4 + index + 1))

            q_buff = jnp.concatenate([q_buff, q_temp[1:]])
            dq_buff = jnp.concatenate([dq_buff, dq_temp[1:]])

        tau = state[-4:]

        return q, q_buff, dq, dq_buff, tau

    return _split_state(state)


@jax.jit
def L_mass_func(M):
    # top_M = jnp.triu(M, k=1)
    # bot_M = jnp.tril(M, k=-1)
    diag_M = jnp.diagonal(M)
    # diag_M = jnp.real(jnp.linalg.eigvals(M))
    diag_M = jnp.clip(diag_M, a_min=None, a_max=0)
    return jnp.mean((M - jnp.transpose(M)) ** 2) + jnp.sum(diag_M ** 2)


def lagrangian_dyn_builder(state: jnp.array,
                           lagrangian: Callable = None,
                           friction: Callable = None
                           ):
    q, q_buff, dq, dq_buff, tau = split_state(state, 10)

    # obtain equation terms
    M = jax.hessian(lagrangian, 2)(q, q_buff, dq, dq_buff)
    cross_hessian = jax.jacobian(jax.jacobian(lagrangian, 2), 0)(q, q_buff, dq, dq_buff)
    N = cross_hessian @ dq - jax.grad(lagrangian, 0)(q, q_buff, dq, dq_buff)

    # bundle terms
    state_current = (q, dq, tau)
    dynamics = (M, N)
    frictions = friction(q, q_buff, dq, dq_buff)

    return state_current, dynamics, frictions


def energy_dyn_builder(state: jnp.array,
                       kinetic: Callable = None,
                       potential: Callable = None,
                       lagrangian: Callable = None,
                       friction: Callable = None
                       ):
    q, q_buff, dq, dq_buff, tau = split_state(state, 10)

    # obtain equation terms
    M = jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)
    cross_hessian = jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff)
    N = cross_hessian @ dq + (jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff) -
                              jax.grad(potential, 0)(q, q_buff, dq, dq_buff))

    # bundle terms
    state_current = (q, dq, tau)
    dynamics = (M, N)
    frictions = friction(q, q_buff, dq, dq_buff)

    return state_current, dynamics, frictions

def inertia_dyn_builder(state: jnp.array,
                       kinetic: Callable = None,
                       potential: Callable = None,
                       inertia: Callable = None,
                       friction: Callable = None
                       ):
    q, q_buff, dq, dq_buff, tau = split_state(state, 10)

    # obtain equation terms
    M = inertia(q, q_buff, dq, dq_buff)
    cross_hessian = jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff)
    # dM_q = jax.jacobian(inertia, 0)(q, q_buff, dq, dq_buff)
    # cross_hessian = jnp.transpose(dq) @ dM_q @ dq
    N = cross_hessian @ dq + (jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff) -
                              jax.grad(potential, 0)(q, q_buff, dq, dq_buff))

    # bundle terms
    state_current = (q, dq, tau)
    dynamics = (M, N)
    frictions = friction(q, q_buff, dq, dq_buff)

    return state_current, dynamics, frictions


@jax.jit
def inv_dyn_lagrangian(terms):
    # unpack terms
    # (q, dq, tau), (M, N), (k_tau, k_dq) = terms
    (q, dq, tau), (M, N), k_dq = terms

    # correct the torque
    # tau_aff = - k_tau * tau
    tau_f = - k_dq * dq
    tau_eff = tau + tau_f

    # perform inverse dynamics
    ddq = jnp.linalg.pinv(M) @ (tau_eff - N)

    return jnp.concatenate([dq, ddq])


@jax.jit
def for_dyn_lagrangian(ddq, terms):
    # unpack terms
    # ((q, dq, tau_target), (M, N), (k_tau, k_dq)) = terms
    ((q, dq, tau_target), (M, N), k_dq) = terms

    # perform inverse dynamics
    tau_eff = M @ ddq[-4:] + N

    # correct the torque
    # tau_aff = - k_tau * tau_target
    tau_f = - k_dq * dq
    tau = tau_eff - tau_f

    return tau, tau_target, tau_f


# @jax.jit
def calc_dynamics(state: jnp.array,
                  kinetic: Callable = None,
                  potential: Callable = None,
                  friction: Callable = None
                  ) -> jnp.array:
    q, q_buff, dq, dq_buff, tau = split_state(state, 10)

    # M = jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)
    # C = jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff) @ dq - \
    #     jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff)
    # g = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)
    # k_f = friction(q, q_buff, dq, dq_buff)
    @jax.jit
    def inertia(q, q_buff, dq, dq_buff):
        M = jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)
        return M

    def friction_simple(q, dq):
        return friction(q, q_buff, dq, dq_buff)

    # build dynamic functions
    def inertia_simple(q, dq):
        return inertia(q, q_buff, dq, dq_buff)

    # def coriolis(q, dq):
    #     return jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff) - \
    #         1 / 2 * jnp.transpose(dq) @ \
    #         jax.jacobian(inertia, 0)(q, q_buff, dq, dq_buff)

    @jax.jit
    def coriolis(q, dq):
        M = jax.hessian(kinetic, 2)
        jac_inertia = jax.jacobian(M, 0)(q, q_buff, dq, dq_buff)
        jac_cross = jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff)
        return jac_cross - 1 / 2 * jnp.transpose(dq) @ jac_inertia

    gamma = 1.75
    K_x = np.diag([gamma, gamma])
    K = np.bmat([[K_x, -K_x], [-K_x, K_x]])
    K = jnp.array(K)

    @jax.jit
    def gravity(q, dq):
        return jax.grad(potential, 0)(q, q_buff, dq, dq_buff) - K @ q

    # return q, dq, tau, M, C_mat @ dq, g, k_f
    return q, dq, tau, inertia_simple, coriolis, K, gravity, friction_simple


def dynamic_matrices(state: jnp.array,
                     dynamics: Callable
                     ) -> jnp.array:
    q, dq, tau, inertia, coriolis, K, gravity, friction = dynamics(state)
    M = inertia(q, dq)
    C = coriolis(q, dq)
    g = gravity(q, dq)
    tau_f = friction(q, dq)

    # return q, dq, tau, M, C, K, g, k_f
    return q, dq, tau, M, C, K, g, tau_f


def simplified_dynamic_matrices(state: jnp.array,
                                dynamics: Callable
                                ) -> jnp.array:
    # _, _, _, M, C, K, g, k_f = dynamics(state)
    _, _, _, M, C, K, g, tau_f = dynamics(state)

    # inertias
    M_rob = M[:2, :2]
    B = M[2:, 2:]

    # coriolis
    C_rob = C[:2, :2]

    # potentials
    K_red = K[:2, :2]
    g_rob = g[:2]

    # frictions
    # k_f_rob = k_f[:2]
    # k_f_mot = k_f[2:]
    tau_f_rob = tau_f[:2]
    tau_f_mot = tau_f[2:]

    return (M_rob, C_rob, g_rob, tau_f_rob), (B, tau_f_mot), K_red


def equation_of_motion(state: jnp.array,
                       dynamics: Callable
                       ) -> jnp.array:
    # q, dq, tau, M, C, g, k_f = dynamics(state)
    # q, dq, tau, M, C, K, g, k_f = dynamics(state)
    q, dq, tau, M, C, K, g, tau_f = dynamics(state)

    # account for friction
    # tau_eff = tau - k_f * dq
    tau_eff = tau - tau_f

    # backward dyns.
    # ddq = jnp.linalg.pinv(M) @ (tau_eff - C - g)
    ddq = jnp.linalg.pinv(M) @ (tau_eff - C @ dq - K @ q - g)

    return jnp.concatenate([dq, ddq])


# def equation_of_motion_lag(lagrangian: Callable, state: jnp.array, t=None):
#     q, q_t, tau = jnp.split(state, 3)
#     q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
#             @ (jax.grad(lagrangian, 0)(q, q_t) + tau
#                - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
#     return jnp.concatenate([q_t, q_tt])

def forward_dynamics(ddq: jnp.array,
                     state: jnp.array,
                     dynamics: Callable,
                     ) -> jnp.array:
    # q, dq, tau_target, M, C, g, k_f = dynamics(state)
    # q, dq, tau_target, M, C, K, g, k_f = dynamics(state)
    q, dq, tau_target, M, C, K, g, tau_f = dynamics(state)

    # foward dyns.
    tau_eff = M @ ddq + C @ dq + K @ q + g

    # account for friction
    # tau_fric = k_f * dq
    tau = tau_eff + tau_f

    # TODO: Move this elsewhere
    T_rec = 1 / 2 * jnp.transpose(dq) @ M @ dq
    # jnp.clip(T_rec, a_min=0, a_max=None)
    pow_fric = jnp.transpose(dq) @ tau_f
    pow_cont = jnp.transpose(dq) @ tau_target

    return (tau, tau_target, tau_f), (pow_cont, pow_fric), M, T_rec, g


@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state: jnp.array, times: jnp.array):
    return odeint(model.f_analytical, initial_state, t=times, rtol=1e-13, atol=1e-13)


# def solve_lagrangian(initial_state, lagrangian, **kwargs):
#     @partial(jax.jit, backend='cpu')
#     def f(initial_state):
#         eqs_motion = partial(equation_of_motion_lag, lagrangian)
#         return odeint(eqs_motion,
#                       initial_state,
#                       **kwargs)
#
#     return f(initial_state)


def energy_func(params: dict, train_state: ts.TrainState,
                output: str = 'lagrangian') -> Callable:
    @jax.jit
    def compiled_func(q: jnp.array, q_buff: jnp.array, dq: jnp.array, dq_buff:
    jnp.array):
        # state = jnp.concatenate([q, q_buff, dq, dq_buff])
        state = jnp.concatenate([q, q_buff])
        out = train_state.apply_fn({'params': params}, x=state)
        l = out[:10]
        L = [[l[0], 0, 0, 0],
             [l[1], l[2], 0, 0],
             [l[3], l[4], l[5], 0],
             [l[6], l[7], l[8], l[9]]]
        L = jnp.array(L)
        M = L @ jnp.transpose(L) + jnp.eye(4,4) * 1
        T = 1/2 * jnp.transpose(dq) @ M @ dq
        # T = out[0]
        V = out[10]
        # k_dq = out[-4:]
        # k_dq = jnp.tanh(out[-4:]) + 1
        # k_dq = jnp.abs(out[-4:])
        # k_tau = out[-4:] ** 2
        if output == 'energies':
            return T, V
        elif output == 'lagrangian':
            return T - V
        elif output == 'potential':
            return V
        elif output == 'kinetic':
            return T
        elif output == 'friction':
            # return (k_tau, k_dq)
            return jnp.array([0,0,0,0])
            # return k_dq
        elif output == 'inertia':
            return M

    return compiled_func


# @jax.jit
def energy_calcuations(batch,
                       dyn_terms,
                       kinetic: Callable,
                       potential: Callable):
    # Unpack the terms
    state, ddq = batch
    q, q_buff, dq, dq_buff, tau = split_state(state, 10)
    # _, (M, N), (k_tau, k_dq) = dyn_terms
    _, (M, N), k_dq = dyn_terms

    # Get the energies as the output of the NN
    T = kinetic(q, q_buff, dq, dq_buff)
    V = potential(q, q_buff, dq, dq_buff)
    # M = inertia(q, q_buff, dq, dq_buff)

    # Get the derivatives on q and dq
    dT_q = jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff)
    dV_q = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)
    # dV_dq = jax.grad(potential, 2)(q, q_buff, dq, dq_buff)

    # Reconstruct the kin.energy from its deriv.
    T_rec = 1 / 2 * jnp.transpose(dq) @ M @ dq

    # Reconstruct the gradient of V
    dV_q_rec = - 1 / 2 * (M @ ddq[-4:] + N + dT_q - dV_q)

    # Calculate powers
    pow_T = jnp.transpose(dq) @ dT_q
    pow_V = jnp.transpose(dq) @ dV_q
    pow_input = jnp.transpose(dq) @ tau
    # pow_aff = - jnp.transpose(dq) @ (k_tau * tau)
    pow_f = jnp.transpose(dq) @ (- k_dq * dq)
    # pow_f = jnp.transpose(dq) @ tau_f

    # return T, V, T_rec, dV_dq, M, (pow_V, pow_T, pow_input, pow_aff, pow_f)
    return T, V, T_rec, dV_q, dV_q_rec, M, (pow_V, pow_T, pow_input, pow_f)


@jax.jit
def loss(params: dict,
         train_state: ts.TrainState,
         batch: (jnp.array, jnp.array)) -> jnp.array:
    # @jax.jit
    def reduced_loss(params, train_state, batch):
        # (L_acc_ddq, L_acc_tau), \
        #     (L_mass, L_kin_pos, L_kin_shape, L_dV_shape), \
        #     L_con = loss_instant(params, train_state, batch)

        (L_acc_ddq, L_acc_tau) = loss_instant(params, train_state, batch)

        L_acc = (L_acc_ddq + L_acc_tau * 10000) / 2
        # L_mec = L_mass + L_kin_pos * 100 + L_kin_shape * 100 + L_dV_shape * 100
        # L_energies = L_con

        return L_acc

    compiled_loss = jax.vmap(jax.jit(partial(reduced_loss,
                                             params,
                                             train_state,
                                             )))
    L = compiled_loss(batch)
    return jnp.mean(L)


@jax.jit
def loss_instant(params: dict,
                 train_state: ts.TrainState,
                 batch: (jnp.array, jnp.array)) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch

    # Build dynamics
    kinetic_func = energy_func(params, train_state, output='kinetic')
    potential_func = energy_func(params, train_state, output='potential')
    lagrangian_func = energy_func(params, train_state, output='lagrangian')
    friction_func = energy_func(params, train_state, output='friction')
    inertia_func = energy_func(params, train_state, output='inertia')
    # dyn_builder = partial(lagrangian_dyn_builder,
    #                       lagrangian=lagrangian_func,
    #                       friction=friction_func)
    # dyn_builder = partial(energy_dyn_builder,
    #                       kinetic=kinetic_func,
    #                       potential=potential_func,
    #                       lagrangian=lagrangian_func,
    #                       friction=friction_func)
    dyn_builder = partial(inertia_dyn_builder,
                          kinetic=kinetic_func,
                          potential=potential_func,
                          inertia=inertia_func,
                          friction=friction_func)
    dyn_builder_compiled = jax.jit(dyn_builder)
    dyn_terms = dyn_builder_compiled(state)

    # Calculate the inverse and forward dynamics
    ddq_pred = inv_dyn_lagrangian(dyn_terms)
    tau_pred, tau_target, _ = for_dyn_lagrangian(ddq=ddq_target,
                                                 terms=dyn_terms)

    # # Calculate the energies and related terms
    # energy_calcs = jax.jit(partial(energy_calcuations,
    #                                kinetic=kinetic_func,
    #                                potential=potential_func))
    # T, V, T_rec, dV_q, dV_q_rec, M, (pow_V, pow_T, pow_input, pow_f) = \
    #     energy_calcs(batch=batch,
    #                  dyn_terms=dyn_terms)

    # Caclulate model accuracy error
    L_acc_ddq = jnp.mean((ddq_pred - ddq_target) ** 2)
    L_acc_tau = jnp.mean((tau_pred - tau_target) ** 2)

    # # Calculate energetic errors
    # # positive hamiltonian
    # # H = T + V
    # # L_ham = jnp.mean(jnp.clip(H, a_min=None, a_max=0) ** 2)
    #
    # # energy conservation on the losses
    # # pow_loss = pow_aff + pow_f
    # pow_loss = pow_f
    # power_residual = pow_V + pow_T - pow_input - pow_loss
    # L_con = jnp.mean(power_residual ** 2)
    #
    # # purely dissipative losses
    # # pow_loss_pos = jnp.clip(pow_loss, a_min=0, a_max=None)
    # # L_loss = jnp.mean(pow_loss_pos ** 2)
    # # L_energies = L_ham + L_con + L_loss
    #
    # # Calculate error in energy generation
    # L_mass = L_mass_func(M)
    # L_kin_pos = jnp.clip(T, a_min=None, a_max=0) ** 2
    # L_kin_shape = jnp.mean((T - T_rec) ** 2)
    # dV_q_rec = jax.lax.stop_gradient(dV_q_rec)
    # L_dV_shape = jnp.mean((dV_q - dV_q_rec) ** 2)
    # # L_pot = jnp.mean(dV_dq ** 2) * 1000
    # # L_mec = L_mass + L_kin_pos + L_kin_shape
    #
    # return (L_acc_ddq, L_acc_tau), \
    #     (L_mass, L_kin_pos, L_kin_shape, L_dV_shape), \
    #     L_con

    return (L_acc_ddq, L_acc_tau)


# @jax.jit
# def loss(params: dict,
#          train_state: ts.TrainState,
#          batch: (jnp.array, jnp.array)) -> jnp.array:
#     # Unpack training data
#     state, qdd_target = batch
#
#     # Build dynamics
#     kinetic_func = energy_func(params, train_state, output='kinetic')
#     potential_func = energy_func(params, train_state, output='potential')
#     friction_func = energy_func(params, train_state, output='friction')
#     compiled_dynamics = partial(calc_dynamics,
#                                 kinetic=kinetic_func,
#                                 potential=potential_func,
#                                 friction=friction_func)
#     compiled_dyn_wrapper = jax.jit(partial(dynamic_matrices,
#                                            dynamics=compiled_dynamics))
#     inv_dyn = jax.vmap(jax.jit(partial(equation_of_motion,
#                                        dynamics=compiled_dyn_wrapper)))
#     for_dyn_single = jax.jit(partial(forward_dynamics,
#                                      dynamics=compiled_dyn_wrapper))
#
#     # wrap for_dyn_single to handle data format
#     def for_dyn_wrapped(data_point):
#         state, ddq = data_point
#         ddq = ddq[4:8]
#         return for_dyn_single(ddq=ddq, state=state)
#
#     for_dyn = jax.vmap(for_dyn_wrapped)
#
#     # Predict joint accelerations and calculate error
#     qdd_pred = inv_dyn(state)
#     L_acc_qdd = jnp.mean((qdd_pred - qdd_target) ** 2)
#
#     # Predict the torques and calculate error
#     (tau_prediction, tau_target, tau_fric), (pow_cont, pow_fric), M, T_rec, gravity = \
#         for_dyn(batch)
#     L_acc_tau = jnp.mean((tau_prediction - tau_target) ** 2)
#     L_acc = (L_acc_qdd + L_acc_tau * 1000) / 2
#
#     # Calculate energies and their derivatives
#     energies = jax.vmap(partial(learned_energies,
#                                 params=params,
#                                 train_state=train_state))
#     T, V, V_dot, (pow_V, pow_T), q, dq = energies(state)
#
#     # Impose positive energies
#     H = T + V
#     L_ham = jnp.mean(jnp.clip(H, a_min=None, a_max=0) ** 2)
#
#     # Impose energy conservation
#     power_residual = pow_V + pow_T - pow_cont - pow_fric
#     L_pow = jnp.mean(power_residual ** 2)
#
#     # Impose clean derivative
#     L_kin = jnp.mean((T - T_rec) ** 2) * 10
#
#     # Imppose symetry of inertia matrix
#     @jax.jit
#     def L_mass_func(M):
#         # top_M = jnp.triu(M, k=1)
#         # bot_M = jnp.tril(M, k=-1)
#         diag_M = jnp.diagonal(M)
#         # diag_M = jnp.real(jnp.linalg.eigvals(M))
#         diag_M = jnp.clip(diag_M, a_min=None, a_max=0)
#         return jnp.mean((M - jnp.transpose(M)) ** 2) + jnp.sum(diag_M ** 2)
#
#     L_mass = jnp.mean(jax.vmap(L_mass_func)(M)) * 10
#
#     # Impose independence form q_dot on V due to mechanical system
#     L_pot = jnp.mean(V_dot ** 2)
#
#     # Impose friction model
#     pow_fric_pos = jnp.clip(pow_fric, a_min=0, a_max=None)
#     L_fric = jnp.mean(pow_fric_pos ** 2)
#
#     # Impose no gravity
#     # L_g = jnp.mean(gravity ** 2)
#
#     # return L_acc
#     L_mec = L_kin + L_pot + L_mass
#     # L_refining = 1000 * L_mec + 100 * L_pow + 1000 * L_tau_fric + 100 * L_g
#     L_refining = 100 * L_pow + L_fric + L_ham
#     return L_acc + 1000 * L_mec + L_refining


# # TODO: This function needs to be unified with the previous one somehow
# @jax.jit
# def loss_sample(pair, params=None, train_state=None):
#     state, targets = pair
#
#     # calculate energies
#     T, V, T_rec, V_dot, M = partial(learned_energies, params=params,
#                                     train_state=train_state)(state)
#     H = T_rec + V
#
#     # predict joint accelerations
#     eqs_motion = partial(equation_of_motion,
#                          kinetic=energy_func(params, train_state,
#                                              output='kinetic'),
#                          potential=energy_func(params, train_state,
#                                                output='potential'))
#     preds = eqs_motion(state)
#     L_acc = jnp.mean((preds - targets) ** 2)
#     # print(L_acc.shape)
#
#     # impose energy conservation
#     L_con = H ** 2
#
#     # impose clean derivative
#     L_kin = (T - T_rec) ** 2
#
#     # impose independence form q_dot on V due to mechanical system
#     L_pot = jnp.mean(V_dot ** 2)
#
#     def L_mass_func(M):
#         top_M = jnp.triu(M, k=1)
#         bot_M = jnp.tril(M, k=-1)
#         diag_M = jnp.diag(M)
#         diag_M = jnp.clip(diag_M, a_min=None, a_max=0)
#         return jnp.mean((top_M - jnp.transpose(bot_M)) ** 2) + jnp.mean(diag_M ** 2)
#
#     L_mass = L_mass_func(M)
#
#     L_total = L_acc + 1000 * (L_kin + L_pot + L_mass)
#     return L_total, L_acc, 1000 * (L_kin + L_pot + L_mass)


def create_train_state(settings: dict, learning_rate_fn: Callable, params: dict =
None) -> ts.TrainState:
    # Unpack settings
    key = jax.random.PRNGKey(settings['seed'])
    buffer_length = settings['buffer_length']
    we_param = settings['weight_decay']
    num_dof = settings['num_dof']

    # Create network
    network = MLP()

    # If available load the parameters
    if params is None:
        input_size = (1 * num_dof * buffer_length,)
        params = network.init(key, jax.random.normal(key, input_size))['params']

    # Set up the optimizer and bundle everything into a train state
    adam_opt = optax.adamw(learning_rate=learning_rate_fn, weight_decay=we_param)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)


@partial(jax.jit, static_argnums=2)
def train_step(train_state: ts.TrainState, batch: (jnp.array, jnp.array),
               learning_rate_fn: Callable) -> (ts.TrainState, dict):
    # Creates compiled function that contains the batch data
    @jax.jit
    def loss_fn(params: dict):
        return loss(params, train_state, batch)

    # Update the model
    loss_value, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    # Build the result metrics
    metrics = {'learning_rate': learning_rate_fn(train_state.step), 'loss': loss_value}

    return train_state, metrics


@jax.jit
def eval_step(train_state: ts.TrainState, test_batch: (jnp.array, jnp.array)) -> dict:
    loss_value = loss(train_state.params, train_state, test_batch)
    return {'loss': loss_value}


def run_training(train_state: ts.TrainState, dataloader: Callable, settings: dict,
                 run: Run) -> (dict, tuple):
    # Unpack Settings
    test_every = settings['test_every']
    num_batches = settings['num_batches']
    num_minibatches = settings['num_minibatches']
    num_epochs = settings['num_epochs']
    num_skips = settings['eff_datasampling']
    lr_func = settings['lr_func']
    early_stopping_gain = settings['es_gain']

    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    best_loss = np.inf
    best_params = None

    try:
        epoch_loss_last = np.inf
        epoch = 0
        x_train_large = None
        xt_train_large = None
        x_test_large = None
        xt_test_large = None
        while epoch < num_epochs:
            # Sample key for each epoch
            random_key = jax.random.PRNGKey(settings['seed'] + epoch)

            # Get new samples for the next num_skips epochs
            if epoch % num_skips == 0:
                batch_train_large, batch_test_large = dataloader(settings['seed'] +
                                                                 epoch)
                x_train_large, xt_train_large = batch_train_large
                x_test_large, xt_test_large = batch_test_large

                x_train_large = jnp.split(x_train_large, num_skips)
                xt_train_large = jnp.split(xt_train_large, num_skips)

                x_test_large = jnp.split(x_test_large, num_skips)
                xt_test_large = jnp.split(xt_test_large, num_skips)

            # Split training data into minibatches
            x_train_minibatches = jnp.split(x_train_large[epoch % num_skips],
                                            num_minibatches)
            xt_train_minibatches = jnp.split(xt_train_large[epoch % num_skips],
                                             num_minibatches)
            batch_test = (
                x_test_large[epoch % num_skips], xt_test_large[epoch % num_skips])

            # Train model on each batch
            epoch_loss = 0
            epoch_test_loss = 0
            train_metrics = None
            for batch in range(num_batches):
                train_loss = 0
                for minibatch in range(num_minibatches):
                    minibatch_current = (
                        x_train_minibatches[minibatch], xt_train_minibatches[minibatch])
                    train_state, train_metrics = train_step(train_state,
                                                            minibatch_current, lr_func)
                    train_loss += train_metrics['loss'] / num_minibatches

                # When a batch is done
                epoch_loss += train_loss / num_batches

                # Evaluate model with the test data
                test_loss = eval_step(train_state, batch_test)['loss']
                epoch_test_loss += test_loss / num_batches

                # Check for the best params per batch (test_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_params = copy(train_state.params)

            if epoch_loss > epoch_loss_last * early_stopping_gain:
                print(f'Early stopping! Epoch loss: {epoch_loss}. Now resetting.')
                settings['seed'] += 10
                train_state = create_train_state(settings, lr_func,
                                                 params=best_params)
                epoch = 0

                # batch_train, batch_test = dataloader(settings['seed'] + epoch)
                # x_train, xt_train = batch_train
                # x_train_minibatches = jnp.split(x_train, num_minibatches)
                # xt_train_minibatches = jnp.split(xt_train, num_minibatches)

                continue

            # Record train and test losses
            train_losses.append(epoch_loss)
            test_losses.append(epoch_test_loss)
            run.track(epoch, name='epoch')
            run.track(epoch_loss, name='epoch_loss')
            run.track(epoch_test_loss, name='test_loss')
            run.track(train_metrics['learning_rate'], name='learning_rate')

            # Output progress every 'test_every' epochs
            if epoch % test_every == 0:
                print(
                    f"Epoch={epoch}, "
                    f"train={epoch_loss:.4f}, "
                    f"test={epoch_test_loss:.4f}, "
                    f"lr={train_metrics['learning_rate']:.10f}")

            # Update epoch and record the last loss
            epoch += 1
            epoch_loss_last = epoch_loss

    except KeyboardInterrupt:
        # Save params from model
        print('Terminating learning!')

    return best_params, (train_losses, test_losses)


def generate_trajectory_data(x0: jnp.array, t_window: jnp.array, section_num: int,
                             key: int) -> (jnp.array, jnp.array, jnp.array):
    x_traj = None
    xt_traj = None
    x_start = x0
    for section in range(section_num):
        # Simulate the section from the starting point and update starting point
        x_traj_sec = solve_analytical(x_start, t_window)
        x_start = x_traj_sec[-1]

        # Randomize the order of the data and calculate labels
        x_traj_sec = jax.random.permutation(key, x_traj_sec)
        xt_traj_sec = jax.vmap(model.f_analytical)(x_traj_sec)
        x_traj_sec = jax.vmap(model.normalize)(x_traj_sec)

        # Check that the simulation ran correctly
        if jnp.any(jnp.isnan(x_traj_sec)) or jnp.any(jnp.isnan(xt_traj_sec)):
            raise ValueError(f'One of the sections contained "nan": {x_traj_sec}')

        # Add section to array
        if x_traj is None:
            x_traj = x_traj_sec
            xt_traj = xt_traj_sec
        else:
            x_traj = jnp.append(x_traj, x_traj_sec, axis=0)
            xt_traj = jnp.append(xt_traj, xt_traj_sec, axis=0)

    print(
        f"Generation successful! Ranges: {jnp.amax(x_traj, axis=0)}, {jnp.amin(x_traj, axis=0)}")
    return (x_traj, xt_traj), x_start


def generate_data(settings: dict) -> ((jnp.array, jnp.array), (jnp.array, jnp.array)):
    # Unpack settings
    N = settings['data_size']
    x0 = np.asarray(settings['starting_point'], dtype=np.float32)
    time_step = settings['time_step']
    section_num = settings['sections_num']
    key = jax.random.PRNGKey(settings['seed'])

    # Create time window
    t_window = np.arange(N / section_num, dtype=np.float32) * time_step

    # Create training data
    print('Generating train data:')
    train_data, x0_test = generate_trajectory_data(x0, t_window, section_num, key)

    # Create test data
    print('Generated test data:')
    # noise = np.random.RandomState(0).randn(x0.size)
    # x0_test = x0 + noise * 1e-3
    test_data, _ = generate_trajectory_data(x0_test, t_window, section_num, key)

    return train_data, test_data


def build_simple_dataloader(batch_train: tuple, batch_test: tuple,
                            settings: dict) -> Callable:
    def dataloader(key):
        return batch_train, batch_test

    return dataloader


def build_general_dataloader(batch_train: tuple, batch_test: tuple,
                             settings: dict) -> Callable:
    # Unpack the settings
    batch_size = settings['batch_size']
    num_minibathces = settings['num_minibatches']

    # Set up help valriables
    data_size = batch_size * num_minibathces
    eqs_motion = jax.jit(jax.vmap(model.f_analytical))

    # eqs_motion = jax.jit(jax.vmap(partial(poormans_solve, time_step)))

    def dataloader(key):
        # Randomly sample inputs
        y0 = jnp.concatenate([jax.random.uniform(key, (data_size, 2)) * 2.0 * np.pi,
                              (jax.random.uniform(key + 10,
                                                  (data_size, 1)) - 0.5) * 10 * 2,
                              (jax.random.uniform(key + 20,
                                                  (data_size, 1)) - 0.5) * 10 * 4],
                             axis=1)
        y0 = jax.vmap(model.normalize)(y0)

        return (y0, eqs_motion(y0)), batch_test

    return dataloader


@jax.jit
def normalize(q):
    return (q + np.pi) % (2 * np.pi) - np.pi


# @jax.jit
def extract_from_sample(sample,
                        buffer_length,
                        buffer_length_max,
                        indices=(0, 0, 1)):
    output = jnp.array([])
    for iteration in range(indices[2] - indices[1]):
        index = iteration + indices[0]
        start = index * buffer_length_max
        end = start + buffer_length
        buffer_chunk = jnp.array(sample[start:end])
        output = jnp.concatenate([output, buffer_chunk])

    return output


def extract_from_sample_split(sample,
                              buffer_length,
                              indices=(0, 1)):
    start = indices[0] * buffer_length
    end = indices[1] * buffer_length

    return jnp.array(sample[start:end])


# @jax.jit
def format_sample(sample, buffer_length, buffer_length_max):
    # build q
    # q1_start = jnp.array([sample[0]])
    # q1_buffer = sample[8: 8 + buffer_length - 1]
    # q1 = jnp.concatenate([q1_start, q1_buffer])
    # q1 = normalize(q1)

    # q2_start = jnp.array([sample[1]])
    # q2_buffer = sample[8 + buffer_length - 1:
    #                    8 + 2 * (buffer_length - 1)]
    # q2 = jnp.concatenate([q2_start, q2_buffer])
    # q2 = normalize(q2)
    # q = jnp.concatenate([q1, q2])
    q_n = extract_from_sample(sample,
                              buffer_length,
                              buffer_length_max,
                              indices=(0, 0, 4))
    q_n = normalize(q_n)

    # build dq
    # dq1_start = jnp.array([sample[2]])
    # dq1_buffer = sample[8 + 2 * (buffer_length - 1):
    #                     8 + 3 * (buffer_length - 1)]
    # dq1 = jnp.concatenate([dq1_start, dq1_buffer])
    # dq2_start = jnp.array([sample[3]])
    # dq2_buffer = sample[8 + 3 * (buffer_length - 1):
    #                     8 + 4 * (buffer_length - 1)]
    # dq2 = jnp.concatenate([dq2_start, dq2_buffer])
    # dq = jnp.concatenate([dq1, dq2])
    dq_n = extract_from_sample(sample,
                               buffer_length,
                               buffer_length_max,
                               indices=(4, 0, 4))

    # build dq_0, ddq, tau
    index_end = 8 * buffer_length_max
    dq_0 = jnp.array([dq_n[0 * buffer_length],
                      dq_n[1 * buffer_length],
                      dq_n[2 * buffer_length],
                      dq_n[3 * buffer_length]])
    ddq_0 = jnp.array(sample[index_end: index_end + 4])
    tau = jnp.array(sample[index_end + 4: index_end + 8])

    # build state variables
    state = jnp.concatenate([q_n, dq_n])
    dstate_0 = jnp.concatenate([dq_0, ddq_0])
    state_ext = jnp.concatenate([state, tau])

    # return state_ext, dstate_0
    return jnp.concatenate([dstate_0, state_ext])


# def format_sample_test(sample, buffer_length):
#     # build q
#     q1_start = jnp.array([sample[1]])
#     q1_buffer = sample[7 + 14: 7 + 14 + buffer_length - 1]
#     q1 = jnp.concatenate([q1_start, q1_buffer])
#     q1 = normalize(q1)
#     q2_start = jnp.array([sample[2]])
#     q2_buffer = sample[7 + 14 + buffer_length - 1:
#                        7 + 14 + 2 * (buffer_length - 1)]
#     q2 = jnp.concatenate([q2_start, q2_buffer])
#     q2 = normalize(q2)
#     q = jnp.concatenate([q1, q2])
#
#     # build dq
#     dq1_start = jnp.array([sample[17]])
#     dq1_buffer = sample[7 + 14 + 2 * (buffer_length - 1):
#                         7 + 14 + 3 * (buffer_length - 1)]
#     dq1 = jnp.concatenate([dq1_start, dq1_buffer])
#     dq2_start = jnp.array([sample[18]])
#     dq2_buffer = sample[7 + 14 + 3 * (buffer_length - 1):
#                         7 + 14 + 4 * (buffer_length - 1)]
#     dq2 = jnp.concatenate([dq2_start, dq2_buffer])
#     dq = jnp.concatenate([dq1, dq2])
#
#     # build ddq, tau, target & state
#     ddq = jnp.array([sample[19], sample[20]])
#     tau = jnp.array([sample[7], sample[8]])
#     target = jnp.concatenate([sample[17:19], ddq])
#     state = jnp.concatenate([q, dq])
#
#     return jnp.concatenate([state, tau]), target

def build_database_dataloader_eff(settings: dict) -> Callable:
    # Unpack the settings
    batch_size = settings['batch_size']
    num_minibatches = settings['num_minibatches']
    num_skips = settings['eff_datasampling']
    buffer_length = settings['buffer_length']
    buffer_length_max = settings['buffer_length_max']

    # Set up help variables
    data_size_train = batch_size * num_minibatches * num_skips
    data_size_test = batch_size * num_skips

    # Set up database connection
    database = sqlite3.connect(settings['database_name'])
    cursor = database.cursor()

    # Count the samples
    table_name = settings['table_name']
    samples_total = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    samples_train = int(samples_total * 0.1)
    samples_validation = int(samples_total * 0.1)
    samples_test = int(samples_total * 0.1)

    # Prepare query commands
    query_sample_training = f'SELECT * FROM {table_name} ' \
                            f'LIMIT {samples_train}'
    query_sample_validation = f'SELECT * FROM {table_name} ' \
                              f'LIMIT {samples_validation} OFFSET {samples_train} '

    format_samples = jax.vmap(jax.jit(partial(format_sample,
                                              buffer_length=buffer_length,
                                              buffer_length_max=buffer_length_max)))
    split_data_vec = jax.vmap(split_data)

    # Fetch all training and validation data
    print('Fetching a lot of data...')
    batch_training_raw = jnp.array(cursor.execute(query_sample_training).fetchall())
    batch_validation_raw = jnp.array(cursor.execute(query_sample_validation).fetchall())

    # Format data
    print('Formatting all that data...')
    batch_training_formatted = format_samples(jnp.array(batch_training_raw))
    batch_validation_formatted = format_samples(jnp.array(batch_validation_raw))

    # Close the database connection
    cursor.close()
    database.close()

    # Create the dataloader
    @jax.jit
    def dataloader(seed):
        key = jax.random.PRNGKey(seed)

        # Random subsample from data
        batch_training_sub = jax.random.choice(key, batch_training_formatted,
                                               (data_size_train,))
        batch_validation_sub = jax.random.choice(key, batch_validation_formatted,
                                                 (data_size_test,))

        # split the data
        batch_training = split_data_vec(batch_training_sub)
        batch_validation = split_data_vec(batch_validation_sub)

        return batch_training, batch_validation

    return dataloader


@jax.jit
def split_data(data):
    state = jnp.array(data[:-8])
    ddq = jnp.array(data[-8:])

    return state, ddq


def build_database_dataloader(settings: dict) -> Callable:
    # Unpack the settings
    batch_size = settings['batch_size']
    num_minibathces = settings['num_minibatches']
    num_skips = settings['eff_datasampling']
    buffer_length = settings['buffer_length']
    buffer_length_max = settings['buffer_length_max']

    # Set up help valriables
    data_size_train = batch_size * num_minibathces * num_skips
    data_size_test = batch_size * num_skips

    # set up database
    database = sqlite3.connect(settings['database_name'])
    cursor = database.cursor()

    # count the samples
    table_name = settings['table_name']
    samples_total = cursor.execute(
        f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    samples_train = int(samples_total * 0.9)
    samples_validation = int(samples_total * 0.1)
    # samples_test = int(samples_total * 0.1)

    # query commands
    # query_sample_test = f'SELECT * FROM your_table ORDER BY RANDOM() LIMIT ' \
    #                     f'{samples_train + samples_validation},{samples_test} ' \
    #                     f'LIMIT {data_size}'

    format_samples = jax.vmap(jax.jit(partial(format_sample,
                                              buffer_length=buffer_length,
                                              buffer_length_max=buffer_length_max)))

    # @jax.jit
    def dataloader(key):
        def query_sample_training(key):
            query = f'SELECT * FROM (SELECT * FROM {table_name} ' \
                    f'LIMIT {samples_train}) ORDER BY RANDOM() ' \
                    f'LIMIT {data_size_train}'
            return query

        def query_sample_validation(key):
            query = f'SELECT * FROM (SELECT * FROM {table_name} ' \
                    f'LIMIT {samples_validation} ' \
                    f'OFFSET {samples_train}) ' \
                    f'ORDER BY RANDOM() ' \
                    f'LIMIT {data_size_test}'
            return query

        batch_training_raw = cursor.execute(query_sample_training(key)).fetchall()
        batch_training = format_samples(jnp.array(batch_training_raw))

        batch_validation_raw = cursor.execute(query_sample_validation(key)).fetchall()
        batch_validation = format_samples(jnp.array(batch_validation_raw))

        return batch_training, batch_validation

    return dataloader


def calibrate(E_ana, E_learned):
    # Calculate means
    mean_ana = jnp.mean(E_ana)
    mean_learned = jnp.mean(E_learned)

    # Calculate signal height
    height_ana = (jnp.max(E_ana - mean_ana) - jnp.min(E_ana - mean_ana)) / 2
    height_learned = (jnp.max(E_learned - mean_learned) - jnp.min(E_learned -
                                                                  mean_learned)) / 2

    # Calculate coefficients for linear correction
    alpha = height_ana / height_learned
    beta = - mean_learned * alpha + mean_ana
    E_cal = E_learned * alpha + beta
    coeffs = [alpha, beta]

    return coeffs, E_cal


def display_results(losses: tuple):
    train_losses, test_losses = losses
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.yscale('log')
    # plt.ylim(None, 1000)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()
