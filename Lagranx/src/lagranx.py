from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax.training import train_state as ts
from jax.experimental.ode import odeint

from Lagranx.src import dpend_model_arne as model
from Lagranx.src import snake_utils as sys_utils


def energy_dyn_builder(state: jnp.array,
                       kinetic: Callable = None,
                       potential: Callable = None,
                       lagrangian: Callable = None,
                       friction: Callable = None
                       ):
    q, q_buff, dq, dq_buff, tau = sys_utils.split_state(state, 20)

    # obtain equation terms
    M = jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)
    cross_hessian = jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff)
    N = cross_hessian @ dq - (jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff) -
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
    q, q_buff, dq, dq_buff, tau = sys_utils.split_state(state, 20)

    # obtain equation terms
    M = inertia(q, q_buff, dq, dq_buff)

    def dT_q(q, q_buff, dq, dq_buff):
        M = inertia(q, q_buff, dq, dq_buff)
        return M @ dq

    dM = jax.jacobian(dT_q, 0)(q, q_buff, dq, dq_buff)
    N = 1 / 2 * dM @ dq + jax.grad(potential, 0)(q, q_buff, dq, dq_buff)

    # bundle terms
    state_current = (q, dq, tau)
    dynamics = (M, N)
    frictions = friction(q, q_buff, dq, dq_buff)

    return state_current, dynamics, frictions


@jax.jit
def forw_dyn(terms):
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
def inv_dyn(ddq, terms):
    # unpack terms
    # ((q, dq, tau_target), (M, N), (k_tau, k_dq)) = terms
    (q, dq, tau_target), (M, N), k_dq = terms

    # perform inverse dynamics
    tau_eff = M @ ddq[-4:] + N

    # correct the torque
    # tau_aff = - k_tau * tau_target
    tau_f = - k_dq * dq
    tau = tau_eff - tau_f

    return tau, tau_target, tau_f


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

# TODO: run the matrix computations only when M or T are wanted
def energy_func(params: dict, train_state: ts.TrainState,
                output: str = 'lagrangian') -> Callable:
    @jax.jit
    def compiled_func(q: jnp.array, q_buff: jnp.array, dq: jnp.array, dq_buff:
    jnp.array):
        # pack the state
        state = jnp.concatenate([q, q_buff, dq, dq_buff])
        # state = jnp.concatenate([q, q_buff])

        # run the NN
        out = train_state.apply_fn({'params': params}, x=state)

        # build M
        l = out[:10]
        L = [[l[0], 0, 0, 0],
             [l[1], l[2], 0, 0],
             [l[3], l[4], l[5], 0],
             [l[6], l[7], l[8], l[9]]]
        L = jnp.array(L)
        M = L @ jnp.transpose(L) + jnp.eye(4, 4) * 0.01
        # M_diag = jnp.diagonal(M)
        # M_diag = nn.softplus(2 + M_diag) + 0.001
        # M = fill_diagonal(M, M_diag)

        # calculate T
        T = 1 / 2 * jnp.transpose(dq) @ M @ dq

        # get V
        # q_actual, theta = jnp.split(q, 2)
        # K_small = jnp.array([[1.75, 0], [0, 1.75]])
        # K = jnp.block([[K_small, -K_small], [-K_small, K_small]])
        # V = 1/2 * jnp.transpose(q) @ K @ q
        V = out[10]

        # get the friction
        # k_dq = out[-4:]
        k_dq = jnp.array([0.0225] * 4)

        # choose the output
        if output == 'energies':
            return T, V
        elif output == 'lagrangian':
            return T - V
        elif output == 'potential':
            return V
        elif output == 'kinetic':
            return T
        elif output == 'friction':
            return k_dq
        elif output == 'inertia':
            return M

    return compiled_func


# TODO: This function needs to be rewritten
def energy_calcuations(batch,
                       dyn_terms,
                       kinetic: Callable,
                       potential: Callable):
    # Unpack the terms
    state, ddq = batch
    q, q_buff, dq, dq_buff, tau = sys_utils.split_state(state, 20)
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


def build_loss(settings: dict) -> Callable:
    # get the wights
    weights = settings['loss_weights']

    @jax.jit
    def loss(params: dict,
             train_state,
             batch: (jnp.array, jnp.array)) -> jnp.array:
        # TODO: Need to add the loss weights as a dic

        @jax.jit
        def weighted_loss(sample):
            losses = loss_instant(params, train_state, sample)
            return jnp.transpose(jnp.array(weights)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    return loss


@jax.jit
def loss_instant(params: dict,
                 train_state: ts.TrainState,
                 batch: (jnp.array, jnp.array)) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch
    q, q_buff, dq, dq_buff, _ = sys_utils.split_state(state, 20)

    # Build dynamics
    kinetic_func = energy_func(params, train_state, output='kinetic')
    potential_func = energy_func(params, train_state, output='potential')
    lagrangian_func = energy_func(params, train_state, output='lagrangian')
    friction_func = energy_func(params, train_state, output='friction')
    inertia_func = energy_func(params, train_state, output='inertia')
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
    ddq_pred = forw_dyn(dyn_terms)
    tau_pred, tau_target, tau_fric = inv_dyn(ddq=ddq_target,
                                             terms=dyn_terms)

    # Caclulate model accuracy error
    # L_acc_ddq = jnp.mean(((ddq_pred - ddq_target) / jnp.linalg.norm(ddq_target)) ** 2)
    # L_acc_tau = jnp.mean(((tau_pred - tau_target) / jnp.linalg.norm(tau_target)) ** 2)
    L_acc_ddq = jnp.mean((ddq_pred - ddq_target) ** 2)
    L_acc_tau = jnp.mean((tau_pred - tau_target) ** 2)

    # Calculate temporal energy error
    pow_pred = jnp.transpose(dq) @ tau_pred
    pow_target = jnp.transpose(dq) @ tau_target
    # L_energy = jnp.mean(((pow_pred - pow_target) / jnp.abs(pow_target)) ** 2)
    L_energy = jnp.mean((pow_pred - pow_target) ** 2)

    # Refining losses
    # V = potential_func(q, q_buff, dq, dq_buff)
    # T = kinetic_func(q, q_buff, dq, dq_buff)
    # q_actual, theta = jnp.split(q, 2)
    # K_small = jnp.array([[1.75, 0], [0, 1.75]])
    # K = jnp.block([[K_small, -K_small], [-K_small, K_small]])
    # V_ana = 1/2 * jnp.transpose(q) @ K @ q
    # H_ana = T + V_ana
    # H_ana = jax.lax.stop_gradient(H_ana)
    #
    # L_pot = jnp.mean(** 2)

    return jnp.array([L_acc_ddq, L_acc_tau, L_energy])
