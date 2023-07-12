from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


def energy_dyn_builder(
    state: jnp.array,
    split_tool: Callable = None,
    kinetic: Callable = None,
    potential: Callable = None,
    friction: Callable = None,
):
    # unpack the data
    q, q_buff, dq, dq_buff, tau = split_tool(state)

    # # obtain equation terms
    # M = jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)
    # cross_hessian = jax.jacobian(jax.grad(kinetic, 2), 0)(q, q_buff, dq, dq_buff)
    # N = cross_hessian @ dq - (jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff) -
    #                           jax.grad(potential, 0)(q, q_buff, dq, dq_buff))

    # bundle terms
    state_current = (q, dq, tau)
    # dynamics = (M, N)
    dynamics = (0, 0)
    frictions = friction(q, q_buff, dq, dq_buff, tau)

    return state_current, dynamics, frictions


def inertia_dyn_builder(
    state: jnp.array,
    split_tool: Callable = None,
    potential: Callable = None,
    kinetic: Callable = None,
    inertia: Callable = None,
    friction: Callable = None,
):
    q, q_buff, dq, dq_buff, tau = split_tool(state)

    # obtain equation terms
    M = inertia(q, q_buff, dq, dq_buff)

    def dT_dq(q, q_buff, dq, dq_buff):
        M = inertia(q, q_buff, dq, dq_buff)
        return M @ dq

    dM = jax.jacobian(dT_dq, 0)(q, q_buff, dq, dq_buff)
    # dM = jax.jacobian(jax.grad(kinetic, 2), 0)(q, q_buff, dq, dq_buff)

    # C = 1 / 2 * dM
    C = dM - 1 / 2 * jnp.transpose(dM)
    dV_q = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)

    # bundle terms
    state_current = (q, dq, tau)
    dynamics = (M, C, dV_q)
    frictions = friction(q, q_buff, dq, dq_buff)

    return state_current, dynamics, frictions


@partial(jax.jit, static_argnums=1)
def decouple_model(state: jnp.array, dynamics: Callable = None):
    _, dynamics, frictions = dynamics(state)

    # (q, dq, tau) = state_current
    (M, C, dV_q) = dynamics
    k_dq = frictions

    # matrices
    rows, cols = M.shape
    M_rob = M[: rows // 2, : cols // 2]
    C_rob = C[: rows // 2, : cols // 2]
    M_mot = M[rows // 2 :, cols // 2 :]
    # C_mot = C[rows // 2:, cols // 2:]

    # vectors
    dV_q_rob, dV_q_mot = jnp.split(dV_q, 2)
    k_dq_rob, k_dq_mot = jnp.split(k_dq, 2)

    return (M_rob, C_rob, dV_q_rob, k_dq_rob), (M_mot, k_dq_mot)


def eom_wrapped(
    x: jnp.array,
    t: jnp.array,
    params: jnp.array,
    q_buff: jnp.array = None,
    dq_buff: jnp.array = None,
    potential: Callable = None,
    inertia: Callable = None,
):
    # Unpack data
    q, dq = jnp.split(x, 2)
    tau = params[0]

    # obtain equation terms
    M = inertia(q, q_buff, dq, dq_buff)

    def dT_dq(q, q_buff, dq, dq_buff):
        M = inertia(q, q_buff, dq, dq_buff)
        return M @ dq

    dM = jax.jacobian(dT_dq, 0)(q, q_buff, dq, dq_buff)
    C = dM - 1 / 2 * jnp.transpose(dM)
    dV_q = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)

    # Calculate ddq
    ddq = jnp.linalg.inv(M) @ (tau - C @ dq - dV_q)

    return jnp.concatenate([dq, ddq])


@jax.jit
def forward_dynamics(terms):
    # unpack terms
    (q, dq, tau), (M, C, dV_q), k_dq = terms

    # account for frictions
    tau_f = -k_dq * dq
    tau_eff = tau + tau_f

    # forward dynamics (EOMs)
    ddq = jnp.linalg.inv(M) @ (tau_eff - C @ dq - dV_q)

    return jnp.concatenate([dq, ddq])


@jax.jit
def inverse_dynamics(ddq, terms):
    # unpack terms
    (q, dq, tau_target), (M, C, dV_q), k_dq = terms

    # inverse dynamics
    if len(ddq) > len(dq):
        _, ddq = jnp.split(ddq, 2)
    tau_eff = M @ ddq + C @ dq + dV_q

    # account for frictions
    tau_f = - k_dq * dq
    tau = tau_eff - tau_f

    return tau, tau_target, tau_f


@jax.jit
def forward_dynamics_energies(terms):
    # unpack terms
    (q, dq, tau), (M, N), k_dq = terms

    # account for frictions
    tau_f = -k_dq * dq
    # tau_f = k_dq
    tau_eff = tau + tau_f

    # forward dynamics (EOMs)
    ddq = jnp.linalg.inv(M) @ (tau_eff - N)
    # ddq = k_dq

    return jnp.concatenate([dq, ddq])


@jax.jit
def inverse_dynamics_energies(ddq, terms):
    # unpack terms
    (q, dq, tau_target), (M, N), k_dq = terms

    # inverse dynamics
    if len(ddq) > len(dq):
        _, ddq = jnp.split(ddq, 2)
    tau_eff = M @ ddq + N

    # account for frictions
    tau_f = -k_dq * dq
    tau = tau_eff - tau_f

    return tau, tau_target, tau_f
