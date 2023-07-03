from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax.training import train_state as ts
from jax.experimental.ode import odeint

from Lagranx.src import dpend_model_arne as model


# def energy_dyn_builder(state: jnp.array,
#                        kinetic: Callable = None,
#                        potential: Callable = None,
#                        friction: Callable = None
#                        ):
#     # unpack the data
#     q, q_buff, dq, dq_buff, tau = sys_utils.split_state(state, 20)
# 
#     # obtain equation terms
#     M = jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)
#     cross_hessian = jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff)
#     N = cross_hessian @ dq - (jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff) -
#                               jax.grad(potential, 0)(q, q_buff, dq, dq_buff))
# 
#     # bundle terms
#     state_current = (q, dq, tau)
#     dynamics = (M, N)
#     frictions = friction(q, q_buff, dq, dq_buff)
# 
#     return state_current, dynamics, frictions


def inertia_dyn_builder(state: jnp.array,
                        split_tool: Callable = None,
                        potential: Callable = None,
                        inertia: Callable = None,
                        friction: Callable = None
                        ):
    q, q_buff, dq, dq_buff, tau = split_tool(state)

    # obtain equation terms
    M = inertia(q, q_buff, dq, dq_buff)

    def dT_q(q, q_buff, dq, dq_buff):
        M = inertia(q, q_buff, dq, dq_buff)
        return M @ dq

    dM = jax.jacobian(dT_q, 0)(q, q_buff, dq, dq_buff)
    C = 1 / 2 * dM
    dV_q = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)

    # bundle terms
    state_current = (q, dq, tau)
    dynamics = (M, C, dV_q)
    frictions = friction(q, q_buff, dq, dq_buff)

    return state_current, dynamics, frictions


@partial(jax.jit, static_argnums=1)
def decouple_model(state: jnp.array,
                   dynamics: Callable = None):
    _, dynamics, frictions = dynamics(state)

    # (q, dq, tau) = state_current
    (M, C, dV_q) = dynamics
    k_dq = frictions

    # matrices
    rows, cols = M.shape
    M_rob = M[:rows // 2, :cols // 2]
    C_rob = C[:rows // 2, :cols // 2]
    M_mot = M[rows // 2:, cols // 2:]
    # C_mot = C[rows // 2:, cols // 2:]

    # vectors
    dV_q_rob, dV_q_mot = jnp.split(dV_q, 2)
    k_dq_rob, k_dq_mot = jnp.split(k_dq, 2)

    return (M_rob, C_rob, dV_q_rob, k_dq_rob), (M_mot, k_dq_mot)


@jax.jit
def forward_dynamics(terms):
    # unpack terms
    (q, dq, tau), (M, C, dV_q), k_dq = terms

    # account for frictions
    tau_f = - k_dq * dq
    tau_eff = tau + tau_f

    # forward dynamics (EOMs)
    ddq = jnp.linalg.pinv(M) @ (tau_eff - C @ dq - dV_q)

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


# TODO: Test if this is working
@partial(jax.jit, backend='cpu')
def solve_analytical(model, initial_state: jnp.array, times: jnp.array):
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
                settings: dict = None,
                output: str = 'lagrangian') -> Callable:
    h_dim = settings['h_dim']
    num_dof = settings['num_dof']
    friction = settings['friction']

    @jax.jit
    def compiled_func(q: jnp.array, q_buff: jnp.array,
                      dq: jnp.array, dq_buff: jnp.array):

        # pack the state
        state = jnp.concatenate([q, q_buff, dq, dq_buff])

        # run the NN
        out = train_state.apply_fn({'params': params},
                                   x=state,
                                   net_size=h_dim,
                                   num_dof=num_dof,
                                   friction=friction
                                   )

        # build M
        l = out[:10]
        L = [[l[0], 0, 0, 0],
             [l[1], l[2], 0, 0],
             [l[3], l[4], l[5], 0],
             [l[6], l[7], l[8], l[9]]]
        L = jnp.array(L)
        M = L @ jnp.transpose(L) + jnp.eye(4, 4) * 0.01

        # TODO: Test this transformation
        # M_diag = jnp.diagonal(M)
        # M_diag = nn.softplus(2 + M_diag) + 0.001
        # M = fill_diagonal(M, M_diag)

        # calculate T
        T = 1 / 2 * jnp.transpose(dq) @ M @ dq

        # get V
        V = out[10]

        # get the friction
        # k_dq = out[-4:]
        k_dq = jnp.array([0] * 4)

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


def test_calculations(batch,
                      dyn_terms,
                      split_tool: Callable,
                      kinetic: Callable,
                      potential: Callable):
    # Unpack the terms
    state, ddq = batch
    q, q_buff, dq, dq_buff, tau = split_tool(state)
    _, (M, C, dV_q), k_dq = dyn_terms

    # Get the energies as the output of the NN
    T = kinetic(q, q_buff, dq, dq_buff)
    V = potential(q, q_buff, dq, dq_buff)

    # Get the derivatives on q and dq
    dT_q = jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff)
    # dV_q = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)

    # Calculate powers
    pow_T = jnp.transpose(dq) @ dT_q
    pow_V = jnp.transpose(dq) @ dV_q
    pow_input = jnp.transpose(dq) @ tau
    pow_f = jnp.transpose(dq) @ (- k_dq * dq)
    # pow_f = jnp.transpose(dq) @ tau_f

    return T, V, (pow_V, pow_T, pow_input, pow_f)


def build_loss(settings: dict) -> Callable:
    # get the weights
    weights = settings['loss_weights']
    sys_utils = settings['sys_utils']
    buff_len = settings['buffer_length']
    split_tool = sys_utils.build_split_tool(buff_len)

    @jax.jit
    def loss(params: dict,
             train_state,
             batch: (jnp.array, jnp.array)) -> jnp.array:
        # Build dynamics
        potential_func = energy_func(params, train_state, settings=settings,
                                     output='potential')
        friction_func = energy_func(params, train_state, settings=settings,
                                    output='friction')
        inertia_func = energy_func(params, train_state, settings=settings,
                                   output='inertia')
        dyn_builder = partial(inertia_dyn_builder,
                              split_tool=split_tool,
                              potential=potential_func,
                              inertia=inertia_func,
                              friction=friction_func)
        dyn_builder_compiled = jax.jit(dyn_builder)

        # Compile vectorized batch function
        @jax.jit
        def weighted_loss(sample):
            losses = loss_sample(split_tool,
                                 dyn_builder_compiled,
                                 sample)
            return jnp.transpose(jnp.array(weights)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    return loss


@partial(jax.jit, static_argnums=[0, 1])
def loss_sample(split_tool: Callable,
                dyn_builder: Callable,
                batch: (jnp.array, jnp.array)
                ) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch
    _, _, dq, _, _ = split_tool(state)
    dyn_terms = dyn_builder(state)

    # Calculate the inverse and forward dynamics
    ddq_pred = forward_dynamics(dyn_terms)
    tau_pred, tau_target, tau_fric = inverse_dynamics(ddq=ddq_target,
                                                      terms=dyn_terms)

    # Calculate model accuracy error
    # L_acc_ddq = jnp.mean(((ddq_pred - ddq_target) / jnp.linalg.norm(ddq_target)) ** 2)
    # L_acc_tau = jnp.mean(((tau_pred - tau_target) / jnp.linalg.norm(tau_target)) ** 2)
    L_acc_ddq = jnp.mean((ddq_pred - ddq_target) ** 2)
    L_acc_tau = jnp.mean((tau_pred - tau_target) ** 2)

    # Calculate temporal energy error
    pow_pred = jnp.transpose(dq) @ tau_pred
    pow_target = jnp.transpose(dq) @ tau_target
    # L_energy = jnp.mean(((pow_pred - pow_target) / jnp.abs(pow_target)) ** 2)
    L_energy = jnp.mean((pow_pred - pow_target) ** 2)

    return jnp.array([L_acc_ddq, L_acc_tau, L_energy])
