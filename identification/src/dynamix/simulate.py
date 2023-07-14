from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from typing import Callable

import identification.identification_utils as utils


# TODO: Test if this is working
@partial(jax.jit, backend="cpu")
def solve_analytical(model, initial_state: jnp.array, times: jnp.array):
    return odeint(model.f_analytical, initial_state, t=times, rtol=1e-13, atol=1e-13)


def solve_eom(initial_state, eom_compiled, time_span, params):
    @partial(jax.jit, backend="cpu")
    # @jax.jit
    def f(initial_state):
        dx = eom_compiled(x=initial_state, t=0, params=params)

        final_state = initial_state + dx * time_span[-1]

        return [initial_state, final_state]

    return f(initial_state)


# @jax.jit
def simulate(
        state,
        tau,
        buffer_length,
        sys_utils,
        num_dof,
        samples_num,
        eom_prepared,
        split_tool,
        data_formatted,
):
    # unpack settings
    # buffer_length = settings['buffer_length']
    # sys_utils = settings['sys']
    # num_dof = settings['num_dof']

    # prepare initial state
    state_long, _ = sys_utils.split_data(data_formatted[0])
    q, q_buff_input, dq, dq_buff_input, _ = split_tool(state[0])
    q_buff_long, dq_buff_long = jnp.split(state_long[:-4], 2)
    q_buff_wide = jnp.reshape(q_buff_long, (num_dof, buffer_length))
    dq_buff_wide = jnp.reshape(dq_buff_long, (num_dof, buffer_length))
    x_input = jnp.concatenate([q, dq])

    q_sim = jnp.array([q])
    dq_sim = jnp.array([dq])
    # ddq_sim = jnp.array([0])
    for index, tau_cur in enumerate(tau):
        print(f"Progress: {index / samples_num * 100}%")

        # prepare the equation
        eom_compiled = jax.jit(
            partial(eom_prepared, q_buff=q_buff_input, dq_buff=dq_buff_input)
        )

        # calculate the enchilada
        out = solve_eom(x_input, eom_compiled, jnp.array([0, 1 / 100]), tau_cur)
        # out = odeint(eom_compiled,
        #              x_input,
        #              jnp.array([0, 1 / 100]),
        #              (tau_cur,))

        # update the values
        x_input = out[-1]
        q_cur, dq_cur = jnp.split(x_input, 2)
        (q_buff_input, dq_buff_input), (
            q_buff_wide,
            dq_buff_wide,
        ) = utils.format_state_sim(q_cur, q_buff_wide, dq_cur, dq_buff_wide)

        # save in vectors
        q_sim = jnp.append(q_sim, jnp.array([q_cur]), axis=0)
        dq_sim = jnp.append(dq_sim, jnp.array([dq_cur]), axis=0)
        # dq_sim = jnp.append(dq_sim, jnp.array([dq_cur]), axis=0)

    return q_sim, dq_sim


@partial(jax.jit, backend='cpu', static_argnums=0)
def solve_analytical(analytical_eom: Callable,
                     initial_state: jnp.array,
                     times: jnp.array):
    return odeint(analytical_eom, initial_state, t=times, rtol=1e-13, atol=1e-13)

# def solve_lagrangian(initial_state, lagrangian, **kwargs):
#     @partial(jax.jit, backend='cpu')
#     def f(initial_state):
#         eqs_motion = partial(equation_of_motion_lag, lagrangian)
#         return odeint(eqs_motion,
#                       initial_state,
#                       **kwargs)
#
#     return f(initial_state)
