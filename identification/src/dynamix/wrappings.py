from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from systems import snake_utils

import energiex as ex
import motionx as mx


def energy_wrapper(
    state: jnp.array, split_tool: Callable, kinetic: Callable, potential: Callable
):
    q, q_buff, dq, dq_buff, _ = split_tool(state)

    T = kinetic(q, q_buff, dq, dq_buff)
    V = potential(q, q_buff, dq, dq_buff)

    # scale the energies
    coef_V = [1, 0]
    coef_T = [0.1795465350151062, -0.06468642503023148]
    # diff_beta = coef_T[1] - coef_V[1]

    V_f = V * coef_V[0] + coef_V[1]
    T_f = T * coef_T[0]

    return T_f, V_f


def build_dynamics(settings, params, train_state):
    # Create basic building blocks
    kinetic = ex.energy_func_model(
        params, train_state, settings=settings, output="kinetic"
    )
    potential = ex.energy_func_model(
        params, train_state, settings=settings, output="potential"
    )
    friction = ex.energy_func_model(
        params, train_state, settings=settings, output="friction"
    )
    inertia = ex.energy_func_model(
        params, train_state, settings=settings, output="inertia"
    )

    split_tool = snake_utils.build_split_tool(settings["model_settings"]["buffer_length"])

    # Create compiled dynamics
    dyn_builder = partial(
        mx.inertia_dyn_builder,
        split_tool=split_tool,
        kinetic=kinetic,
        potential=potential,
        inertia=inertia,
        friction=friction,
    )
    # dyn_builder = partial(lx.energy_dyn_builder,
    #                       split_tool=split_tool,
    #                       potential=potential,
    #                       kinetic=kinetic,
    #                       friction=friction)
    dyn_builder_compiled = jax.jit(dyn_builder)

    # Vectorize some important calculations
    energy_calcs = jax.vmap(
        jax.jit(
            partial(
                ex.test_calculations,
                split_tool=split_tool,
                kinetic=kinetic,
                potential=potential,
            )
        )
    )

    # build the integrable EOMs
    eom_compiled = partial(mx.eom_wrapped, potential=potential, inertia=inertia)

    return dyn_builder_compiled, energy_calcs, eom_compiled, split_tool
