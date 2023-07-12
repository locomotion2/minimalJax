from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from identification.systems import snake_utils

import identification.src.dynamix.energiex as ex
import identification.src.dynamix.motionx as mx


def build_energy_call(settings, params, train_state):
    # build the energy functions
    kinetic = ex.build_energy_func(
        params, train_state, settings=settings, output="kinetic"
    )
    potential = ex.build_energy_func(
        params, train_state, settings=settings, output="potential"
    )

    # choose a splitting tool
    split_tool = settings["system_settings"]["sys_utils"].build_split_tool(
        settings["model_settings"]["buffer_length"])

    # Load the calibration coeffs
    coeff_data = jnp.array(settings["system_settings"]["calib_coeffs"])
    coef_V, coef_T = jnp.split(coeff_data, 2)

    @jax.jit
    def energy_call(state: jnp.array):
        # unpack the state
        q, q_buff, dq, dq_buff, _ = split_tool(state)

        # calculate raw energies
        T = kinetic(q, q_buff, dq, dq_buff)
        V = potential(q, q_buff, dq, dq_buff)

        # scale up the energies
        V_f = V * coef_V[0] + coef_V[1]
        T_f = T * coef_T[0]

        return T_f, V_f

    return energy_call


def build_dynamics(goal, settings, params, train_state):

    # build energy function
    kinetic = None
    potential = None
    friction = None
    inertia = None
    if goal == "energy":
        kinetic = ex.build_energy_func(
            params, train_state, settings=settings, output="kinetic"
        )
        potential = ex.build_energy_func(
            params, train_state, settings=settings, output="potential"
        )
        friction = ex.build_energy_func(
            params, train_state, settings=settings, output="friction"
        )
        inertia = ex.build_energy_func(
            params, train_state, settings=settings, output="inertia"
        )
    elif goal == "model":
        kinetic = ex.build_energy_func_model(
            params, train_state, settings=settings, output="kinetic"
        )
        potential = ex.build_energy_func_model(
            params, train_state, settings=settings, output="potential"
        )
        friction = ex.build_energy_func_model(
            params, train_state, settings=settings, output="friction"
        )
        inertia = ex.build_energy_func_model(
            params, train_state, settings=settings, output="inertia"
        )

    # choose a splitting tool
    split_tool = settings["system_settings"]["sys_utils"].build_split_tool(
        settings["model_settings"]["buffer_length"])

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

    return dyn_builder_compiled

def build_test_funcs(settings, params, train_state):
    # Create basic building blocks
    kinetic = ex.build_energy_func(
        params, train_state, settings=settings, output="kinetic"
    )
    potential = ex.build_energy_func(
        params, train_state, settings=settings, output="potential"
    )
    friction = ex.build_energy_func(
        params, train_state, settings=settings, output="friction"
    )
    inertia = ex.build_energy_func(
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
