from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

import energiex as ex
import motionx as mx


def build_loss(settings: dict) -> (Callable, Callable):
    # get the weights
    weights = settings["training_settings"]["loss_weights"]
    weights_model = settings["training_settings"]["loss_weights_model"]
    weights_red = settings["training_settings"]["loss_weights_red"]
    sys_utils = settings["system_settings"]["sys_utils"]
    buff_len = settings["model_settings"]["buffer_length"]
    split_tool = sys_utils.build_split_tool(buff_len)

    @jax.jit
    def loss_energies(
        params: dict, train_state, batch: (jnp.array, jnp.array)
    ) -> jnp.array:
        # Build dynamics
        potential_func = ex.energy_func(
            params, train_state, settings=settings, output="potential"
        )
        friction_func = ex.energy_func(
            params, train_state, settings=settings, output="friction"
        )
        inertia_func = ex.energy_func(
            params, train_state, settings=settings, output="inertia"
        )
        dyn_builder = partial(
            mx.inertia_dyn_builder,
            split_tool=split_tool,
            potential=potential_func,
            inertia=inertia_func,
            friction=friction_func,
        )
        dyn_builder_compiled = jax.jit(dyn_builder)

        # Compile vectorized batch function
        @jax.jit
        def weighted_loss(sample):
            losses = loss_sample(split_tool, dyn_builder_compiled, sample)
            return jnp.transpose(jnp.array(weights)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    @jax.jit
    def loss_model(
        params: dict, train_state, batch: (jnp.array, jnp.array)
    ) -> jnp.array:
        # Build dynamics
        potential_func = ex.energy_func_model(
            params, train_state, settings=settings, output="potential"
        )
        inertia_func = ex.energy_func_model(
            params, train_state, settings=settings, output="inertia"
        )
        friction_func = ex.energy_func_model(
            params, train_state, settings=settings, output="friction"
        )
        dyn_builder = partial(
            mx.inertia_dyn_builder,
            split_tool=split_tool,
            potential=potential_func,
            inertia=inertia_func,
            friction=friction_func,
        )
        dyn_builder_compiled = jax.jit(dyn_builder)

        # Compile vectorized batch function
        @jax.jit
        def weighted_loss(sample):
            losses = loss_sample(split_tool, dyn_builder_compiled, sample)
            return jnp.transpose(jnp.array(weights_model)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    @jax.jit
    def loss_red(
        params: dict, train_state, train_state_red, batch: (jnp.array, jnp.array)
    ) -> jnp.array:
        # Build dynamics
        potential_func = ex.energy_func_red(
            params, train_state_red, settings=settings, output="potential"
        )
        friction_func = ex.energy_func_red(
            params, train_state_red, settings=settings, output="friction"
        )
        inertia_func = ex.energy_func_red(
            params, train_state_red, settings=settings, output="inertia"
        )
        inertia_func_boot = ex.energy_func(
            train_state.params, train_state, settings=settings, output="inertia"
        )
        dyn_builder = partial(
            mx.inertia_dyn_builder,
            split_tool=split_tool,
            potential=potential_func,
            inertia=inertia_func,
            friction=friction_func,
        )
        dyn_builder_compiled = jax.jit(dyn_builder)

        # Compile vectorized batch function
        @jax.jit
        def weighted_loss(sample):
            losses = loss_sample_red(
                split_tool, dyn_builder_compiled, inertia_func_boot, sample
            )
            return jnp.transpose(jnp.array(weights_red)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    loss = None
    if settings["model_settings"]["goal"] == "energy":
        loss = loss_energies
    elif settings["model_settings"]["goal"] == "model":
        loss = loss_model

    return loss, loss_red


@partial(jax.jit, static_argnums=[0, 1])
def loss_sample(
    split_tool: Callable, dyn_builder: Callable, batch: (jnp.array, jnp.array)
) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch
    _, _, dq, _, _ = split_tool(state)
    dyn_terms = dyn_builder(state)

    # Calculate the inverse and forward dynamics
    ddq_pred = mx.forward_dynamics(dyn_terms)
    tau_pred, tau_target, tau_fric = mx.inverse_dynamics(
        ddq=ddq_target, terms=dyn_terms
    )

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


@partial(jax.jit, static_argnums=[0, 1])
def loss_sample_model(
    split_tool: Callable, dyn_builder: Callable, batch: (jnp.array, jnp.array)
) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch
    _, _, dq, _, _ = split_tool(state)
    dyn_terms = dyn_builder(state)

    # Calculate the inverse and forward dynamics
    ddq_pred = mx.forward_dynamics(dyn_terms)
    tau_pred, tau_target, tau_fric = mx.inverse_dynamics(
        ddq=ddq_target, terms=dyn_terms
    )

    # Calculate model accuracy error
    # L_acc_ddq = jnp.mean(((ddq_pred - ddq_target) / jnp.linalg.norm(ddq_target)) ** 2)
    # L_acc_tau = jnp.mean(((tau_pred - tau_target) / jnp.linalg.norm(tau_target)) ** 2)
    L_acc_ddq = jnp.mean((ddq_pred - ddq_target) ** 2)
    L_acc_tau = jnp.mean((tau_pred - tau_target) ** 2)
    # L_acc_tau = 0

    # Calculate temporal energy error
    pow_pred = jnp.transpose(dq) @ tau_pred
    pow_target = jnp.transpose(dq) @ tau_target
    # L_energy = jnp.mean(((pow_pred - pow_target) / jnp.abs(pow_target)) ** 2)
    L_energy = jnp.mean((pow_pred - pow_target) ** 2)
    # L_energy = 0

    return jnp.array([L_acc_ddq, L_acc_tau, L_energy])


@partial(jax.jit, static_argnums=[0, 1, 2])
def loss_sample_red(
    split_tool: Callable,
    dyn_builder: Callable,
    inertia_boot: Callable,
    batch: (jnp.array, jnp.array),
) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch
    q, q_buff, dq, dq_buff, _ = split_tool(state)
    dyn_terms = dyn_builder(state)
    _, (M_pred, _, _), _ = dyn_terms

    # Calculate the inverse and forward dynamics
    ddq_pred = mx.forward_dynamics(dyn_terms)
    tau_pred, tau_target, tau_fric = mx.inverse_dynamics(
        ddq=ddq_target, terms=dyn_terms
    )

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

    # Calculate the bootstrapping error
    M_boot = jax.lax.stop_gradient(inertia_boot(q, q_buff, dq, dq_buff))
    L_boot = jnp.mean((M_pred - M_boot) ** 2)

    return jnp.array([L_acc_ddq, L_acc_tau, L_energy, L_boot])
