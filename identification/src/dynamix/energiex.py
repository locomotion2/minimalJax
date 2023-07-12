from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

from flax.training import train_state as ts

from scipy.integrate import cumtrapz

import identification.identification_utils as utils


# TODO: run the matrix computations only when M or T are wanted
def build_energy_func(
    params: dict,
    train_state: ts.TrainState,
    settings: dict = None,
    output: str = "lagrangian",
) -> Callable:
    h_dim = settings["model_settings"]["h_dim"]
    num_dof = settings["system_settings"]["num_dof"]
    friction = settings["model_settings"]["friction"]
    model = settings["model_settings"]["model_pot"]

    @jax.jit
    def compiled_func(
        q: jnp.array, q_buff: jnp.array, dq: jnp.array, dq_buff: jnp.array
    ):
        # pack the state
        state = jnp.concatenate([q, q_buff, dq, dq_buff])

        # run the NN
        out = train_state.apply_fn(
            {"params": params},
            x=state,
            net_size=h_dim,
            num_dof=num_dof,
            friction=friction,
        )

        # build M
        @jax.jit
        def smooth(l_diag):
            return nn.softplus(l_diag + 2)

        l = out[:10]
        L = jnp.array(
            [
                [smooth(l[0]), 0, 0, 0],
                [l[1], smooth(l[2]), 0, 0],
                [l[3], l[4], smooth(l[5]), 0],
                [l[6], l[7], l[8], smooth(l[9])],
            ]
        )
        M = L @ jnp.transpose(L) + jnp.eye(num_dof, num_dof) * 0.001

        # calculate T
        T = 1 / 2 * jnp.transpose(dq) @ M @ dq

        # get V
        if model:
            q_rob, theta = jnp.split(q, 2)
            V = 1.75 / 2 * jnp.sum((theta - q_rob) ** 2)
        else:
            V = out[10]

        # get the friction
        k_dq = out[-4:]
        # k_dq = jnp.array([0.0205] * 4)

        # choose the output
        if output == "energies":
            return T, V
        elif output == "lagrangian":
            return T - V
        elif output == "potential":
            return V
        elif output == "kinetic":
            return T
        elif output == "friction":
            return k_dq
        elif output == "inertia":
            return M

    return compiled_func


# TODO: run the matrix computations only when M or T are wanted
def build_energy_func_model(
    params: dict,
    train_state: ts.TrainState,
    settings: dict = None,
    output: str = "lagrangian",
) -> Callable:
    h_dim = settings["model_settings"]["h_dim_model"]
    num_dof = settings["system_settings"]["num_dof"]
    friction = settings["model_settings"]["friction_model"]
    model = settings["model_settings"]["model_pot_model"]

    @jax.jit
    def compiled_func(
        q: jnp.array, q_buff: jnp.array, dq: jnp.array, dq_buff: jnp.array
    ):
        # pack the state
        state = jnp.concatenate([q, q_buff, dq, dq_buff])

        # run the NN
        out = train_state.apply_fn(
            {"params": params},
            x=state,
            net_size=h_dim,
            num_dof=num_dof,
            friction=friction,
        )

        # build M
        @jax.jit
        def smooth(l_diag):
            return nn.softplus(l_diag + 2)

        l = out[:10]
        L = jnp.array(
            [
                [smooth(l[0]), 0, 0, 0],
                [l[1], smooth(l[2]), 0, 0],
                [l[3], l[4], smooth(l[5]), 0],
                [l[6], l[7], l[8], smooth(l[9])],
            ]
        )
        M = L @ jnp.transpose(L) + jnp.eye(num_dof, num_dof) * 0.001

        # calculate T
        T = 1 / 2 * jnp.transpose(dq) @ M @ dq

        # get V
        if model:
            q_rob, theta = jnp.split(q, 2)
            V = 1.75 / 2 * jnp.sum((theta - q_rob) ** 2)
        else:
            V = out[10]

        # get the friction
        k_dq = out[-4:]
        # k_dq = jnp.array([0.0205] * 4)

        # choose the output
        if output == "energies":
            return T, V
        elif output == "lagrangian":
            return T - V
        elif output == "potential":
            return V
        elif output == "kinetic":
            return T
        elif output == "friction":
            return k_dq
        elif output == "inertia":
            return M

    return compiled_func


def build_energy_func_red(
    params: dict,
    train_state: ts.TrainState,
    settings: dict = None,
    output: str = "lagrangian",
) -> Callable:
    h_dim = settings["model_settings"]["h_dim"]
    num_dof = settings["system_settings"]["num_dof"]
    friction = settings["model_settings"]["friction"]
    model = settings["model_settings"]["model_pot"]

    @jax.jit
    def compiled_func(
        q: jnp.array, q_buff: jnp.array, dq: jnp.array, dq_buff: jnp.array
    ):
        # run the NN
        out = train_state.apply_fn(
            {"params": params}, x=q, net_size=h_dim, num_dof=num_dof, friction=friction
        )

        # build M
        @jax.jit
        def smooth(l_diag):
            return nn.softplus(l_diag + 2)

        l = out[:10]
        L = jnp.array(
            [
                [smooth(l[0]), 0, 0, 0],
                [l[1], smooth(l[2]), 0, 0],
                [l[3], l[4], smooth(l[5]), 0],
                [l[6], l[7], l[8], smooth(l[9])],
            ]
        )
        M = L @ jnp.transpose(L) + jnp.eye(num_dof, num_dof) * 0.001

        # calculate T
        T = 1 / 2 * jnp.transpose(dq) @ M @ dq

        # get V
        if model:
            q_rob, theta = jnp.split(q, 2)
            V = 1.75 / 2 * jnp.sum((theta - q_rob) ** 2)
        else:
            V = out[10]

        # get the friction
        k_dq = out[-4:]
        # k_dq = jnp.array([0.0205] * 4)

        # choose the output
        if output == "energies":
            return T, V
        elif output == "lagrangian":
            return T - V
        elif output == "potential":
            return V
        elif output == "kinetic":
            return T
        elif output == "friction":
            return k_dq
        elif output == "inertia":
            return M

    return compiled_func


def calculate_energies(calc_V_ana_vec, pow_input, pow_f, T_lnn, V_lnn, q):
    # Calculate measured energies
    H_mec = cumtrapz(jax.device_get(pow_input), dx=1 / 100, initial=0)
    H_loss = cumtrapz(jax.device_get(pow_f), dx=1 / 100, initial=0)
    H_ana = H_mec + H_loss

    # calculate NN energies
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn
    res_lnn = (T_lnn, V_lnn, H_lnn, L_lnn)

    # calculate analytical energies
    V_ana = calc_V_ana_vec(q)
    T_ana = H_ana - V_ana
    L_ana = T_ana - V_ana
    res_ana = (T_ana, V_ana, H_ana, L_ana)

    return res_ana, res_lnn, (H_mec, H_loss)


def calibrate_energies(settings, V_ana, V_lnn, T_ana, T_lnn, H_loss):
    # calibrate the NN output
    [alpha_V, beta_V], V_cal = utils.calibrate(V_ana, V_lnn)
    print(f"Potential factors: {alpha_V}, {beta_V}.")
    [alpha_T, beta_T], T_cal = utils.calibrate(T_ana, T_lnn)
    print(f"Kinetic factors: {alpha_T}, {beta_T}.")

    # calculate the new calibration
    H_cal = T_cal + V_cal - H_loss
    L_cal = T_cal - V_cal
    res_cal = (T_cal, V_cal, H_cal, L_cal)

    # saved coefficients after calibration
    coeff_data = jnp.array(settings["system_settings"]["calib_coeffs"])
    coef_V, coef_T = jnp.split(coeff_data, 2)

    # calculate final calibrated energies
    V_f = V_lnn * coef_V[0] + coef_V[1]
    T_f = T_lnn * coef_T[0]
    H_f = T_f + V_f - H_loss
    L_f = T_f - V_f
    res_final = (T_f, V_f, H_f, L_f)

    return res_cal, res_final


def test_calculations(
    batch, dyn_terms, split_tool: Callable, kinetic: Callable, potential: Callable
):
    # Unpack the terms
    state, ddq = batch
    q, q_buff, dq, dq_buff, tau = split_tool(state)
    _, _, k_dq = dyn_terms

    # Get the energies as the output of the NN
    T = kinetic(q, q_buff, dq, dq_buff)
    V = potential(q, q_buff, dq, dq_buff)

    # Get the derivatives on q and dq
    dT_q = jax.grad(kinetic, 0)(q, q_buff, dq, dq_buff)
    dV_q = jax.grad(potential, 0)(q, q_buff, dq, dq_buff)

    # Calculate powers
    pow_T = jnp.transpose(dq) @ dT_q
    pow_V = jnp.transpose(dq) @ dV_q
    pow_input = jnp.transpose(dq) @ tau
    pow_f = jnp.transpose(dq) @ (-k_dq * dq)
    # pow_f = jnp.transpose(dq) @ tau_f

    return T, V, (pow_V, pow_T, pow_input, pow_f)
