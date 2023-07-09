from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax.training import train_state as ts
from jax.experimental.ode import odeint
from flax import linen as nn

import identification_utils as utils

from scipy.integrate import cumtrapz

from systems import snake_utils


def energy_dyn_builder(state: jnp.array,
                       split_tool: Callable = None,
                       kinetic: Callable = None,
                       potential: Callable = None,
                       friction: Callable = None
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


def inertia_dyn_builder(state: jnp.array,
                        split_tool: Callable = None,
                        potential: Callable = None,
                        kinetic: Callable = None,
                        inertia: Callable = None,
                        friction: Callable = None
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


def eom_wrapped(x: jnp.array,
                t: jnp.array,
                params: jnp.array,
                q_buff: jnp.array = None,
                dq_buff: jnp.array = None,
                potential: Callable = None,
                inertia: Callable = None):
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
    tau_f = - k_dq * dq
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
    tau_f = - k_dq * dq
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
    tau_f = - k_dq * dq
    tau = tau_eff - tau_f

    return tau, tau_target, tau_f


def energy_wrapper(state: jnp.array,
                   split_tool: Callable,
                   kinetic: Callable,
                   potential: Callable
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


# TODO: Test if this is working
@partial(jax.jit, backend='cpu')
def solve_analytical(model, initial_state: jnp.array, times: jnp.array):
    return odeint(model.f_analytical, initial_state, t=times, rtol=1e-13, atol=1e-13)


def solve_eom(initial_state, eom_compiled, time_span, params):
    @partial(jax.jit, backend='cpu')
    # @jax.jit
    def f(initial_state):
        dx = eom_compiled(x=initial_state, t=0, params=params)

        final_state = initial_state + dx * time_span[-1]

        return [initial_state, final_state]

    return f(initial_state)


# @jax.jit
def simulate(state, tau, buffer_length, sys_utils, num_dof,
             samples_num, eom_prepared, split_tool,
             data_formatted):
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
        eom_compiled = jax.jit(partial(eom_prepared,
                               q_buff=q_buff_input,
                               dq_buff=dq_buff_input))

        # calculate the enchilada
        out = solve_eom(x_input,
                        eom_compiled,
                        jnp.array([0, 1 / 100]),
                        tau_cur)
        # out = odeint(eom_compiled,
        #              x_input,
        #              jnp.array([0, 1 / 100]),
        #              (tau_cur,))

        # update the values
        x_input = out[-1]
        q_cur, dq_cur = jnp.split(x_input, 2)
        (q_buff_input, dq_buff_input), \
            (q_buff_wide, dq_buff_wide) = utils.format_state_sim(q_cur,
                                                                 q_buff_wide,
                                                                 dq_cur,
                                                                 dq_buff_wide)

        # save in vectors
        q_sim = jnp.append(q_sim, jnp.array([q_cur]), axis=0)
        dq_sim = jnp.append(dq_sim, jnp.array([dq_cur]), axis=0)
        # dq_sim = jnp.append(dq_sim, jnp.array([dq_cur]), axis=0)

    return q_sim, dq_sim


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
    model = settings['model_pot']

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
        @jax.jit
        def smooth(l_diag):
            return nn.softplus(l_diag + 2)

        l = out[:10]
        L = jnp.array([[smooth(l[0]), 0, 0, 0],
                       [l[1], smooth(l[2]), 0, 0],
                       [l[3], l[4], smooth(l[5]), 0],
                       [l[6], l[7], l[8], smooth(l[9])]])
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


# TODO: run the matrix computations only when M or T are wanted
def energy_func_model(params: dict, train_state: ts.TrainState,
                      settings: dict = None,
                      output: str = 'lagrangian') -> Callable:
    h_dim = settings['h_dim_model']
    num_dof = settings['num_dof']
    friction = settings['friction_model']
    model = settings['model_pot_model']

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
        @jax.jit
        def smooth(l_diag):
            return nn.softplus(l_diag + 2)

        l = out[:10]
        L = jnp.array([[smooth(l[0]), 0, 0, 0],
                       [l[1], smooth(l[2]), 0, 0],
                       [l[3], l[4], smooth(l[5]), 0],
                       [l[6], l[7], l[8], smooth(l[9])]])
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


def energy_func_red(params: dict, train_state: ts.TrainState,
                    settings: dict = None,
                    output: str = 'lagrangian') -> Callable:
    h_dim = settings['h_dim']
    num_dof = settings['num_dof']
    friction = settings['friction']
    model = settings['model_pot']

    @jax.jit
    def compiled_func(q: jnp.array, q_buff: jnp.array,
                      dq: jnp.array, dq_buff: jnp.array):

        # run the NN
        out = train_state.apply_fn({'params': params},
                                   x=q,
                                   net_size=h_dim,
                                   num_dof=num_dof,
                                   friction=friction
                                   )

        # build M
        @jax.jit
        def smooth(l_diag):
            return nn.softplus(l_diag + 2)

        l = out[:10]
        L = jnp.array([[smooth(l[0]), 0, 0, 0],
                       [l[1], smooth(l[2]), 0, 0],
                       [l[3], l[4], smooth(l[5]), 0],
                       [l[6], l[7], l[8], smooth(l[9])]])
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
    pow_f = jnp.transpose(dq) @ (- k_dq * dq)
    # pow_f = jnp.transpose(dq) @ tau_f

    return T, V, (pow_V, pow_T, pow_input, pow_f)

# TODO: Change thuis function to take any time of system as input
def build_dynamics(settings, params, train_state):
    # Create basic building blocks
    kinetic = energy_func_model(
        params, train_state, settings=settings, output="kinetic"
    )
    potential = energy_func_model(
        params, train_state, settings=settings, output="potential"
    )
    friction = energy_func_model(
        params, train_state, settings=settings, output="friction"
    )
    inertia = energy_func_model(
        params, train_state, settings=settings, output="inertia"
    )

    split_tool = snake_utils.build_split_tool(settings["buffer_length"])

    # Create compiled dynamics
    dyn_builder = partial(
        inertia_dyn_builder,
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
                test_calculations,
                split_tool=split_tool,
                kinetic=kinetic,
                potential=potential,
            )
        )
    )

    # build the integrable EOMs
    eom_compiled = partial(eom_wrapped, potential=potential, inertia=inertia)

    return dyn_builder_compiled, energy_calcs, eom_compiled, split_tool

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
    coef_V, coef_T = jnp.split(settings["calib_coefs"], 2)

    # calculate final calibrated energies
    V_f = V_lnn * coef_V[0] + coef_V[1]
    T_f = T_lnn * coef_T[0]
    H_f = T_f + V_f - H_loss
    L_f = T_f - V_f
    res_final = (T_f, V_cal, H_f, L_f)

    return res_cal, res_final

def build_loss(settings: dict) -> (Callable, Callable):
    # get the weights
    weights = settings['loss_weights']
    weights_model = settings['loss_weights_model']
    weights_red = settings['loss_weights_red']
    sys_utils = settings['sys_utils']
    buff_len = settings['buffer_length']
    split_tool = sys_utils.build_split_tool(buff_len)

    @jax.jit
    def loss_energies(params: dict,
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

    @jax.jit
    def loss_model(params: dict,
                   train_state,
                   batch: (jnp.array, jnp.array)) -> jnp.array:
        # Build dynamics
        potential_func = energy_func_model(params, train_state, settings=settings,
                                           output='potential')
        inertia_func = energy_func_model(params, train_state, settings=settings,
                                         output='inertia')
        friction_func = energy_func_model(params, train_state, settings=settings,
                                          output='friction')
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
            return jnp.transpose(jnp.array(weights_model)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    @jax.jit
    def loss_red(params: dict,
                 train_state,
                 train_state_red,
                 batch: (jnp.array, jnp.array)) -> jnp.array:
        # Build dynamics
        potential_func = energy_func_red(params, train_state_red, settings=settings,
                                         output='potential')
        friction_func = energy_func_red(params, train_state_red, settings=settings,
                                        output='friction')
        inertia_func = energy_func_red(params, train_state_red, settings=settings,
                                       output='inertia')
        inertia_func_boot = energy_func(train_state.params, train_state,
                                        settings=settings,
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
            losses = loss_sample_red(split_tool,
                                     dyn_builder_compiled,
                                     inertia_func_boot,
                                     sample)
            return jnp.transpose(jnp.array(weights_red)) @ losses

        vector_loss = jax.vmap(weighted_loss)
        return jnp.mean(vector_loss(batch))

    loss = None
    if settings['goal'] == 'energy':
        loss = loss_energies
    elif settings['goal'] == 'model':
        loss = loss_model

    return loss, loss_red


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


@partial(jax.jit, static_argnums=[0, 1])
def loss_sample_model(split_tool: Callable,
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
    # L_acc_tau = 0

    # Calculate temporal energy error
    pow_pred = jnp.transpose(dq) @ tau_pred
    pow_target = jnp.transpose(dq) @ tau_target
    # L_energy = jnp.mean(((pow_pred - pow_target) / jnp.abs(pow_target)) ** 2)
    L_energy = jnp.mean((pow_pred - pow_target) ** 2)
    # L_energy = 0

    return jnp.array([L_acc_ddq, L_acc_tau, L_energy])


@partial(jax.jit, static_argnums=[0, 1, 2])
def loss_sample_red(split_tool: Callable,
                    dyn_builder: Callable,
                    inertia_boot: Callable,
                    batch: (jnp.array, jnp.array)
                    ) -> jnp.array:
    # Unpack training data
    state, ddq_target = batch
    q, q_buff, dq, dq_buff, _ = split_tool(state)
    dyn_terms = dyn_builder(state)
    _, (M_pred, _, _), _ = dyn_terms

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

    # Calculate the bootstrapping error
    M_boot = jax.lax.stop_gradient(inertia_boot(q, q_buff, dq, dq_buff))
    L_boot = jnp.mean((M_pred - M_boot) ** 2)

    return jnp.array([L_acc_ddq, L_acc_tau, L_energy, L_boot])
