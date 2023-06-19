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
        size = 64 * 10
        x = self.layer(x, features=size)
        x_out = nn.Dense(features=2 + 4)(x)
        return x_out

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
    def friction_simple(q, dq):
        return friction(q, q_buff, dq, dq_buff)

    # build dynamic functions
    def inertia_simple(q, dq):
        return jax.hessian(kinetic, 2)(q, q_buff, dq, dq_buff)

    def coriolis(q, dq):
        return jax.jacobian(jax.jacobian(kinetic, 2), 0)(q, q_buff, dq, dq_buff) - \
            1 / 2 * jnp.transpose(dq) @ \
            jax.jacobian(jax.hessian(kinetic, 2), 0)(q, q_buff, dq, dq_buff)

    gamma = 1.75
    K_x = np.diag([gamma, gamma])
    K = np.bmat([[K_x, -K_x], [-K_x, K_x]])
    K = jnp.array(K)

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
    k_f = friction(q, dq)

    return q, dq, tau, M, C, K, g, k_f


def simplified_dynamic_matrices(state: jnp.array,
                                dynamics: Callable
                                ) -> jnp.array:
    _, _, _, M, C, K, g, k_f = dynamics(state)
    # inertias
    M_rob = M[:2, :2]
    B = M[2:, 2:]

    # coriolis
    C_rob = C[:2, :2]

    # potentials
    K_red = K[:2, :2]
    g_rob = g[:2]

    # frictions
    k_f_rob = k_f[:2]
    k_f_mot = k_f[2:]

    return (M_rob, C_rob, g_rob, k_f_rob), (B, k_f_mot), K_red


def equation_of_motion(state: jnp.array,
                       dynamics: Callable
                       ) -> jnp.array:
    # q, dq, tau, M, C, g, k_f = dynamics(state)
    q, dq, tau, M, C, K, g, k_f = dynamics(state)

    # account for friction
    tau_eff = tau - k_f * dq

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
    q, dq, tau_target, M, C, K, g, k_f = dynamics(state)

    # foward dyns.
    tau_eff = M @ ddq + C @ dq + K @ q + g

    # account for friction
    tau = tau_eff + k_f * dq

    return tau, tau_target


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


def learned_lagrangian(params: dict, train_state: ts.TrainState,
                       output: str = 'lagrangian') -> Callable:
    @jax.jit
    def lagrangian(q: jnp.array, q_buff: jnp.array, q_dot: jnp.array, dq_buff:
    jnp.array):
        state = jnp.concatenate([q, q_buff, q_dot, dq_buff])
        out = train_state.apply_fn({'params': params}, x=state)
        T = out[0]
        V = out[1]
        k_f = out[-4:] ** 2
        if output == 'energies':
            return T, V
        elif output == 'lagrangian':
            return T - V
        elif output == 'potential':
            return V
        elif output == 'kinetic':
            return T
        elif output == 'friction':
            return k_f

    return lagrangian


@jax.jit
def learned_energies(state: jnp.array, params: dict = None,
                     train_state: ts.TrainState = None):
    # Split the state variables
    # q, q_dot, _ = jnp.split(state, 3)
    q, q_buff, dq, dq_buff, tau = split_state(state, 10)

    # Get the energies as the output of the NN
    T, V = learned_lagrangian(params, train_state, output='energies')(q, q_buff,
                                                                      dq, dq_buff)

    # Reconstruct the kin.energy from its deriv.
    M = jax.hessian(learned_lagrangian(params, train_state, output='kinetic'), 2)
    M = M(q, q_buff, dq, dq_buff)
    T_rec = 1 / 2 * jnp.transpose(dq) @ M @ dq

    # Get the derivative on q_dot from the potential energy
    V_dot = jax.grad(learned_lagrangian(params, train_state, output='potential'), 1)
    V_dot = V_dot(q, q_buff, dq, dq_buff)

    # Get the friction torques
    # tau_f = learned_lagrangian(params, train_state, output='friction')(q, q_dot)

    return T, V, T_rec, V_dot, M


@jax.jit
def loss(params: dict, train_state: ts.TrainState,
         batch: (jnp.array, jnp.array)) -> jnp.array:
    # Unpack training data
    state, qdd_target = batch

    # Calculate energies and their derivatives
    T, V, T_rec, V_dot, M = jax.vmap(partial(learned_energies, params=params,
                                             train_state=train_state))(state)

    # Calculate total energy (Hamiltonian)
    # H = T + V

    # Build dynamics
    kinetic_func = learned_lagrangian(params, train_state, output='kinetic')
    potential_func = learned_lagrangian(params, train_state, output='potential')
    friction_func = learned_lagrangian(params, train_state, output='friction')
    compiled_dynamics = partial(calc_dynamics,
                                kinetic=kinetic_func,
                                potential=potential_func,
                                friction=friction_func)
    compiled_dyn_wrapper = jax.jit(partial(dynamic_matrices,
                                           dynamics=compiled_dynamics))
    inv_dyn = jax.vmap(jax.jit(partial(equation_of_motion,
                                       dynamics=compiled_dyn_wrapper)))
    for_dyn_single = jax.jit(partial(forward_dynamics,
                                     dynamics=compiled_dyn_wrapper))

    # wrap for_dyn_single to handle data format
    def for_dyn_wrapped(data_point):
        state, ddq = data_point
        ddq = ddq[4:8]
        return for_dyn_single(ddq=ddq, state=state)

    for_dyn = jax.vmap(for_dyn_wrapped)

    # Predict joint accelerations and calculate error
    qdd_pred = inv_dyn(state)
    L_acc_qdd = jnp.mean((qdd_pred - qdd_target) ** 2)
    # L_acc = L_acc_qdd

    # Predict the torques and calculate error
    tau_prediction, tau_target = for_dyn(batch)
    L_acc_tau = jnp.mean((tau_prediction - tau_target) ** 2)
    L_acc = (L_acc_qdd + L_acc_tau * 100000) / 2
    # L_acc = L_acc_tau

    # Impose energy conservation
    # L_con = jnp.mean(H ** 2)

    # Impose clean derivative
    L_kin = jnp.mean((T - T_rec) ** 2)

    # Imppose symetry of inertia matrix
    def L_mass_func(M):
        top_M = jnp.triu(M, k=1)
        bot_M = jnp.tril(M, k=-1)
        diag_M = jnp.diagonal(M)
        diag_M = jnp.clip(diag_M, a_min=None, a_max=0)
        return jnp.mean((top_M - jnp.transpose(bot_M)) ** 2) + jnp.sum(diag_M ** 2)

    L_mass = jnp.mean(jax.vmap(L_mass_func)(M))

    # Impose independence form q_dot on V due to mechanical system
    L_pot = jnp.mean(V_dot ** 2)

    # Impose small frictions
    # L_f = jnp.mean(tau_f ** 2)

    return L_acc + 1000 * (L_kin + L_pot + L_mass)


# TODO: This function needs to be unified with the previous one somehow
@jax.jit
def loss_sample(pair, params=None, train_state=None):
    state, targets = pair

    # calculate energies
    T, V, T_rec, V_dot, M = partial(learned_energies, params=params,
                                    train_state=train_state)(state)
    H = T_rec + V

    # predict joint accelerations
    eqs_motion = partial(equation_of_motion,
                         kinetic=learned_lagrangian(params, train_state,
                                                    output='kinetic'),
                         potential=learned_lagrangian(params, train_state,
                                                      output='potential'))
    preds = eqs_motion(state)
    L_acc = jnp.mean((preds - targets) ** 2)
    # print(L_acc.shape)

    # impose energy conservation
    L_con = H ** 2

    # impose clean derivative
    L_kin = (T - T_rec) ** 2

    # impose independence form q_dot on V due to mechanical system
    L_pot = jnp.mean(V_dot ** 2)

    def L_mass_func(M):
        top_M = jnp.triu(M, k=1)
        bot_M = jnp.tril(M, k=-1)
        diag_M = jnp.diag(M)
        diag_M = jnp.clip(diag_M, a_min=None, a_max=0)
        return jnp.mean((top_M - jnp.transpose(bot_M)) ** 2) + jnp.mean(diag_M ** 2)

    L_mass = L_mass_func(M)

    L_total = L_acc + 1000 * (L_kin + L_pot + L_mass)
    return L_total, L_acc, 1000 * (L_kin + L_pot + L_mass)


def create_train_state(settings: dict, learning_rate_fn: Callable, params: dict =
None) -> ts.TrainState:
    # Unpack settings
    key = jax.random.PRNGKey(settings['seed'])
    buffer_length = settings['buffer_length']
    num_dof = settings['num_dof']

    # Create network
    network = MLP()

    # If available load the parameters
    if params is None:
        input_size = (2 * num_dof * buffer_length,)
        params = network.init(key, jax.random.normal(key, input_size))['params']

    # Set up the optimizer and bundle everything into a train state
    adam_opt = optax.adamw(learning_rate=learning_rate_fn)
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
        end = index * buffer_length_max + buffer_length
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
    dq_0 = jnp.array([sample[4 * buffer_length_max],
                      sample[5 * buffer_length_max],
                      sample[6 * buffer_length_max],
                      sample[7 * buffer_length_max]])
    ddq_0 = jnp.array(sample[index_end: index_end + 4])
    tau = jnp.array(sample[index_end + 4: index_end + 8])

    # build state variables
    state = jnp.concatenate([q_n, dq_n])
    dstate_0 = jnp.concatenate([dq_0, ddq_0])
    state_ext = jnp.concatenate([state, tau])

    return state_ext, dstate_0


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
    samples_train = int(samples_total * 0.8)
    samples_validation = int(samples_total * 0.1)
    samples_test = int(samples_total * 0.1)

    # query commands
    # query_sample_test = f'SELECT * FROM your_table ORDER BY RANDOM() LIMIT ' \
    #                     f'{samples_train + samples_validation},{samples_test} ' \
    #                     f'LIMIT {data_size}'

    format_samples = jax.vmap(partial(format_sample,
                                      buffer_length=buffer_length,
                                      buffer_length_max=buffer_length_max))

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
