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

from Lagranx.src import dpend_model_arne as model


class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        size = 64 * 10
        x = self.layer(x, features=size)
        x_out = nn.Dense(features=2)(x)
        return x_out

    def layer(self, x: jnp.array, features: int = 128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.softplus(x)
        return x


def equation_of_motion(state: jnp.array, kinetic: Callable = None, potential: Callable = None) -> jnp.array:
    q, q_t = jnp.split(state, 2)

    @jax.jit
    def lagrangian(q, q_t):
        return kinetic(q, q_t) - potential(q, q_t)

    M = jax.hessian(kinetic, 1)(q, q_t)
    q_tt = jnp.linalg.pinv(M) @ (jax.grad(lagrangian, 0)(q, q_t) - jax.jacobian(jax.jacobian(kinetic, 1), 0)(q, q_t) @ q_t)

    return jnp.concatenate([q_t, q_tt])


@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state: jnp.array, times: jnp.array):
    return odeint(model.f_analytical, initial_state, t=times, rtol=1e-8, atol=1e-9)


def learned_lagrangian(params: dict, train_state: ts.TrainState, output: str = 'lagrangian') -> Callable:
    @jax.jit
    def lagrangian(q: jnp.array, q_dot: jnp.array):
        state = jnp.concatenate([q, q_dot])
        out = train_state.apply_fn({'params': params}, x=state)
        T = out[0]
        V = out[1]
        if output == 'energies':
            return T, V
        elif output == 'lagrangian':
            return T - V
        elif output == 'potential':
            return V
        elif output == 'kinetic':
            return T

    return lagrangian

@jax.jit
def learned_energies(state: jnp.array, params: dict = None, train_state: ts.TrainState = None):
    # Split the state variables
    q, q_dot = jnp.split(state, 2)

    # Get the energies as the output of the NN
    T, V = learned_lagrangian(params, train_state, output='energies')(q, q_dot)

    # Reconstruct the kin.energy from its deriv.
    M = jax.hessian(learned_lagrangian(params, train_state, output='kinetic'), 1)(q, q_dot)
    T_rec = 1 / 2 * jnp.transpose(q_dot) @ M @ q_dot

    # Get the derivative on q_dot from the potential energy
    V_dot = jax.grad(learned_lagrangian(params, train_state, output='potential'), 1)(q, q_dot)

    return T, V, T_rec, V_dot, M


@jax.jit
def loss(params: dict, train_state: ts.TrainState, batch: (jnp.array, jnp.array)) -> jnp.array:
    # Unpack training data
    state, q_ddot_target = batch

    # Calculate energies and their derivatives
    T, V, T_rec, V_dot, M = jax.vmap(partial(learned_energies, params=params,
                                             train_state=train_state))(state)

    # Calculate total energy (Hamiltonian)
    # H = T + V

    # Predict joint accelerations and calculate error
    eqs_motion = jax.vmap(partial(equation_of_motion,
                                  kinetic=learned_lagrangian(params, train_state, output='kinetic'),
                                  potential=learned_lagrangian(params, train_state, output='potential')))
    q_ddot_prediction = eqs_motion(state)
    L_acc = jnp.mean((q_ddot_prediction - q_ddot_target) ** 2)

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

    return L_acc + 1000 * (L_kin + L_pot + L_mass)


# TODO: This function needs to be unified with the previous one somehow
@jax.jit
def loss_sample(pair, params=None, train_state=None):
    state, targets = pair

    # calculate energies
    T, V, T_rec, V_dot, M = partial(learned_energies, params=params, train_state=train_state)(state)
    H = T_rec + V

    # predict joint accelerations
    eqs_motion = partial(equation_of_motion,
                                  kinetic=learned_lagrangian(params, train_state, output='kinetic'),
                                  potential=learned_lagrangian(params, train_state, output='potential'))
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
    return L_total, L_acc, L_con, 10 * L_kin, 10 * L_pot


def create_train_state(key: int, learning_rate_fn: Callable, params: dict = None) -> \
        ts.TrainState:
    # Create network
    network = MLP()

    # If available load the parameters
    if params is None:
        params = network.init(key, jax.random.normal(key, (4,)))['params']

    # Set up the optipizer and bundle everything into a train state
    adam_opt = optax.adamw(learning_rate=learning_rate_fn)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)


@partial(jax.jit, static_argnums=2)
def train_step(train_state: ts.TrainState, batch: (jnp.array, jnp.array), learning_rate_fn: Callable) -> (ts.TrainState, dict):

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


def run_training(train_state: ts.TrainState, dataloader: Callable, settings: dict, run: Run) -> (dict, tuple):
    # Unpack Settings
    test_every = settings['test_every']
    num_batches = settings['num_batches']
    num_minibatches = settings['num_minibatches']
    num_epochs = settings['num_epochs']
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
        while epoch < num_epochs:
            # Sample key for each epoch
            random_key = jax.random.PRNGKey(settings['seed'] + epoch)

            # Split training data into minibatches
            batch_train, batch_test = dataloader(random_key)
            x_train, xt_train = batch_train
            x_train_minibatches = jnp.split(x_train, num_minibatches)
            xt_train_minibatches = jnp.split(xt_train, num_minibatches)

            # Train model on each batch
            epoch_loss = 0
            train_metrics = None
            for batch in range(num_batches):
                train_loss = 0
                for minibatch in range(num_minibatches):
                    minibatch_current = (x_train_minibatches[minibatch], xt_train_minibatches[minibatch])
                    train_state, train_metrics = train_step(train_state, minibatch_current, lr_func)
                    train_loss += train_metrics['loss'] / num_minibatches

                # When a batch is done
                epoch_loss += train_loss / num_batches

                # Check for the best params per batch
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = copy(train_state.params)

            # Evaluate model with the test data
            test_loss = eval_step(train_state, batch_test)['loss']

            if epoch_loss > epoch_loss_last * early_stopping_gain:
                print(f'Early stopping! Epoch loss: {epoch_loss}. Now resetting.')
                settings['seed'] += 10
                train_state = create_train_state(settings['seed'], lr_func, params=best_params)
                epoch = 0
                continue

            # Record train and test losses
            train_losses.append(epoch_loss)
            test_losses.append(test_loss)
            run.track(epoch, name='epoch')
            run.track(epoch_loss, name='epoch_loss')
            run.track(test_loss, name='test_loss')
            run.track(train_metrics['learning_rate'], name='learning_rate')

            # Output progress every 'test_every' epochs
            if epoch % test_every == 0:
                print(f"Epoch={epoch}, train={epoch_loss:.4f}, test={test_loss:.4f}, lr={train_metrics['learning_rate']:.10f}")

            # Update epoch and record the last loss
            epoch += 1
            epoch_loss_last = epoch_loss

    except KeyboardInterrupt:
        # Save params from model
        print('Terminating learning!')

    return best_params, (train_losses, test_losses)


def generate_trajectory_data(x0: jnp.array, t_window: jnp.array, section_num: int, key: int) -> (jnp.array, jnp.array, jnp.array):
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

    print(f"Generation successful! Ranges: {jnp.amax(x_traj, axis=0)}, {jnp.amin(x_traj, axis=0)}")
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


def build_simple_dataloader(batch_train: tuple, batch_test: tuple, settings: dict) -> Callable:
    def dataloader(key):
        return batch_train, batch_test

    return dataloader


def build_general_dataloader(batch_train: tuple, batch_test: tuple, settings: dict) -> Callable:
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
                              (jax.random.uniform(key + 10, (data_size, 1)) - 0.5) * 10 * 2,
                              (jax.random.uniform(key + 20, (data_size, 1)) - 0.5) * 10 * 4], axis=1)
        y0 = jax.vmap(model.normalize)(y0)

        return (y0, eqs_motion(y0)), batch_test

    return dataloader


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
