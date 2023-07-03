from copy import deepcopy as copy
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from aim import Run
from flax.training import train_state as ts

import stable_baselines3.common.save_util as loader

from src import dpendulum_utils
from src import snake_utils


def choose_data_loader(settings: dict):
    system = settings['system']
    data_source = settings['data_source']

    batch_train, batch_test, dataloader = None, None, None
    if system == 'snake':
        dataloader = snake_utils.build_database_dataloader_eff(settings)
    elif system == 'dpend':
        if data_source == 'dummy':
            batch_train, batch_test = loader.load_from_pkl(path=settings['data_dir'],
                                                           verbose=1)
            dataloader = build_dummy_dataloader(batch_train, batch_test, settings)
        elif data_source == 'pickle':  # TODO: Save in sqlite3
            batch_train, batch_test = loader.load_from_pkl(path=settings['data_dir'],
                                                           verbose=1)
            dataloader = dpendulum_utils.build_random_data_dataloader(batch_train,
                                                                      batch_test,
                                                                      settings)

    return dataloader


def create_train_state(settings: dict,
                       learning_rate_fn: Callable,
                       params: dict = None) -> ts.TrainState:
    # Unpack settings
    key = jax.random.PRNGKey(settings['seed'])
    buffer_length = settings['buffer_length']
    we_param = settings['weight_decay']
    sys_utils = settings['sys_utils']

    # network params
    h_dim = settings['h_dim']
    num_dof = settings['num_dof']
    friction = settings['friction']

    # Create network
    network = sys_utils.DeLaNN()

    # If available load the parameters
    if params is None:
        input_size = (2 * num_dof * buffer_length,)
        params = network.init(key,
                              jax.random.normal(key, input_size),
                              friction=friction,
                              net_size=h_dim,
                              num_dof=num_dof)['params']

    # Set up the optimizer and bundle everything into a train state
    adam_opt = optax.adamw(learning_rate=learning_rate_fn, weight_decay=we_param)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)


@partial(jax.jit, static_argnums=[2, 3])
def train_step(train_state: ts.TrainState,
               batch: (jnp.array, jnp.array),
               loss_func: Callable,
               learning_rate_fn: Callable) -> (ts.TrainState, dict):
    # Creates compiled function that contains the batch data
    @jax.jit
    def loss_fn(params: dict):
        return loss_func(params, train_state, batch)

    # Update the model
    loss_value, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    # Build the result metrics
    metrics = {'learning_rate': learning_rate_fn(train_state.step), 'loss': loss_value}

    return train_state, metrics


@partial(jax.jit, static_argnums=2)
def eval_step(train_state: ts.TrainState,
              test_batch: (jnp.array, jnp.array),
              loss_func: Callable) -> dict:
    loss_value = loss_func(train_state.params, train_state, test_batch)
    return {'loss': loss_value}


def run_training(train_state: ts.TrainState,
                 dataloader: Callable,
                 loss_func: Callable,
                 settings: dict,
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

    # build the compiled train_step
    # train_step = build_train_step(train_state, loss_func, lr_func)

    try:
        epoch_loss_last = np.inf
        epoch = 0
        x_train_large = None
        xt_train_large = None
        x_test_large = None
        xt_test_large = None
        while epoch < num_epochs:
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

            # Split training data into mini-batches
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
                                                            minibatch_current,
                                                            loss_func,
                                                            lr_func)
                    train_loss += train_metrics['loss'] / num_minibatches

                # When a batch is done
                epoch_loss += train_loss / num_batches

                # Evaluate model with the test data
                test_loss = eval_step(train_state, batch_test, loss_func)['loss']
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
        print('Terminating learning!')

    return best_params, (train_losses, test_losses)


def build_dummy_dataloader(batch_train: tuple,
                           batch_test: tuple,
                           settings: dict) -> Callable:
    def dataloader(key):
        return batch_train, batch_test

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
