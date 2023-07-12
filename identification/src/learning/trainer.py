from copy import deepcopy as copy
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import optax
from aim import Run
from flax.training import train_state as ts

import stable_baselines3.common.save_util as loader

from identification.systems import dpendulum_utils, snake_utils


def choose_data_loader(settings: dict):
    settings_system = settings['system_settings']
    settings_data = settings['data_settings']

    system = settings_system['system']
    data_source = settings_data['data_source']

    batch_train, batch_test, dataloader = None, None, None
    if system == 'snake':
        dataloader = snake_utils.build_database_dataloader_eff(settings)
    elif system == 'dpend':
        if data_source == 'dummy':
            batch_train, batch_test = loader.load_from_pkl(path=settings_data['data_dir'],
                                                           verbose=1)
            dataloader = build_dummy_dataloader(batch_train, batch_test, settings)
        elif data_source == 'pickle':  # TODO: Save in sqlite3
            batch_train, batch_test = loader.load_from_pkl(path=settings_data['data_dir'],
                                                           verbose=1)
            dataloader = dpendulum_utils.build_random_data_dataloader(batch_train,
                                                                      batch_test,
                                                                      settings)

    return dataloader


def create_train_state(goal: str,
                       settings: dict,
                       learning_rate_fn: Callable,
                       params: dict = None) -> ts.TrainState:
    # Unpack settings
    settings_training = settings['training_settings']
    settings_model = settings['model_settings']
    settings_system = settings['system_settings']

    key = jax.random.PRNGKey(settings_training['seed'])
    we_param = settings_training['weight_decay']

    sys_utils = settings_system['sys_utils']
    num_dof = settings_system['num_dof']

    network = None
    h_dim = None
    friction = None
    buffer_length = settings_model['buffer_length']
    if goal == "energy":
        h_dim = settings_model['h_dim']
        friction = settings_model['friction']

        network = sys_utils.DeLaNN()

    elif goal == "model":
        h_dim = settings_model['h_dim_model']
        friction = settings_model['friction_model']

        network = sys_utils.black_box_model()

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

# def create_train_state_PowNN(settings: dict,
#                               learning_rate_fn: Callable,
#                               params: dict = None) -> ts.TrainState:
#     # Unpack settings
#     settings_training = settings['training_settings']
#     settings_model = settings['model_settings']
#     settings_system = settings['system_settings']
#
#     key = jax.random.PRNGKey(settings_training['seed'])
#     we_param = settings_training['weight_decay']
#
#     h_dim = settings_model['h_dim_model']
#     friction = settings_model['friction_model']
#     buffer_length = settings_model['buffer_length']
#
#     sys_utils = settings_system['sys_utils']
#     num_dof = settings_system['num_dof']
#
#     # Create network
#
#
#     # If available load the parameters
#     if params is None:
#         input_size = (2 * num_dof * buffer_length,)
#         params = network.init(key,
#                               jax.random.normal(key, input_size),
#                               friction=friction,
#                               net_size=h_dim,
#                               num_dof=num_dof)['params']
#
#     # Set up the optimizer and bundle everything into a train state
#     adam_opt = optax.adamw(learning_rate=learning_rate_fn, weight_decay=we_param)
#     return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)


def create_train_state_red(settings: dict,
                           learning_rate_fn: Callable,
                           params: dict = None) -> ts.TrainState:
    # Unpack settings
    settings_training = settings['training_settings']
    settings_model = settings['model_settings']
    settings_system = settings['system_settings']

    key = jax.random.PRNGKey(settings_training['seed'])
    we_param = settings_training['weight_decay']

    h_dim = settings_model['h_dim']
    friction = settings_model['friction']

    sys_utils = settings_system['sys_utils']
    num_dof = settings_system['num_dof']

    # Create network
    network = sys_utils.DeLaNN_RED()

    # If available load the parameters
    if params is None:
        input_size = (num_dof,)
        params = network.init(key,
                              jax.random.normal(key, input_size),
                              friction=friction,
                              net_size=h_dim,
                              num_dof=num_dof)['params']

    # Set up the optimizer and bundle everything into a train state
    adam_opt = optax.adamw(learning_rate=learning_rate_fn, weight_decay=we_param)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)


@partial(jax.jit, static_argnums=[0, 4, 5, 6])
def train_step(bootstrapping: bool,
               train_state: ts.TrainState,
               train_state_red: ts.TrainState,
               batch: (jnp.array, jnp.array),
               loss_func: Callable,
               loss_func_red: Callable,
               learning_rate_fn: Callable) -> (ts.TrainState, ts.TrainState, dict):
    # Creates compiled function that contains the batch data
    @jax.jit
    def loss_fn(params: dict):
        return loss_func(params, train_state, batch)

    @jax.jit
    def loss_fn_red(params: dict):
        return loss_func_red(params, train_state, train_state_red, batch)

    # Update the large model
    loss_value_red = 0
    if not bootstrapping:
        loss_value, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
    else:
        loss_value = loss_fn(train_state.params)

        # Update the small model
        loss_value_red, grads_red = jax.value_and_grad(loss_fn_red)(
            train_state_red.params)
        train_state_red = train_state_red.apply_gradients(grads=grads_red)

    # Build the result metrics
    metrics = {'learning_rate': learning_rate_fn(train_state.step),
               'loss': loss_value,
               'loss_red': loss_value_red}

    return train_state, train_state_red, metrics


@partial(jax.jit, static_argnums=[2])
def eval_step(train_state: ts.TrainState,
              test_batch: (jnp.array, jnp.array),
              loss_func: Callable) -> dict:

    loss_value = loss_func(train_state.params, train_state, test_batch)
    loss_value_red = 0
    # if bootstrapping:
    #     loss_value_red = loss_func_red(train_state_red.params,
    #                                    train_state,
    #                                    train_state_red,
    #                                    test_batch)
    return {'loss': loss_value, 'loss_red': loss_value_red}


def run_training(train_state: ts.TrainState,
                 train_state_red: ts.TrainState,
                 dataloader: Callable,
                 loss_func: Callable,
                 loss_func_red: Callable,
                 settings: dict,
                 run: Run) -> (dict, tuple):
    # Unpack Settings
    settings_training = settings['training_settings']
    settings_data = settings['data_settings']

    test_every = settings_training['test_every']
    num_batches = settings_training['num_batches']
    num_minibatches = settings_training['num_minibatches']
    num_epochs = settings_training['num_epochs']
    lr_func = settings_training['lr_func']
    early_stopping_gain = settings_training['es_gain']

    num_skips = settings_data['eff_datasampling']

    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    best_loss = np.inf
    best_params = None

    train_losses_red = []
    test_losses_red = []
    best_loss_red = np.inf
    best_params_red = None

    try:
        epoch_loss_last = np.inf
        epoch_loss_last_red = np.inf
        epoch = 0
        x_train_large = None
        xt_train_large = None
        x_test_large = None
        xt_test_large = None
        while epoch < num_epochs:
            # Get new samples for the next num_skips epochs
            if epoch % num_skips == 0:
                batch_train_large, batch_test_large = dataloader(settings_training['seed'] +
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
            epoch_loss_red = 0
            epoch_test_loss_red = 0
            train_metrics = None
            for batch in range(num_batches):
                train_loss = 0
                train_loss_red = 0
                for minibatch in range(num_minibatches):
                    minibatch_current = (
                        x_train_minibatches[minibatch], xt_train_minibatches[minibatch])
                    train_state, train_state_red, train_metrics = train_step(
                        settings_training['bootstrapping'],
                        train_state,
                        train_state_red,
                        minibatch_current,
                        loss_func,
                        loss_func_red,
                        lr_func)
                    train_loss += train_metrics['loss'] / num_minibatches
                    train_loss_red += train_metrics['loss_red'] / num_minibatches

                # When a batch is done
                epoch_loss += train_loss / num_batches
                epoch_loss_red += train_loss_red / num_batches

                # Evaluate model with the test data
                eval_losses = eval_step(train_state,
                                        batch_test,
                                        loss_func)
                epoch_test_loss += eval_losses['loss'] / num_batches
                epoch_test_loss_red += eval_losses['loss_red'] / num_batches

                # Check for the best params per batch (test_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_params = copy(train_state.params)

                if settings_training['bootstrapping']:
                    if epoch_loss_red < best_loss_red:
                        best_loss_red = epoch_loss_red
                        best_params_red = copy(train_state_red.params)

            if epoch_loss > epoch_loss_last * early_stopping_gain and False:
                print(f'Early stopping! Epoch loss: {epoch_loss}. Now resetting.')
                settings_training['seed'] += 10
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

            train_losses_red.append(epoch_loss_red)
            test_losses_red.append(epoch_test_loss_red)
            # run.track(epoch_loss_red, name='epoch_loss_red')
            # run.track(epoch_test_loss_red, name='test_loss_red')

            # Output progress every 'test_every' epochs
            if epoch % test_every == 0:
                print(
                    f"Standard: "
                    f"Epoch={epoch}, "
                    f"train={epoch_loss:.4f}, "
                    f"test={epoch_test_loss:.4f}, "
                    f"lr={train_metrics['learning_rate']:.10f}")
                if settings_training['bootstrapping']:
                    print(
                        f"Reduced: "
                        f"Epoch={epoch}, "
                        f"train={epoch_loss_red:.4f}, "
                        f"test={epoch_test_loss_red:.4f}, "
                        f"lr={train_metrics['learning_rate']:.10f}")

            # Update epoch and record the last loss
            epoch += 1
            epoch_loss_last = epoch_loss
            epoch_loss_last_red = epoch_loss_red

    except KeyboardInterrupt:
        print('Terminating learning!')

    return best_params, best_params_red, (train_losses, test_losses), \
        (train_losses_red, test_losses_red)


def build_dummy_dataloader(batch_train: tuple,
                           batch_test: tuple,
                           settings: dict) -> Callable:
    def dataloader(key):
        return batch_train, batch_test

    return dataloader


def handle_data(settings, cursor, table_name, samples_num, offset_num):
    # Get raw data from database
    query = (
        f"SELECT * FROM {table_name} " f"LIMIT {samples_num} " f"OFFSET {offset_num}"
    )
    data_raw = jnp.array(cursor.execute(query).fetchall())

    # Format the samples
    buffer_length = settings["model_settings"]["buffer_length"]
    buffer_length_max = settings["model_settings"]["buffer_length_max"]
    format_samples = jax.vmap(
        partial(
            snake_utils.format_sample,
            buffer_length=buffer_length,
            buffer_length_max=buffer_length_max,
        )
    )
    data_formatted = format_samples(data_raw)

    # Break the formatted samples into useful magnitudes
    # TODO: Add settings to the input of this function
    split_tool = snake_utils.build_split_tool(buffer_length)
    state, ddq_target = jax.vmap(snake_utils.split_data)(data_formatted)
    q, _, dq, _, _ = jax.vmap(split_tool)(state)

    return (q, dq, ddq_target), state, data_formatted



