from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from identification.systems import dpend_model_arne as model
from identification.src import lagranx as lx

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

def build_random_data_dataloader(batch_train: tuple, batch_test: tuple,
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

def generate_trajectory_data(x0: jnp.array, t_window: jnp.array, section_num: int,
                             key: int) -> (jnp.array, jnp.array, jnp.array):
    x_traj = None
    xt_traj = None
    x_start = x0
    for section in range(section_num):
        # Simulate the section from the starting point and update starting point
        x_traj_sec = lx.solve_analytical(x_start, t_window)
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
