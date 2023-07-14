from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm

import pandas as pd

from identification.systems import dpend_model_arne as model

from identification.src.dynamix import simulate


def generate_data(settings: dict) -> ((jnp.array, jnp.array), (jnp.array, jnp.array)):
    # unpack training settings
    training_settings = settings["training_settings"]
    batch_size = training_settings["batch_size"]
    num_minibatches = training_settings["num_minibatches"]
    key = jax.random.PRNGKey(training_settings['seed'])

    # unpack system settings
    system_settings = settings["system_settings"]
    num_dof = system_settings["num_dof"]
    x0 = np.asarray(system_settings['starting_point'], dtype=np.float32)
    time_step = system_settings['time_step']

    # unpack data settings
    data_settings = settings["data_settings"]
    section_num = data_settings['num_sections']

    # create help variables
    data_size = batch_size * num_minibatches
    t_window = np.arange(data_size / section_num, dtype=np.float32) * time_step

    # create training data
    print('Generating data!')
    train_data, _ = gen_trajectory_data(x0, t_window, section_num, num_dof, key)

    # create test data
    # print('Generated test data:')
    # # noise = np.random.RandomState(0).randn(x0.size)
    # # x0_test = x0 + noise * 1e-3
    # test_data, _ = gen_trajectory_data(x0_test, t_window, section_num, num_dof, key)

    return train_data


@partial(jax.jit, static_argnums=0)
def gen_section_data(eom_compiled: Callable,
                     key,
                     x_start: jnp.array,
                     t_window: jnp.array):
    # Simulate the section from the starting point and update starting point
    x_traj_sec = simulate.solve_analytical(eom_compiled,
                                           x_start,
                                           t_window)
    x_start = x_traj_sec[-1]

    # Randomize the order of the data and calculate labels
    x_traj_sec = jax.random.permutation(key, x_traj_sec)
    xt_traj_sec = jax.vmap(eom_compiled)(x_traj_sec)
    x_traj_sec = jax.vmap(model.wrap_angle)(x_traj_sec)

    return x_start, (x_traj_sec, xt_traj_sec)


# @jax.jit
def gen_trajectory_data(x0: jnp.array,
                        t_window: jnp.array,
                        section_num: int,
                        num_dof: int,
                        key) -> (jnp.array, jnp.array, jnp.array):
    x_traj_df = None
    x_start = x0
    columns_q = [f"q{column}" for column in range(num_dof)]
    columns_dq = [f"dq{column}" for column in range(num_dof)]
    columns_ddq = [f"ddq{column}" for column in range(num_dof)]
    eom_compiled = jax.jit(model.f_analytical)
    for section in tqdm(range(section_num),
                        desc='Number of sections',
                        unit='section',
                        dynamic_ncols=True,
                        leave=False,
                        disable=True):
        # print(f"Progress: {section/section_num*100}%")
        x_start, (x_traj_sec, xt_traj_sec) = gen_section_data(eom_compiled,
                                                              key,
                                                              x_start,
                                                              t_window)

        # Check that the simulation ran correctly
        if jnp.any(jnp.isnan(x_traj_sec)) or jnp.any(jnp.isnan(xt_traj_sec)):
            raise ValueError(f'One of the sections contained "nan": {x_traj_sec}')

        # Add section to array
        xt_traj_sec = xt_traj_sec[:, :-num_dof]
        state_df = pd.DataFrame(x_traj_sec, columns=columns_q + columns_dq)
        dstate_df = pd.DataFrame(xt_traj_sec, columns=columns_ddq)
        section_df = pd.concat([state_df, dstate_df], axis=1)
        x_traj_df = update_df(x_traj_df, section_df)

    return x_traj_df, x_start


# @jax.jit
def update_df(traj, traj_sec):
    if traj is None:
        traj = traj_sec
    else:
        traj = pd.concat([traj, traj_sec], axis=0)

    return traj


def generate_random_data(settings: dict) -> (
        (jnp.array, jnp.array), (jnp.array, jnp.array)):
    # unpack training settings
    training_settings = settings["training_settings"]
    batch_size = training_settings["batch_size"]
    num_minibatches = training_settings["num_minibatches"]
    key = jax.random.PRNGKey(training_settings['seed'])

    # unpack system settings
    system_settings = settings["system_settings"]
    num_dof = system_settings["num_dof"]
    time_step = system_settings['time_step']

    # unpack data settings
    data_settings = settings['data_settings']
    num_sections = data_settings['num_sections']
    num_generators = data_settings['num_generators']

    # create help variables
    data_size = batch_size * num_minibatches
    t_window = np.arange(data_size / (num_sections * num_generators),
                         dtype=np.float32) * time_step

    # create starting points
    starting_points = random_points(key, num_generators)

    # create an empty DataFrame to store the trajectory data
    x_traj_train = None
    for point in tqdm(starting_points,
                      desc='Number of points',
                      unit='point',
                      dynamic_ncols=True,
                      leave=True):
        # create training data
        x_traj_point_df, _ = gen_trajectory_data(point,
                                                 t_window,
                                                 num_sections,
                                                 num_dof,
                                                 key)
        print(f"Samples: {x_traj_point_df.shape[0]}")
        x_traj_train = update_df(x_traj_train, x_traj_point_df)

    return x_traj_train


# @jax.jit
def random_points(key, num_points):
    jru = jax.random.uniform
    y0 = jnp.concatenate([jru(key, (num_points, 2)) * 2.0 * np.pi,
                          (jru(key + 10, (num_points, 1)) - 0.5) * 10 * 2,
                          (jru(key + 20, (num_points, 1)) - 0.5) * 10 * 4],
                         axis=1)

    return y0


def build_random_data_dataloader(batch_train: tuple, batch_test: tuple,
                                 settings: dict) -> Callable:
    # Unpack the settings
    batch_size = settings["training_settings"]['batch_size']
    num_minibathces = settings["training_settings"]['num_minibatches']

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
        y0 = jax.vmap(model.wrap_angle)(y0)

        return (y0, eqs_motion(y0)), batch_test

    return dataloader


def build_split_tool(buffer_length):
    @jax.jit
    def _split_tool(state):
        q = jnp.array([state[0 * buffer_length],
                       state[1 * buffer_length]])

        dq = jnp.array([state[4 * buffer_length],
                        state[5 * buffer_length]])

        q_buff = jnp.array([])
        dq_buff = jnp.array([])

        def extract_from_sample_split(sample,
                                      buffer_length,
                                      indices=(0, 1)):
            start = indices[0] * buffer_length
            end = indices[1] * buffer_length

            return jnp.array(sample[start:end])

        for index in range(4):
            q_temp = extract_from_sample_split(state,
                                               buffer_length,
                                               indices=(index, index + 1))

            dq_temp = extract_from_sample_split(state,
                                                buffer_length,
                                                indices=(4 + index, 4 + index + 1))

            q_buff = jnp.concatenate([q_buff, q_temp[1:]])
            dq_buff = jnp.concatenate([dq_buff, dq_temp[1:]])

        tau = state[-2:]

        return q, q_buff, dq, dq_buff, tau

    return _split_tool
