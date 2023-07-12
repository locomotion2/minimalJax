from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from identification.systems import dpend_model_arne as model

from identification.src.dynamix import simulate


def generate_data(settings: dict) -> ((jnp.array, jnp.array), (jnp.array, jnp.array)):
    # unpack training settings
    training_settings = settings["training_settings"]
    batch_size = training_settings["batch_size"]
    num_minibatches = training_settings["num_minibatches"]
    section_num = training_settings['sections_num']
    key = jax.random.PRNGKey(training_settings['seed'])

    # unpack system settings
    system_settings = settings["system_settings"]
    x0 = np.asarray(system_settings['starting_point'], dtype=np.float32)
    time_step = system_settings['time_step']

    # create help variables
    data_size = batch_size * num_minibatches
    t_window = np.arange(data_size / section_num, dtype=np.float32) * time_step

    # create training data
    print('Generating train data:')
    train_data, x0_test = generate_trajectory_data(x0, t_window, section_num, key)

    # create test data
    print('Generated test data:')
    # noise = np.random.RandomState(0).randn(x0.size)
    # x0_test = x0 + noise * 1e-3
    test_data, _ = generate_trajectory_data(x0_test, t_window, section_num, key)

    return train_data, test_data


def generate_random_traj_data(settings: dict) -> (
        (jnp.array, jnp.array), (jnp.array, jnp.array)):
    # unpack training settings
    training_settings = settings["training_settings"]
    batch_size = training_settings["batch_size"]
    num_minibatches = training_settings["num_minibatches"]
    num_sections = training_settings['num_sections']
    num_generators = training_settings['num_generators']
    key = jax.random.PRNGKey(training_settings['seed'])

    # unpack system settings
    system_settings = settings["system_settings"]
    x0 = np.asarray(system_settings['starting_point'], dtype=np.float32)
    time_step = system_settings['time_step']

    # create help variables
    data_size = batch_size * num_minibatches
    t_window = np.arange(data_size / (num_sections * num_generators),
                         dtype=np.float32) * time_step

    @jax.jit
    def update_list(traj, traj_target, traj_sec, traj_target_sec):
        if traj is None:
            traj = traj_sec
            traj_target = traj_target_sec
        else:
            traj = jnp.append(traj, traj_sec, axis=0)
            traj_target = jnp.append(traj_target, traj_target_sec, axis=0)

        return traj, traj_target

    # create starting points
    starting_points = random_points(key, num_generators)

    # simulate a trajectory starting from each point
    x_traj_train = None
    xt_traj_train = None
    x_traj_test = None
    xt_traj_test = None
    for point in starting_points:
        # create training data
        print('Generating train data.')
        (x_traj_point, xt_traj_point), x0_test = generate_trajectory_data(point,
                                                                          t_window,
                                                                          num_sections,
                                                                          key)
        x_traj_train, xt_traj_train = update_list(x_traj_train,
                                                  xt_traj_train,
                                                  x_traj_point,
                                                  xt_traj_point)

        # create test data
        print('Generated test data.')
        # noise = np.random.RandomState(0).randn(x0.size)
        # x0_test = x0 + noise * 1e-3
        (x_traj_point, xt_traj_point), _ = generate_trajectory_data(x0_test,
                                                                    t_window,
                                                                    num_sections,
                                                                    key)
        x_traj_test, xt_traj_test = update_list(x_traj_test,
                                                xt_traj_test,
                                                x_traj_point,
                                                xt_traj_point)

    return (x_traj_train, xt_traj_train), (x_traj_test, xt_traj_test)


@jax.jit
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


@jax.jit
def generate_trajectory_data(x0: jnp.array,
                             t_window: jnp.array,
                             section_num: int,
                             key) -> (jnp.array, jnp.array, jnp.array):
    x_traj = None
    xt_traj = None
    x_start = x0
    for section in range(section_num):
        # Simulate the section from the starting point and update starting point
        x_traj_sec = simulate.solve_analytical(x_start, t_window)
        x_start = x_traj_sec[-1]

        # Randomize the order of the data and calculate labels
        x_traj_sec = jax.random.permutation(key, x_traj_sec)
        xt_traj_sec = jax.vmap(model.f_analytical)(x_traj_sec)
        x_traj_sec = jax.vmap(model.wrap_angle)(x_traj_sec)

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
        f"Generation successful! Ranges: "
        f"{jnp.amax(x_traj, axis=0)}, "
        f"{jnp.amin(x_traj, axis=0)}")
    return (x_traj, xt_traj), x_start
