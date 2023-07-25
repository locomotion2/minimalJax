from typing import Callable

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

from tqdm import tqdm

import pandas as pd

from identification.systems import dpend_model_arne as model

from identification.src.dynamix import simulate

import sqlite3

import identification.identification_utils as utils

class DeLaNN(nn.Module):

    @nn.compact
    def __call__(self, x, num_dof=2, net_size=64 * 4, friction=False):
        # unpacking data
        q_state, _ = jnp.split(x, 2)
        q = q_state[0:2]
        # q_state = jnp.concatenate([q, jnp.array([0] * 76)])
        # full_state = jnp.concatenate([q_state, jnp.array([0] * len(q_state))])

        # build kinetic net
        dim = int(num_dof * (num_dof + 1) / 2)
        x_kin_1 = self.layer(q_state, features=net_size)
        x_kin_2 = self.layer(x_kin_1, features=net_size)
        x_kin_3 = self.layer(x_kin_2, features=net_size)
        x_kin = nn.Dense(features=dim)(x_kin_1 + x_kin_2 + x_kin_3)

        # build potential net
        x_pot_1 = self.layer(q, features=net_size)
        x_pot_2 = self.layer(x_pot_1, features=net_size)
        x_pot_3 = self.layer(x_pot_2, features=net_size)
        # x_pot_4 = self.layer(x_pot_2, features=net_size)
        x_pot = nn.Dense(features=dim)(x_pot_1 + x_pot_2 + x_pot_3)

        # build friction net
        # x_f_1 = self.layer(full_state, features=net_size)
        # x_f_2 = self.layer(x_f_1, features=net_size)
        # x_f = self.layer(x_f_1 + x_f_2, features=num_dof) * int(friction)

        # x_pot = jnp.array([0])
        x_f = jnp.array([0] * num_dof)
        return jnp.concatenate([x_kin, x_pot, x_f])

    def layer(self, x: jnp.array, features: int = 128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.tanh(x)
        return x

class DeLaNN_RED(nn.Module):

    @nn.compact
    def __call__(self, x, num_dof=2, net_size=64 * 4, friction=False):
        # unpacking data
        q = x

        # build kinetic net
        dim = int(num_dof * (num_dof + 1) / 2)
        x_kin_1 = self.layer(q, features=net_size)
        x_kin_2 = self.layer(x_kin_1, features=net_size)
        x_kin_3 = self.layer(x_kin_2, features=net_size)
        x_kin = nn.Dense(features=dim)(x_kin_1 + x_kin_2 + x_kin_3)

        # build potential net
        # x_pot_1 = self.layer(q_state, features=net_size)
        # x_pot_2 = self.layer(x_pot_1, features=net_size)
        # x_pot_3 = self.layer(x_pot_2, features=net_size)
        # x_pot_4 = self.layer(x_pot_2, features=net_size)
        # x_pot = nn.Dense(features=1)(x_pot_1 + x_pot_2 + x_pot_3)

        # build friction net
        # x_f_1 = self.layer(full_state, features=net_size)
        # x_f_2 = self.layer(x_f_1, features=net_size)
        # x_f = self.layer(x_f_1 + x_f_2, features=num_dof) * int(friction)

        x_pot = jnp.array([0])
        x_f = jnp.array([0] * 4)
        return jnp.concatenate([x_kin, x_pot, x_f])

    def layer(self, x: jnp.array, features: int = 128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.tanh(x)
        return x

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


# @jax.jit
def build_state_buffer_gen(data_size, buffer_length):
    # @jax.jit
    def state_buffer_gen(trajectory: jnp.array):
        state_buffer = []

        # Iterate over the trajectory starting from index m
        for i in jnp.arange(0, data_size):
            if i >= buffer_length:
                # Extract the previous m states
                prev_states = trajectory[i - buffer_length:i, :].T.flatten()
            else:
                # Repeat the first state if not enough early states
                first_state_rep = jnp.tile(trajectory[0, :], [buffer_length - i, 1])

                # Get the remaining states
                prev_states = trajectory[0:i, :]

                # Join states and flatten
                prev_states = jnp.concatenate([first_state_rep, prev_states], axis=0)
                prev_states = prev_states.T.flatten()

            state_buffer.append(prev_states)

        return jnp.array(state_buffer)

    return state_buffer_gen


# @jax.jit
def extract_state(trajectory_buffered, num_dof, buffer_length):
    return jnp.array([trajectory_buffered[:, coord * buffer_length] for coord in range(num_dof * 2)]).T


@partial(jax.jit, static_argnums=0)
def gen_section_data(eom_compiled: Callable,
                     x_start: jnp.array,
                     t_window: jnp.array):
    # simulate the section from the starting point and update starting point
    x_traj_sec = simulate.solve_analytical(eom_compiled,
                                           x_start,
                                           t_window)
    x_start = x_traj_sec[-1]
    x_traj_sec = jax.vmap(model.wrap_angle)(x_traj_sec)

    return x_start, x_traj_sec


# @jax.jit
def gen_trajectory_data(buffer_generator: Callable,
                        x0: jnp.array,
                        t_window: jnp.array,
                        section_num: int,
                        num_dof: int,
                        buffer_length_max,
                        key) -> (jnp.array, jnp.array, jnp.array):
    # unpack / build help variables
    x_traj = None
    x_start = x0
    eom_compiled = jax.jit(model.f_analytical)

    # define columns
    columns_q = [f"q{column}_{index}" for column in range(num_dof) for index in range(buffer_length_max)]
    columns_dq = [f"dq{column}_{index}" for column in range(num_dof) for index in range(buffer_length_max)]
    columns_ddq = [f"ddq{column}" for column in range(num_dof)]

    # iterate through the sections
    for section in tqdm(range(section_num),
                        desc='Number of sections',
                        unit='section',
                        dynamic_ncols=True,
                        leave=False,
                        disable=True):

        x_start, x_traj_sec = gen_section_data(eom_compiled,
                                               x_start,
                                               t_window)

        # check that the simulation ran correctly
        if jnp.any(jnp.isnan(x_traj_sec)):
            raise ValueError(f'One of the sections contained "nan": {x_traj_sec}')

        # add section to array
        x_traj = update_array(x_traj, x_traj_sec)

    # build state buffer
    x_traj_buff = buffer_generator(x_traj)

    # randomize the order of the data and calculate labels
    x_traj_buff_perm = jax.random.permutation(key, x_traj_buff)
    x_traj_perm = extract_state(x_traj_buff_perm, num_dof, buffer_length_max)
    dx_traj_perm = jax.vmap(eom_compiled)(x_traj_perm)
    dx_traj_perm = dx_traj_perm[:, :-num_dof]

    # convert all to dataframes
    state_df = pd.DataFrame(x_traj_buff_perm, columns=columns_q + columns_dq)
    dstate_df = pd.DataFrame(dx_traj_perm, columns=columns_ddq)
    point_df = pd.concat([state_df, dstate_df], axis=1)

    return point_df, x_start


# @jax.jit
def update_df(traj, traj_sec):
    if traj is None:
        traj = traj_sec
    else:
        traj = pd.concat([traj, traj_sec], axis=0)

    return traj


@jax.jit
def update_array(traj: jnp.array, traj_sec):
    if traj is None:
        traj = traj_sec
    else:
        traj = jnp.concatenate([traj, traj_sec], axis=0)

    return traj


def generate_random_data(settings: dict,
                         num_points: int,
                         num_points_start: int) -> (
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

    # upack model settings
    model_settings = settings['model_settings']
    buffer_length_max = model_settings['buffer_length_max']

    # create help variables
    data_size = batch_size * num_minibatches
    t_size = int(np.max([data_size / (num_sections * num_generators), buffer_length_max]))
    t_window = np.arange(t_size, dtype=np.float32) * time_step

    # create starting points
    starting_points = random_points(key, num_generators)

    buffer_generator = build_state_buffer_gen(t_size, buffer_length_max)
    try:
        filepath = f"{settings['data_settings']['data_dir']}"
        with open(filepath, 'r'):
            # setup new database
            database = sqlite3.connect(f"{filepath}")
            for num in tqdm(range(num_points),
                            desc='Number of points',
                            unit='point',
                            dynamic_ncols=True,
                            leave=True):
                # number point
                num_eff = num + num_points_start

                # create training data
                x_traj_point_df, _ = gen_trajectory_data(buffer_generator,
                                                         starting_points[num_eff],
                                                         t_window,
                                                         num_sections,
                                                         num_dof,
                                                         buffer_length_max,
                                                         key)

                # save data
                x_traj_point_df.to_sql(f"point_{num_eff}", database, if_exists='replace', index=False)

                # add more randomness
                key += 10 + num_eff
    finally:
        # close everything
        database.close()
        print("Data generation finished.")


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

        dq = jnp.array([state[2 * buffer_length],
                        state[3 * buffer_length]])

        q_buff = jnp.array([])
        dq_buff = jnp.array([])

        def extract_from_sample_split(sample,
                                      buffer_length,
                                      indices=(0, 1)):
            start = indices[0] * buffer_length
            end = indices[1] * buffer_length

            return jnp.array(sample[start:end])

        for index in range(2):
            q_temp = extract_from_sample_split(state,
                                               buffer_length,
                                               indices=(index, index + 1))

            dq_temp = extract_from_sample_split(state,
                                                buffer_length,
                                                indices=(2 + index, 2 + index + 1))

            q_buff = jnp.concatenate([q_buff, q_temp[1:]])
            dq_buff = jnp.concatenate([dq_buff, dq_temp[1:]])

        tau = state[-2:]

        return q, q_buff, dq, dq_buff, tau

    return _split_tool


def scramble_data(database, database_target, params):
    # define the subset of interesting columns
    extra_columns = ['ddq1_fil', 'ddq2_fil']

    # define the subset of columns to include in the buffer
    buffer_columns = ['q1', 'q2', 'dq1_fil', 'dq2_fil']

    # create new column names for the buffer columns
    buffer_column_names = [f'{column}_{t}' for column in buffer_columns for t in
                           range(params['model_settings']['buffer_length_max'])]

    interesting_guys = buffer_column_names + extra_columns

    aggregated_df = pd.DataFrame(columns=interesting_guys)
    total_points = 879
    for point_id in tqdm(range(total_points),
                         desc='Number of points',
                         unit='point',
                         dynamic_ncols=True,
                         leave=True):
        # get point df
        point_target = f'point_{point_id}'
        df = get_dataframe(point_target, database)
        df.rename(columns=dict(zip(df.columns, interesting_guys)), inplace=True)

        # add to the bucket
        aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)

    # add the torques
    data_len = aggregated_df.shape[0]
    aggregated_df['tau1'] = np.zeros(data_len)
    aggregated_df['tau2'] = np.zeros(data_len)

    # do the scrambling
    scrambled_df = aggregated_df.sample(frac=1, random_state=params['training_settings']['seed'])
    scrambled_df.to_sql(params['data_settings']['table_name'], database_target, if_exists='replace', index=False)
    print('New scrambled data saved.')


def get_dataframe(target_point, database):
    # conn = sqlite3.connect(params['database_name'])
    query = f"SELECT * FROM {target_point}"
    df = pd.read_sql_query(query, database)
    # conn.close()

    return df

def build_database_dataloader_eff(settings: dict) -> Callable:
    # Unpack the settings
    settings_training = settings['training_settings']
    settings_model = settings['model_settings']
    settings_system = settings['system_settings']
    settings_data = settings['data_settings']

    batch_size = settings_training['batch_size']
    num_minibatches = settings_training['num_minibatches']

    num_skips = settings_data['eff_datasampling']
    partition = settings_data['data_partition']

    buffer_length = settings_model['buffer_length']
    buffer_length_max = settings_model['buffer_length_max']

    # Set up help variables
    data_size_train = batch_size * num_minibatches * num_skips
    data_size_test = batch_size * num_skips

    # Set up database connection
    database = sqlite3.connect(settings_data['database_name'])
    cursor = database.cursor()

    # Count the samples
    table_name = settings_data['table_name']
    samples_total = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    samples_train = int(samples_total * partition[0])
    samples_validation = int(samples_total * partition[1])
    samples_test = int(samples_total * partition[2])

    # Prepare query commands
    query_sample_training = f'SELECT * FROM {table_name} ' \
                            f'LIMIT {samples_train}'
    query_sample_validation = f'SELECT * FROM {table_name} ' \
                              f'LIMIT {samples_validation} OFFSET {samples_train} '

    format_samples = jax.vmap(jax.jit(partial(format_sample,
                                              buffer_length=buffer_length,
                                              buffer_length_max=buffer_length_max)))

    print(f"Epoch size: {data_size_train}, Training size: {samples_train}")

    split_data_vec = jax.vmap(split_data)

    # Fetch all training and validation data
    print('Fetching a lot of data...')
    batch_training_raw = jnp.array(cursor.execute(query_sample_training).fetchall())
    batch_validation_raw = jnp.array(cursor.execute(query_sample_validation).fetchall())

    # Format data
    print('Formatting all that data...')
    batch_training_formatted = format_samples(jnp.array(batch_training_raw))
    batch_validation_formatted = format_samples(jnp.array(batch_validation_raw))

    # Close the database connection
    cursor.close()
    database.close()

    # Create the dataloader
    @jax.jit
    def dataloader(seed):
        key = jax.random.PRNGKey(seed)

        # Random subsample from data
        batch_training_sub = jax.random.choice(key, batch_training_formatted,
                                               (data_size_train,), replace=False)
        batch_validation_sub = jax.random.choice(key, batch_validation_formatted,
                                                 (data_size_test,), replace=False)

        # split the data
        batch_training = split_data_vec(batch_training_sub)
        batch_validation = split_data_vec(batch_validation_sub)

        return batch_training, batch_validation

    return dataloader

def format_sample(sample, buffer_length, buffer_length_max):

    # define function to extract one coordinate
    def extract_from_sample(sample,
                            buffer_length,
                            buffer_length_max,
                            indices=(0, 0, 1)):
        output = jnp.array([])
        for iteration in range(indices[2] - indices[1]):
            index = iteration + indices[0]
            start = index * buffer_length_max
            end = start + buffer_length
            buffer_chunk = jnp.array(sample[start:end])
            output = jnp.concatenate([output, buffer_chunk])

        return output

    # build q vector
    q_n = extract_from_sample(sample,
                              buffer_length,
                              buffer_length_max,
                              indices=(0, 0, 2))
    q_n = utils.wrap_angle(q_n)

    # build dq vector
    dq_n = extract_from_sample(sample,
                               buffer_length,
                               buffer_length_max,
                               indices=(2, 0, 2))

    # build dq_0, ddq, tau
    index_end = 4 * buffer_length_max
    dq_0 = jnp.array([dq_n[0 * buffer_length],
                      dq_n[1 * buffer_length]])
    ddq_0 = jnp.array(sample[index_end: index_end + 2])
    tau = jnp.array(sample[index_end + 2: index_end + 4])

    # build state variables
    state = jnp.concatenate([q_n, dq_n])
    dstate_0 = jnp.concatenate([dq_0, ddq_0])
    state_ext = jnp.concatenate([state, tau])

    # return state_ext, dstate_0
    return jnp.concatenate([state_ext, dstate_0])

@jax.jit
def split_data(data):
    state = jnp.array(data[:-4])
    ddq = jnp.array(data[-4:])

    return state, ddq