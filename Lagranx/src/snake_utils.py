import sqlite3
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from Lagranx.src import utils


class DeLaNN(nn.Module):

    @nn.compact
    def __call__(self, x, num_dof=4, net_size=64 * 4, friction=False):
        # unpacking data
        q_state, _ = jnp.split(x, 2)
        full_state = jnp.concatenate([q_state, jnp.array([0] * len(q_state))])

        # build kinetic net
        dim = int(num_dof * (num_dof + 1) / 2)
        x_kin_1 = self.layer(q_state, features=net_size)
        x_kin_2 = self.layer(x_kin_1, features=net_size)
        x_kin = nn.Dense(features=dim)(x_kin_1 + x_kin_2)

        # build potential net
        x_pot_1 = self.layer(q_state, features=net_size)
        x_pot_2 = self.layer(x_pot_1, features=net_size)
        x_pot = nn.Dense(features=1)(x_pot_1 + x_pot_2)

        # # build friction net
        x_f_1 = self.layer(full_state, features=net_size)
        x_f_2 = self.layer(x_f_1, features=net_size)
        x_f = self.layer(x_f_1 + x_f_2, features=num_dof) * int(friction)

        return jnp.concatenate([x_kin, x_pot, x_f])

    def layer(self, x: jnp.array, features: int = 128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.softplus(x)
        return x


def build_split_tool(buffer_length):
    @jax.jit
    def _split_tool(state):
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

        tau = state[-4:]

        return q, q_buff, dq, dq_buff, tau

    return _split_tool


# @jax.jit
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
                              indices=(0, 0, 4))
    q_n = utils.wrap_angle(q_n)

    # build dq vector
    dq_n = extract_from_sample(sample,
                               buffer_length,
                               buffer_length_max,
                               indices=(4, 0, 4))

    # build dq_0, ddq, tau
    index_end = 8 * buffer_length_max
    dq_0 = jnp.array([dq_n[0 * buffer_length],
                      dq_n[1 * buffer_length],
                      dq_n[2 * buffer_length],
                      dq_n[3 * buffer_length]])
    ddq_0 = jnp.array(sample[index_end: index_end + 4])
    tau = jnp.array(sample[index_end + 4: index_end + 8])

    # build state variables
    state = jnp.concatenate([q_n, dq_n])
    dstate_0 = jnp.concatenate([dq_0, ddq_0])
    state_ext = jnp.concatenate([state, tau])

    # return state_ext, dstate_0
    return jnp.concatenate([state_ext, dstate_0])


def build_database_dataloader_eff(settings: dict) -> Callable:
    # Unpack the settings
    batch_size = settings['batch_size']
    num_minibatches = settings['num_minibatches']
    num_skips = settings['eff_datasampling']
    buffer_length = settings['buffer_length']
    buffer_length_max = settings['buffer_length_max']
    partition = settings['data_partition']

    # Set up help variables
    data_size_train = batch_size * num_minibatches * num_skips
    data_size_test = batch_size * num_skips

    # Set up database connection
    database = sqlite3.connect(settings['database_name'])
    cursor = database.cursor()

    # Count the samples
    table_name = settings['table_name']
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
                                               (data_size_train,))
        batch_validation_sub = jax.random.choice(key, batch_validation_formatted,
                                                 (data_size_test,))

        # split the data
        batch_training = split_data_vec(batch_training_sub)
        batch_validation = split_data_vec(batch_validation_sub)

        return batch_training, batch_validation

    return dataloader


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
    samples_train = int(samples_total * 0.1)
    samples_validation = int(samples_total * 0.1)
    # samples_test = int(samples_total * 0.1)

    # query commands
    # query_sample_test = f'SELECT * FROM your_table ORDER BY RANDOM() LIMIT ' \
    #                     f'{samples_train + samples_validation},{samples_test} ' \
    #                     f'LIMIT {data_size}'

    format_samples = jax.vmap(jax.jit(partial(format_sample,
                                              buffer_length=buffer_length,
                                              buffer_length_max=buffer_length_max)))

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


@jax.jit
def split_data(data):
    state = jnp.array(data[:-8])
    ddq = jnp.array(data[-8:])

    return state, ddq
