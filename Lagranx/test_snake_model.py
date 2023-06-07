import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3.common.save_util as loader
from src import dpend_model_arne as model
from src import lagranx as lx

from hyperparams import settings

import sqlite3

from functools import partial

import seaborn as sns

if __name__ == "__main__":
    # load model
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(settings, 0,
                                        params=params)

    # build dynamics
    kinetic = lx.learned_lagrangian(params, train_state, output='kinetic')
    potential = lx.learned_lagrangian(params, train_state, output='potential')
    compiled_dynamics = jax.jit(partial(lx.calc_dynamics,
                                        kinetic=kinetic,
                                        potential=potential))
    format_samples = jax.vmap(partial(lx.format_sample_test, buffer_length=25))
    inv_dyn = jax.vmap(jax.jit(partial(lx.equation_of_motion,
                                       dynamics=compiled_dynamics)))
    for_dyn_single = jax.jit(partial(lx.forward_dynamics,
                                     dynamics=compiled_dynamics))


    # wrap fd to handle data format
    def for_dyn_wrapped(data_point):
        state, ddq = data_point
        ddq = ddq[2:4]
        return for_dyn_single(ddq=ddq, state=state)


    for_dyn = jax.vmap(for_dyn_wrapped)

    # set up database
    database = sqlite3.connect('Lagranx/databases/database_points')
    table_name = 'point_3'
    samples_num = 500
    offset_num = 500
    cursor = database.cursor()

    # define query
    def query():
        query = f'SELECT * FROM {table_name} ' \
                f'LIMIT {samples_num} ' \
                f'OFFSET {offset_num}'
        return query

    # format data
    data_raw = jnp.array(cursor.execute(query()).fetchall())
    data_formatted = format_samples(data_raw)
    state, ddq_target = data_formatted

    # calculate magnitudes of interest
    tau, tau_target = for_dyn(data_formatted)
    ddq = inv_dyn(state)

    # Plotting
    sns.set(style="darkgrid")

    # Accelerations
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(ddq_target[:, 2:4], linewidth=2, label='target')
    plt.plot(ddq[:, 2:4], linewidth=2, label='pred')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Comparison between predicted and actual joint acc.')
    plt.ylabel('rad/s')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Torques
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(tau_target, linewidth=2, label='target')
    plt.plot(tau, linewidth=2, label='pred')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Comparison between predicted and actual joint torques.')
    plt.ylabel('Nm')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()
