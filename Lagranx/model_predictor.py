import sys
import links_and_nodes as ln

from functools import partial

from hyperparams import settings

import jax
import jax.numpy as jnp
import numpy as np

from src import lagranx as lx
from src import trainer
from src import snake_utils
from src import utils

import stable_baselines3.common.save_util as loader

from scipy.signal import savgol_filter


@jax.jit
def update_format_buffers(variable_buffer, variable):
    # var_list_save = []
    # var_list_send = []
    # for index, element in enumerate(variable):
    # update buffer
    vector = variable_buffer[:, 0:-1]
    vector = jnp.insert(vector, 0, variable, axis=1)
    var_list_save = vector

    # filter
    # vector = np.array(vector)
    # vector_filtered = filter_svg(vector)

    # format buffer to be sent
    var_list_send = vector
    # var_list_send = vector[:, ::10]
    # var_list_send.append(vector)

    return jnp.array(var_list_save), jnp.concatenate(var_list_send)


@jax.jit
def format_state(q, q_buff, dq, dq_buff, tau):
    # format into the right sizes
    q = utils.wrap_angle(q)

    # update buffers, subsample and flatten
    q_buff, q_out = update_format_buffers(q_buff, q)
    dq_buff, dq_out = update_format_buffers(dq_buff, dq)

    # prepare package
    state = jnp.concatenate([q_out, dq_out, tau])

    return state, q_buff, dq_buff


def filter_svg(signal):
    window = 19
    order = 3
    sampling_freq = 1 / 1000
    return savgol_filter(signal,
                         window_length=window,
                         polyorder=order,
                         delta=sampling_freq,
                         axis=1,
                         deriv=0)

if __name__ == '__main__':
    # config ln coms
    print('Snake learned controller initiated.')
    clnt = ln.client(sys.argv[0], sys.argv)
    subscriber = clnt.subscribe("ff.controller.in", "ff_controller_in")
    publisher = clnt.publish("ff.controller.out", "ff_controller_out")
    print('LN connection established')

    # load params from settings
    num_dof = settings['num_dof']
    buffer_length = settings['buffer_length']

    # load the trained model
    settings['sys_utils'] = snake_utils
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = trainer.create_train_state(settings, 0, params=params)

    # build dynamics
    kinetic = lx.energy_func(params, train_state, settings=settings,
                             output='kinetic')
    potential = lx.energy_func(params, train_state, settings=settings,
                               output='potential')
    friction = lx.energy_func(params, train_state, settings=settings,
                              output='friction')
    inertia = lx.energy_func(params, train_state, settings=settings,
                             output='inertia')
    split_tool = snake_utils.build_split_tool(buffer_length)
    dyn_builder = partial(lx.inertia_dyn_builder,
                          split_tool=split_tool,
                          kinetic=kinetic,
                          potential=potential,
                          inertia=inertia,
                          friction=friction)
    dyns_compiled = jax.jit(dyn_builder)
    energies = jax.jit(partial(lx.energy_wrapper,
                               split_tool=split_tool,
                               potential=potential,
                               kinetic=kinetic))

    # set up the state buffering
    # q_buff = jnp.zeros((num_dof, buffer_length * 10))
    # dq_buff = jnp.zeros((num_dof, buffer_length * 10))
    q_buff = jnp.zeros((num_dof, buffer_length))
    dq_buff = jnp.zeros((num_dof, buffer_length))
    try:
        while True:
            # Get q, dq, ddq, time
            # print('Listening to topic...')
            subscriber.read()
            time = subscriber.packet.time
            q = jnp.array(subscriber.packet.q)
            dq = jnp.array(subscriber.packet.dq)
            ddq = jnp.array(subscriber.packet.ddq)
            tau_target = jnp.array(subscriber.packet.tau)
            # print(f"received: {q}, {dq}, {ddq}, {tau_target}")

            # Calculate dynamics
            state, q_buff, dq_buff = format_state(q, q_buff, dq, dq_buff, tau_target)
            dyn_terms = dyn_builder(state)
            ddq_pred = lx.forward_dynamics(dyn_terms)
            tau_pred, _, _ = lx.inverse_dynamics(ddq=ddq,
                                                 terms=dyn_terms)
            T, V = energies(state)

            # Send tau
            publisher.packet.tau = np.array(tau_pred[2:4])
            publisher.packet.ddq = np.array(ddq_pred[4:8])
            publisher.packet.T = np.array(T)
            publisher.packet.V = np.array(V)
            # print(f"sending: {tau_pred[2:4]}, {ddq_pred[4:8]}")
            publisher.write()

    except KeyboardInterrupt:
        print('User commanded termination.')

    print('Program terminated successfully.')
