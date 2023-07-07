import sys
import links_and_nodes as ln

from functools import partial

from hyperparams import settings

import jax
import jax.numpy as jnp
import numpy as np

from src import lagranx as lx
from src import trainer
from systems import snake_utils
from src import utils

import stable_baselines3.common.save_util as loader

from scipy.signal import savgol_filter


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
    params = loader.load_from_pkl(path=settings['ckpt_dir_model'], verbose=1)
    train_state = trainer.create_train_state_PowNN(settings, 0, params=params)

    # build dynamics
    kinetic = lx.energy_func_model(params, train_state, settings=settings,
                                   output='kinetic')
    potential = lx.energy_func_model(params, train_state, settings=settings,
                                     output='potential')
    friction = lx.energy_func_model(params, train_state, settings=settings,
                                    output='friction')
    inertia = lx.energy_func_model(params, train_state, settings=settings,
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
    # q_buff = jnp.zeros((num_dof, buffer_length))
    # dq_buff = jnp.zeros((num_dof, buffer_length))
    try:
        while True:
            # Get q, dq, ddq, time
            # print('Listening to topic...')
            subscriber.read()
            time = subscriber.packet.time
            q_buff = jnp.array(subscriber.packet.q)
            dq_buff = jnp.array(subscriber.packet.dq)
            ddq = jnp.array(subscriber.packet.ddq)
            tau_target = jnp.array(subscriber.packet.tau)
            print(f"received: {ddq}, {tau_target}")

            # Format input
            q_buff = utils.wrap_angle(q_buff)
            state = jnp.concatenate([q_buff, dq_buff, tau_target])

            # Calculate dynamics
            # state, q_buff, dq_buff = utils.format_state(q, q_buff, dq, dq_buff,
            #                                             tau_target)
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
            print(f"sending: {tau_pred[2:4]}, {ddq_pred[4:8]}")
            publisher.write()

    except KeyboardInterrupt:
        print('User commanded termination.')

    print('Program terminated successfully.')
