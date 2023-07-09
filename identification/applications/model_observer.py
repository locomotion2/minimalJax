import sys
import links_and_nodes as ln

from functools import partial

from hyperparams import settings

import jax
import jax.numpy as jnp
import numpy as np

from src.dynamix import simulate as lx
from src.learning import trainer
from systems import snake_utils
import identification_utils as utils

import stable_baselines3.common.save_util as loader


# @jax.jit
# def update_format_buffers(variable_buffer, variable):
#     var_list_save = []
#     var_list_send = []
#     for index, element in enumerate(variable):
#         # update buffer
#         vector = variable_buffer[index, 0:-1]
#         vector = jnp.insert(vector, 0, element)
#         var_list_save.append(vector)
#
#         # format buffer to be send
#         # var_list_send.append(vector[::10])
#         var_list_send.append(vector)
#
#     return jnp.array(var_list_save), jnp.concatenate(var_list_send)

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
def format_state(q, q_buff, dq, dq_buff):
    # format into the right sizes
    q = utils.wrap_angle(q)

    # update buffers, subsample and flatten
    q_buff, q_out = update_format_buffers(q_buff, q)
    dq_buff, dq_out = update_format_buffers(dq_buff, dq)

    # prepare package
    state = jnp.concatenate([q_out, dq_out, jnp.array([0, 0, 0, 0])])

    return state, q_buff, dq_buff


if __name__ == '__main__':
    # config ln coms
    print('Snake learned controller initiated.')
    clnt = ln.client(sys.argv[0], sys.argv)
    subscriber = clnt.subscribe("ff.controller.in", "ff_controller_in")
    publisher = clnt.publish("observer.red.out", "observer_red_out")
    print('LN connection established')

    # load params from settings
    num_dof = settings['num_dof']
    buffer_length = settings['buffer_length']

    # load the trained model
    settings['sys_utils'] = snake_utils
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = trainer.create_train_state_DeLaNN(settings, 0, params=params)

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
    dyns_reduced = partial(lx.decouple_model, dynamics=dyns_compiled)

    # set up the state buffering
    # q_buff = jnp.zeros((4, buffer_length * 10))
    # dq_buff = jnp.zeros((4, buffer_length * 10))
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
            # ddq = jnp.array(subscriber.packet.ddq)
            # tau_target = jnp.array(subscriber.packet.tau)
            # print(f"received: {q}, {dq}")

            # Calculate dynamics
            state, q_buff, dq_buff = format_state(q, q_buff, dq, dq_buff)
            (M_rob, C_rob, g_rob, k_f_rob), (B, k_f_mot) = dyns_reduced(state)

            # Send tau
            publisher.packet.M_rob = np.array(M_rob.flatten())
            publisher.packet.C_rob = np.array(C_rob.flatten())
            publisher.packet.g_rob = np.array(g_rob.flatten())
            publisher.packet.k_f_rob = np.array(k_f_rob.flatten())
            publisher.packet.B = np.array(B.flatten())
            publisher.packet.k_f_mot = np.array(k_f_mot.flatten())
            # publisher.packet.K_red = np.array(K_red.flatten())
            # print(f"sending: {tau[0:2]}, {ddq_pred[4:8]}")
            publisher.write()

    except KeyboardInterrupt:
        print('User commanded termination.')

    print('Program terminated successfully.')
