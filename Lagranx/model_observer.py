import sys
import links_and_nodes as ln

from functools import partial

from hyperparams import settings

import jax
import jax.numpy as jnp
import numpy as np

from src import lagranx as lx

import stable_baselines3.common.save_util as loader


@jax.jit
def update_format_buffers(variable_buffer, variable):
    var_list_save = []
    var_list_send = []
    for index, element in enumerate(variable):
        # update buffer
        vector = variable_buffer[index, 0:-1]
        vector = jnp.insert(vector, 0, element)
        var_list_save.append(vector)

        # format buffer to be send
        # var_list_send.append(vector[::10])
        var_list_send.append(vector)

    return jnp.array(var_list_save), jnp.concatenate(var_list_send)


# @jax.jit
def format_state(q, q_buff, dq, dq_buff):
    # format into the right sizes
    q = lx.normalize(q)

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

    # load the trained model
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(settings, 0, params=params)

    # build dynamics
    kinetic = lx.learned_lagrangian(params, train_state, output='kinetic')
    potential = lx.learned_lagrangian(params, train_state, output='potential')
    friction = lx.learned_lagrangian(params, train_state, output='friction')
    compiled_dynamics = partial(lx.calc_dynamics,
                                kinetic=kinetic,
                                potential=potential,
                                friction=friction)
    compiled_dyn_matrices = partial(lx.dynamic_matrices,
                                    dynamics=compiled_dynamics)
    compiled_dyn_matrices_red = jax.jit(partial(lx.simplified_dynamic_matrices,
                                                dynamics=compiled_dyn_matrices))
    cdmr = compiled_dyn_matrices_red
    # inv_dyn = jax.jit(partial(lx.equation_of_motion,
    #                           dynamics=compiled_dyn_matrices))
    # for_dyn = jax.jit(partial(lx.forward_dynamics,
    #                           dynamics=compiled_dyn_matrices))

    # setup the state buffering
    buffer_length = settings['buffer_length']
    # q_buff = jnp.zeros((4, buffer_length * 10))
    # dq_buff = jnp.zeros((4, buffer_length * 10))
    q_buff = jnp.zeros((4, buffer_length))
    dq_buff = jnp.zeros((4, buffer_length))
    try:
        while True:
            # Get q, dq, ddq, time
            print('Listening to topic...')
            subscriber.read()
            time = subscriber.packet.time
            q = jnp.array(subscriber.packet.q)
            dq = jnp.array(subscriber.packet.dq)
            # ddq = jnp.array(subscriber.packet.ddq)
            # tau_target = jnp.array(subscriber.packet.tau)
            print(f"received: {q}, {dq}")

            # Calculate dynamics
            state, q_buff, dq_buff = format_state(q, q_buff, dq, dq_buff)
            (M_rob, C_rob, g_rob, k_f_rob), (B, k_f_mot), K_red = cdmr(state)

            # Send tau
            publisher.packet.M_rob = np.array(M_rob.flatten())
            publisher.packet.C_rob = np.array(C_rob.flatten())
            publisher.packet.g_rob = np.array(g_rob.flatten())
            publisher.packet.k_f_rob = np.array(k_f_rob.flatten())
            publisher.packet.B = np.array(B.flatten())
            publisher.packet.k_f_mot = np.array(k_f_mot.flatten())
            publisher.packet.K_red = np.array(K_red.flatten())
            # print(f"sending: {tau[0:2]}, {ddq_pred[4:8]}")
            publisher.write()

    except KeyboardInterrupt:
        print('User commanded termination.')

    print('Program terminated successfully.')
