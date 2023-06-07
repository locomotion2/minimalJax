import sys
import links_and_nodes as ln

from functools import partial

from hyperparams import settings

import jax
import jax.numpy as jnp
import numpy as np

from src import lagranx as lx

import stable_baselines3.common.save_util as loader


def update_buffer(sample, buffer):
    return np.insert(buffer[0:-1], 0, sample)


if __name__ == '__main__':
    print('Snake learned controller initiated.')
    clnt = ln.client(sys.argv[0], sys.argv)
    subscriber = clnt.subscribe("ff.controller.in", "ff_controller_in")
    publisher = clnt.publish("ff.controller.out", "ff_controller_out")
    print('LN connection established')

    # Load the trained model
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(settings, 0, params=params)
    kinetic = lx.learned_lagrangian(params, train_state, output='kinetic')
    potential = lx.learned_lagrangian(params, train_state, output='potential')
    compiled_dynamics = jax.jit(partial(lx.calc_dynamics,
                                        kinetic=kinetic,
                                        potential=potential))
    fd = jax.jit(partial(lx.forward_dynamics,
                         dynamics=compiled_dynamics))

    # Setup the state buffering
    buffer_length = settings['buffer_length']
    q1_buff = np.zeros(buffer_length - 1)
    q2_buff = np.zeros(buffer_length - 1)
    dq1_buff = np.zeros(buffer_length - 1)
    dq2_buff = np.zeros(buffer_length - 1)

    try:
        counter = 0
        prev_time = 0
        prev_update_time = 0
        rate = 100
        while True:
            # Get q, dq, ddq, time
            print('Listening to topic...')
            subscriber.read()
            time = subscriber.packet.time
            q = np.array(subscriber.packet.q)
            dq = np.array(subscriber.packet.dq)
            ddq = np.array(subscriber.packet.ddq)
            print(f"received: {q}, {dq}, {ddq}")

            # Format into the right sizes
            q = lx.normalize(q)

            # Count to 10 and update the buffer!
            if time != prev_time:
                counter += 1
            if counter % 9 == 0:
                # prepare package
                q1_out = np.insert(q1_buff, 0, q[0])
                q2_out = np.insert(q2_buff, 0, q[0])
                q_out = np.concatenate([q1_out, q2_out])

                dq1_out = np.insert(dq1_buff, 0, dq[0])
                dq2_out = np.insert(dq2_buff, 0, dq[0])
                dq_out = np.concatenate([dq1_out, dq2_out])
                state = np.concatenate([q_out, dq_out, np.array([0, 0])])

                # Calculate tau
                tau, _ = fd(ddq=ddq, state=state)

                # Send tau
                publisher.packet.tau = tau
                # print(f"sending: {tau}")
                publisher.write()

                # Do updates
                rate = rate * 0.99 + 0.01 / (time - prev_update_time)
                # print(f"Update rate: {rate}")
                # print(f"Previous q buffer: {q_buff}")
                q1_buff = update_buffer(q[0], q1_buff)
                q2_buff = update_buffer(q[1], q2_buff)
                dq1_buff = update_buffer(dq[0], dq1_buff)
                dq2_buff = update_buffer(dq[1], dq2_buff)
                counter = 0
                # print(f"Current q buffer: {q_buff}")
                prev_update_time = time

            # update vars
            prev_time = time

    except KeyboardInterrupt:
        print('User commanded termination.')

    print('Program terminated successfully.')
