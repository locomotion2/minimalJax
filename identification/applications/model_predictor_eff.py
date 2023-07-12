import sys
from functools import partial

import jax
import jax.numpy as jnp
import links_and_nodes as ln
import numpy as np
import stable_baselines3.common.save_util as loader

import identification.identification_utils as utils
from identification.hyperparams import settings
from identification.src.dynamix import energiex as ex, motionx as mx, wrappings
from identification.src.learning import trainer
from identification.systems import snake_utils


if __name__ == '__main__':
    # config ln coms
    print('Snake learned controller initiated.')
    clnt = ln.client(sys.argv[0], sys.argv)
    subscriber = clnt.subscribe("ff.controller.in", "ff_controller_in")
    publisher = clnt.publish("ff.controller.out", "ff_controller_out")
    print('LN connection established')

    # load the trained models
    settings['system_settings']['sys_utils'] = snake_utils
    params_energy = loader.load_from_pkl(path=settings['model_settings']['ckpt_dir'],
                                         verbose=1)
    params_model = loader.load_from_pkl(path=settings['model_settings'][
        'ckpt_dir_model'],
                                        verbose=1)
    train_state_energies = trainer.create_train_state("energy",
                                                      settings, 0, params=params_energy)
    train_state_model = trainer.create_train_state("model",
                                                   settings, 0, params=params_model)

    # build the functions to be called in the loop
    dyns_compiled = wrappings.build_dynamics("model",
                                             settings,
                                             params_model,
                                             train_state_model)
    energy_call_compiled = wrappings.build_energy_call(settings,
                                                       params_energy,
                                                       train_state_energies)
    # blen = settings['model_settings']['buffer_length']
    # blen_max = settings['model_settings']['buffer_length_max']
    # formatting_tool = partial(snake_utils.format_sample,
    #                                   buffer_length=blen,
    #                                   buffer_length_max=blen_max)

    # set up the state buffering
    print('Acc. and torque prediction will now begin!')
    try:
        while True:
            # Get q, dq, ddq, time
            subscriber.read()
            time = subscriber.packet.time
            q_buff = jnp.array(subscriber.packet.q)
            dq_buff = jnp.array(subscriber.packet.dq)
            ddq = jnp.array(subscriber.packet.ddq)
            tau_target = jnp.array(subscriber.packet.tau)
            print(f"received: {ddq}, {tau_target}")

            # Format input
            q_buff = utils.wrap_angle(q_buff)
            state_raw = jnp.concatenate([q_buff, dq_buff, tau_target])
            # state = formatting_tool(state_raw)

            # Calculate dynamics
            dyn_terms = dyns_compiled(state_raw)
            ddq_pred = mx.forward_dynamics(dyn_terms)
            tau_pred, _, _ = mx.inverse_dynamics(ddq=ddq,
                                                 terms=dyn_terms)
            T, V = energy_call_compiled(state_raw)

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
