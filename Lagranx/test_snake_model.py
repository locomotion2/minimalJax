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

from scipy.integrate import cumtrapz

import seaborn as sns

if __name__ == "__main__":
    # load model
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(settings, 0,
                                        params=params)

    # build dynamics
    kinetic = lx.learned_lagrangian(params, train_state, output='kinetic')
    potential = lx.learned_lagrangian(params, train_state, output='potential')
    friction = lx.learned_lagrangian(params, train_state, output='friction')
    compiled_dynamics = partial(lx.calc_dynamics,
                                kinetic=kinetic,
                                potential=potential,
                                friction=friction)
    compiled_dyn_wrapper = jax.jit(partial(lx.dynamics_wrapper,
                                           dynamics=compiled_dynamics))
    format_samples = jax.vmap(partial(lx.format_sample,
                                      buffer_length=10,
                                      buffer_length_max=10))
    inv_dyn = jax.vmap(jax.jit(partial(lx.equation_of_motion,
                                       dynamics=compiled_dyn_wrapper)))
    for_dyn_single = jax.jit(partial(lx.forward_dynamics,
                                     dynamics=compiled_dyn_wrapper))
    energies = jax.vmap(partial(lx.learned_energies,
                                params=params,
                                train_state=train_state))
    vectorized_dynamics = jax.vmap(compiled_dyn_wrapper)


    # wrap fd to handle data format
    def for_dyn_wrapped(data_point):
        state, ddq = data_point
        ddq = ddq[4:8]
        return for_dyn_single(ddq=ddq, state=state)


    for_dyn = jax.vmap(for_dyn_wrapped)

    # set up database
    database = sqlite3.connect('/home/gonz_jm/Documents/thesis_workspace/databases/'
                               'database_points_full')
    table_name = 'point_123'
    samples_num = 1000
    offset_num = 0
    cursor = database.cursor()


    # define query
    def query():
        query = f'SELECT * FROM {table_name} ' \
                f'LIMIT {samples_num} ' \
                f'OFFSET {offset_num}'
        return query


    @jax.jit
    def calc_power(vars):
        dq, tau = vars
        # dq = dq[0:2]
        # tau = tau[0:2]
        return jnp.transpose(dq) @ tau


    calc_power_vec = jax.vmap(calc_power)


    @jax.jit
    def calc_V_ana(tau):
        return 1 / 2 * 1 / 1.75 * jnp.sum(tau ** 2)


    calc_V_ana_vec = jax.vmap(calc_V_ana)

    # format data
    data_raw = jnp.array(cursor.execute(query()).fetchall())
    data_formatted = format_samples(data_raw)
    state, ddq_target = data_formatted
    # q, _, dq, _, _ = jax.vmap(partial(lx.split_state, buffer_length=10))(state)
    q, dq, _, _, _, _, _, k_f = vectorized_dynamics(state)

    # calculate magnitudes of interest
    tau, tau_target = for_dyn(data_formatted)
    tau_loss = k_f * dq
    ddq = inv_dyn(state)
    V_ana = calc_V_ana_vec(tau_target)
    T_lnn, V_lnn, T_rec, _, _ = jax.device_get(energies(state))
    T_lnn = - T_lnn
    V_lnn = - V_lnn
    T_rec = - T_rec

    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn
    V_rec = V_lnn
    L_rec = T_rec - V_rec
    H_rec = T_rec + V_rec
    vars = jnp.array(list(zip(dq, tau_target)))
    power = calc_power_vec(vars)
    H_mec = cumtrapz(jax.device_get(power), dx=1 / 100)

    # Calibration
    # state_rest = jnp.array([[0] * (settings['buffer_length'] * 8)], dtype=float)
    # T_rest, V_rest, T_rec_rest, _, _ = jax.device_get(energies(state_rest))
    # T_rest = -T_rest
    # V_rest = -V_rest
    # T_rec_rest = -T_rec_rest
    # print(f"Rest-> kin: {T_rec_rest}, pot: {V_rest}")
    #
    # state_flex = jnp.array([([np.radians(45)] * (settings['buffer_length'])) +
    #                        [0] * (settings['buffer_length'] * 7)], dtype=float)
    # T_flex, V_flex, T_rec_flex, _, _ = jax.device_get(energies(state_flex))
    # V_known = 1/2 * 1.75 * np.radians(45) ** 2
    # T_flex = -T_flex
    # V_flex = -V_flex
    # T_rec_flex = -T_rec_flex
    # print(f"Flex-> kin: {T_rec_flex}, pot: {V_flex}, known: {V_known}")
    # beta = V_rest
    # alpha = V_known / (V_flex - beta)
    # print(f"Coefs-> alpha: {alpha}, beta: {beta}")
    [alpha, beta], V_cal = lx.calibrate(V_ana, V_rec)
    print(f'Factors: {alpha}, {beta}.')

    T_cal = T_rec
    H_cal = T_cal + V_cal
    L_cal = T_cal - V_cal

    # Plotting
    sns.set(style="darkgrid")

    # Positions
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(q, linewidth=2, label='q')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Joint positions.')
    plt.ylabel('rad')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Speeds
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(dq, linewidth=2, label='q')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Joint speeds.')
    plt.ylabel('rad/s')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Frictions
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(k_f, linewidth=2, label='k_f')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Friction coeffs.')
    plt.ylabel('Something')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Accelerations
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(ddq_target[:, 6:8], linewidth=2, label='target')
    plt.plot(ddq[:, 6:8], linewidth=2, label='pred')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Comparison between predicted and actual joint acc.')
    plt.ylabel('rad/s^2')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Torques
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(tau_target[:, 2:4], linewidth=2, label='target')
    plt.plot(tau[:, 2:4], linewidth=2, label='pred')
    plt.plot(tau_loss[:, 2:4], linewidth=2, label='Loss')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Comparison between predicted and actual joint torques.')
    plt.ylabel('Nm')
    plt.xlabel('sample (n)')
    # plt.legend([r'$L_{tot}$', r'$L_{acc}$', r'$L_{mec}$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Energies
    plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(H_ana, linewidth=3, label='H. Analytic')
    # plt.plot(H_lnn, linewidth=2, label='H. Lnn')
    plt.plot(H_rec, linewidth=2, label='H. Recon')
    plt.plot(H_mec, linewidth=2, label='H. Mec')
    # plt.plot(H_cal, linewidth=2, label='H. Cal')
    # plt.plot(t_sim, H_f, '--', linewidth=3, label='H. Final')
    plt.title('Hamiltonians from the Snake')
    # plt.ylim(0, 2.0)
    # plt.xlim(0, 5)
    plt.ylabel('Energy (J)')
    plt.xlabel('Time (s)')
    plt.legend()
    # plt.legend([r'$H_{ana}$', r'$H_{nn}$', r'$H_{calib.}$'], loc="best")
    # plt.savefig('media/Model identification/Hamiltonian.png')
    plt.show()

    plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(t_sim, L_ana, linewidth=3, label='L. Analytic')
    # plt.plot(L_lnn, linewidth=2, label='L. Lnn')
    plt.plot(L_rec, linewidth=2, label='L. Recon')
    # plt.plot(L_cal, linewidth=2, label='L. Cal')
    # plt.plot(t_sim, L_f, '--', linewidth=2, label='L. Final')
    plt.title('Lagrangians from the Snake')
    # plt.ylim(-1.5, 2.0)
    # plt.xlim(0, 5)
    plt.ylabel('Energy (J)')
    plt.xlabel('Time (s)')
    plt.legend()
    # plt.legend([r'$l_{ana}$', r'$L_{nn}$', r'$L_{calib.}$'], loc="best")
    # plt.savefig('media/Model identification/Lagrangian.png')
    plt.show()

    # Energies
    plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(t_sim, T_ana, linewidth=3, label='Kin. Analytic')
    plt.plot(V_ana, linewidth=2, label='Pot. Ana')
    # plt.plot(T_lnn, linewidth=2, label='Kin. Lnn.')
    # plt.plot(V_lnn, linewidth=2, label='Pot. Lnn.')
    plt.plot(T_rec, linewidth=2, label='Kin. Recon')
    plt.plot(V_rec, linewidth=2, label='Pot. Recon')
    # plt.plot(T_cal, linewidth=2, label='Kin. Cal')
    # plt.plot(V_cal, linewidth=2, label='Pot. Cal')
    # plt.plot(t_sim, T_f, '--', linewidth=2, label='Kin. Final')
    # plt.plot(t_sim, V_f, '--', linewidth=2, label='Pot. Final')
    plt.title('Energies from the Snake')
    # plt.ylim(-0.25, 2.0)
    # plt.xlim(0, 5)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend()
    # plt.legend([r'$T_{ana}$', r'$V_{ana}$', r'$T_{nn}$', r'$V_{nn}$'], loc="best")
    # plt.savefig('media/Model identification/Energies.png')
    plt.show()

    # Losses
