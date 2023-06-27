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


def query():
    return query


# @jax.jit
# def calc_power(vars):
#     dq, tau = vars
#     # tau = jnp.array([tau[3], tau[4], 0, 0])
#     return jnp.transpose(dq) @ tau
#
#
# @jax.jit
# def calc_power_loss(vars):
#     dq, k_f = vars
#     tau = k_f * dq
#     return jnp.transpose(dq) @ tau


@jax.jit
def calc_V_ana(tau):
    return 1 / 2 * 1 / 1.75 * jnp.sum(tau ** 2)


def build_dynamics(params, train_state):
    # Create basic building blocks
    kinetic = lx.energy_func(params, train_state, output='kinetic')
    potential = lx.energy_func(params, train_state, output='potential')
    lagrangian = lx.energy_func(params, train_state, output='lagrangian')
    friction = lx.energy_func(params, train_state, output='friction')
    inertia = lx.energy_func(params, train_state, output='inertia')

    # # Build modelled dynamics
    # compiled_dynamics = partial(lx.calc_dynamics,
    #                             kinetic=kinetic,
    #                             potential=potential,
    #                             friction=friction)
    # compiled_dyn_wrapper = jax.jit(partial(lx.dynamic_matrices,
    #                                        dynamics=compiled_dynamics))
    # vectorized_dynamics = jax.vmap(compiled_dyn_wrapper)
    # inv_dyn = jax.vmap(jax.jit(partial(lx.equation_of_motion,
    #                                    dynamics=compiled_dyn_wrapper)))
    # for_dyn_single = jax.jit(partial(lx.forward_dynamics,
    #                                  dynamics=compiled_dyn_wrapper))
    #
    # def for_dyn_wrapped(data_point):
    #     state, ddq = data_point
    #     ddq = ddq[4:8]
    #     return for_dyn_single(ddq=ddq, state=state)
    #
    # for_dyn = jax.vmap(for_dyn_wrapped)

    # New dynamics
    dyn_builder_lan = partial(lx.lagrangian_dyn_builder,
                          lagrangian=lagrangian,
                          friction=friction)
    dyn_builder_ene = partial(lx.energy_dyn_builder,
                          kinetic=kinetic,
                          potential=potential,
                          lagrangian=lagrangian,
                          friction=friction)
    dyn_builder_ine = partial(lx.inertia_dyn_builder,
                          kinetic=kinetic,
                          potential=potential,
                          inertia=inertia,
                          friction=friction)
    dyn_builder_ine_compiled = jax.vmap(jax.jit(dyn_builder_ine))
    dyn_builder_ene_compiled = jax.vmap(jax.jit(dyn_builder_ene))
    dyn_builder_lan_compiled = jax.vmap(jax.jit(dyn_builder_lan))

    energy_calcs = jax.vmap(jax.jit(partial(lx.energy_calcuations,
                                            kinetic=kinetic,
                                            potential=potential)))

    return dyn_builder_ine_compiled, energy_calcs


def handle_data(cursor):
    query = f'SELECT * FROM {table_name} ' \
            f'LIMIT {samples_num} ' \
            f'OFFSET {offset_num}'

    data_raw = jnp.array(cursor.execute(query).fetchall())
    data_formatted = format_samples(data_raw)
    state, ddq_target = jax.vmap(lx.split_data)(data_formatted)
    q, _, dq, _, _ = jax.vmap(partial(lx.split_state, buffer_length=10))(state)
    # q, dq, _, _, _, _, _, _ = vectorized_dynamics(state)

    return (q, dq, ddq_target), state


if __name__ == "__main__":
    # load model
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(settings, 0,
                                        params=params)

    # load data
    database = sqlite3.connect('/home/gonz_jm/Documents/thesis_workspace/databases/'
                               'database_points_fixed')
    table_name = 'point_55'
    samples_num = 300
    offset_num = 150
    cursor = database.cursor()

    # build dynamics
    dyn_builder_compiled, energy_calcs = build_dynamics(params, train_state)

    # utilities
    format_samples = jax.vmap(partial(lx.format_sample,
                                      buffer_length=10,
                                      buffer_length_max=10))

    calc_V_ana_vec = jax.vmap(calc_V_ana)

    # format data
    (q, dq, ddq_target), state = handle_data(cursor)

    # calculate forward and inverse dynamics
    dyn_terms = dyn_builder_compiled(state)
    ddq_pred = jax.vmap(lx.inv_dyn_lagrangian)(dyn_terms)
    tau_pred, tau_target, tau_loss = jax.vmap(lx.for_dyn_lagrangian)(ddq=ddq_target,
                                                           terms=dyn_terms)

    # calculate energies
    T_lnn, V_lnn, T_rec, _, _, _, (pow_V, pow_T, pow_input, pow_f) = \
        energy_calcs(batch=(state, ddq_target),
                     dyn_terms=dyn_terms)

    # (tau, tau_target, tau_loss), _, _, T_rec, gravity = for_dyn(data_formatted)
    # ddq = inv_dyn(state)

    # calculate losses
    loss_func = jax.vmap(jax.jit(partial(lx.loss_instant,
                                         params,
                                         train_state,
                                         )))
    # (L_acc_qdd, L_acc_tau), \
    #     (L_mass, L_kin_pos, L_kin_shape, L_dV_shape), \
    #     L_con = loss_func((state, ddq_target))

    (L_acc_qdd, L_acc_tau) = loss_func((state, ddq_target))

    # calculate magnitudes of interest
    V_ana = calc_V_ana_vec(tau_target)
    H_ana = T_rec + V_ana
    L_ana = T_rec - V_ana

    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    V_rec = V_lnn
    L_rec = T_rec - V_rec
    H_rec = T_rec + V_rec

    # power_vars = jnp.array(list(zip(dq, tau_target)))
    # loss_vars = jnp.array(list(zip(dq, tau_loss)))
    # power = calc_power_vec(power_vars)
    # power_loss = calc_power_vec(loss_vars)

    H_mec = cumtrapz(jax.device_get(pow_input), dx=1 / 100, initial=0)
    H_loss = cumtrapz(jax.device_get(pow_f), dx=1 / 100, initial=0)
    H_cor = H_mec + H_loss

    # Calibration
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
    plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
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
    plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # # Frictions
    # plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(k_f, linewidth=2, label='k_f')
    # plt.legend()
    # # plt.ylim(-0.1, 0.5)
    # # plt.xlim(0, 5)
    # plt.title('Friction coeffs.')
    # plt.ylabel('Something')
    # plt.xlabel('sample (n)')
    # plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
    # # plt.savefig('media/Model identification/Loss.png')
    # plt.show()

    # Accelerations
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(ddq_target[:, 4:8], linewidth=3, label='target')
    plt.plot(ddq_pred[:, 4:8], linewidth=3, linestyle='--', label='pred')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Accelerations')
    plt.ylabel('rad/s^2')
    plt.xlabel('sample (n)')
    plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$',
                r'$q^p_1$', r'$q^p_2$', r'$\theta^p_1$', r'$\theta^p_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Torques
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(tau_target[:, 2:4], linewidth=3, label='target')
    plt.plot(tau_pred[:, 2:4], linewidth=3, linestyle='--', label='pred')
    plt.plot(tau_loss[:, 2:4], linewidth=2, linestyle='--', label='loss')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Motor torques')
    plt.ylabel('Nm')
    plt.xlabel('sample (n)')
    plt.legend([r'$\theta_1$', r'$\theta_2$',
                r'$\theta^p_1$', r'$\theta^p_2$',
                r'$\theta^f_1$', r'$\theta^f_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Losses
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(L_acc_qdd, linewidth=2, label='ddq')
    plt.plot(L_acc_tau * 10000, linewidth=2, label='tau')
    # plt.plot(L_mass, linewidth=2, label='mass')
    # plt.plot(L_kin_pos, linewidth=2, label='kin_pos')
    # plt.plot(L_kin_shape * 100, linewidth=2, label='kin_shape')
    # plt.plot(L_dV_shape * 100, linewidth=2, label='pot_shape')
    # plt.plot(L_ham, linewidth=2, label='hamiltonian')
    # plt.plot(L_con, linewidth=2, label='conservative')
    # plt.plot(L_loss * 10000, linewidth=2, label='loss')
    plt.legend()
    # plt.ylim(-5, 200)
    # plt.xlim(0, 5)
    plt.title('Losses')
    # plt.ylabel('Nm')
    # plt.xlabel('sample (n)')
    # plt.legend([r'$\theta_1$', r'$\theta_2$',
    #             r'$\theta^p_1$', r'$\theta^p_2$',
    #             r'$\theta^f_1$', r'$\theta^f_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Energies
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(H_ana, linewidth=2, label='H. Ana')
    # plt.plot(H_lnn, linewidth=2, label='H. Lnn')
    # plt.plot(H_rec, linewidth=2, label='H. Recon')
    plt.plot(H_mec, linewidth=2, label='H. Mec')
    plt.plot(H_loss, linewidth=2, label='H. Loss')
    plt.plot(H_cor, linewidth=2, label='H. Cor')
    plt.plot(H_cal, linewidth=2, label='H. Cal')
    # plt.plot(t_sim, H_f, '--', linewidth=3, label='H. Final')
    plt.title('Hamiltonians')
    # plt.ylim(0, 2.0)
    # plt.xlim(0, 5)
    plt.ylabel('Energy (J)')
    plt.xlabel('Time (s)')
    plt.legend()
    # plt.legend([r'$H_{ana}$', r'$H_{nn}$', r'$H_{calib.}$'], loc="best")
    # plt.savefig('media/Model identification/Hamiltonian.png')
    plt.show()

    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(L_ana, linewidth=2, label='L. Ana')
    # plt.plot(L_lnn, linewidth=2, label='L. Lnn')
    # plt.plot(L_rec, linewidth=2, label='L. Recon')
    plt.plot(L_cal, linewidth=2, label='L. Cal')
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
    # plt.plot(V_ana, linewidth=2, label='Pot. Ana')
    plt.plot(T_lnn, linewidth=2, label='Kin. Lnn.')
    # plt.plot(V_lnn, linewidth=2, label='Pot. Lnn.')
    # plt.plot(T_rec, linewidth=2, label='Kin. Recon')
    # plt.plot(V_rec, linewidth=2, label='Pot. Recon')
    plt.plot(T_cal, linewidth=2, label='Kin. Cal')
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
