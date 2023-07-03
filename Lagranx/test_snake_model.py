import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3.common.save_util as loader
from src import dpend_model_arne as model

from src import lagranx as lx
from src import trainer
from src import utils
from src import  snake_utils

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
def calc_V_ana(q):
    # q, theta = jnp.split(angles, 2)
    # K = 1.75
    # x = q - theta
    # return 1/2 * K * jnp.sum(x ** 2)
    K_small = jnp.array([[1.75, 0], [0, 1.75]])
    K = jnp.block([[K_small, -K_small], [-K_small, K_small]])
    return 1/2 * jnp.transpose(q) @ K @ q


def build_dynamics(params, train_state):
    # Create basic building blocks
    kinetic = lx.energy_func(params, train_state, output='kinetic')
    potential = lx.energy_func(params, train_state, output='potential')
    # lagrangian = lx.energy_func(params, train_state, output='lagrangian')
    friction = lx.energy_func(params, train_state, output='friction')
    inertia = lx.energy_func(params, train_state, output='inertia')

    # New dynamics
    # dyn_builder_lan = partial(lx.lagrangian_dyn_builder,
    #                           lagrangian=lagrangian,
    #                           friction=friction)
    # dyn_builder_ene = partial(lx.energy_dyn_builder,
    #                           kinetic=kinetic,
    #                           potential=potential,
    #                           lagrangian=lagrangian,
    #                           friction=friction)
    dyn_builder_ine = partial(lx.inertia_dyn_builder,
                              kinetic=kinetic,
                              potential=potential,
                              inertia=inertia,
                              friction=friction)
    dyn_builder_ine_compiled = jax.vmap(jax.jit(dyn_builder_ine))

    energy_calcs = jax.vmap(jax.jit(partial(lx.energy_calcuations,
                                            kinetic=kinetic,
                                            potential=potential)))

    return dyn_builder_ine_compiled, energy_calcs


def handle_data(cursor):
    query = f'SELECT * FROM {table_name} ' \
            f'LIMIT {samples_num} ' \
            f'OFFSET {offset_num}'

    data_raw = jnp.array(cursor.execute(query).fetchall())
    format_samples = jax.vmap(partial(snake_utils.format_sample,
                                      buffer_length=20,
                                      buffer_length_max=20))
    data_formatted = format_samples(data_raw)
    state, ddq_target = jax.vmap(snake_utils.split_data)(data_formatted)
    q, _, dq, _, _ = jax.vmap(partial(snake_utils.split_state, buffer_length=20))(state)
    # q, dq, _, _, _, _, _, _ = vectorized_dynamics(state)

    return (q, dq, ddq_target), state


if __name__ == "__main__":
    # load model
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = trainer.create_train_state(settings, 0,
                                             params=params)

    # load data
    database = sqlite3.connect('/home/gonz_jm/Documents/thesis_workspace/databases/'
                               'database_points_20buff_command')
    table_name = 'point_2'
    samples_num = 600
    offset_num = 150
    cursor = database.cursor()

    # build dynamics
    dyn_builder_compiled, energy_calcs = build_dynamics(params, train_state)

    calc_V_ana_vec = jax.vmap(calc_V_ana)

    # format data
    (q, dq, ddq_target), state = handle_data(cursor)

    # calculate forward and inverse dynamics
    dyn_terms = dyn_builder_compiled(state)
    ddq_pred = jax.vmap(lx.forw_dyn)(dyn_terms)
    tau_pred, tau_target, tau_loss = jax.vmap(lx.inv_dyn)(ddq=ddq_target,
                                                          terms=dyn_terms)

    # calculate energies
    T_lnn, V_lnn, _, _, _, _, (pow_V, pow_T, pow_input, pow_f) = \
        energy_calcs(batch=(state, ddq_target),
                     dyn_terms=dyn_terms)

    # (tau, tau_target, tau_loss), _, _, T_rec, gravity = for_dyn(data_formatted)
    # ddq = inv_dyn(state)

    # calculate losses
    def loss_split(sample):
        loss_func = jax.jit(partial(lx.loss_instant,
                        params,
                        train_state,
                        ))
        losses = loss_func(sample)
        return jnp.split(losses, 3)
    loss_func = jax.vmap(loss_split)
    (L_acc_qdd, L_acc_tau, L_pot) = loss_func((state, ddq_target))

    # Calculate measured energies
    H_mec = cumtrapz(jax.device_get(pow_input), dx=1 / 100, initial=0)
    H_loss = cumtrapz(jax.device_get(pow_f), dx=1 / 100, initial=0)
    H_cor = H_mec + H_loss

    # calculate magnitudes of interest
    V_ana = calc_V_ana_vec(q)
    T_cor = H_cor - V_ana

    # power_vars = jnp.array(list(zip(dq, tau_target)))
    # loss_vars = jnp.array(list(zip(dq, tau_loss)))
    # power = calc_power_vec(power_vars)
    # power_loss = calc_power_vec(loss_vars)

    # Calibration
    [alpha_V, beta_V], V_cal = utils.calibrate(V_ana, V_lnn)
    print(f'Factors: {alpha_V}, {beta_V}.')
    [alpha_T, beta_T], T_cal = utils.calibrate(T_cor, T_lnn)
    print(f'Factors: {alpha_T}, {beta_T}.')

    coef_V = [0.010010098107159138, -0.23374556005001068]
    coef_T = [0.19293320178985596, -0.06825241446495056]
    diff_beta = coef_T[1] - coef_V[0]

    # T_cal = T_lnn * alpha_T
    H_cal = T_cal + V_cal - H_loss
    L_cal = T_cal - V_cal

    H_ana = T_cal + V_ana
    L_ana = T_cal - V_ana

    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    V_ana += diff_beta
    V_f = V_lnn * coef_V[0] + coef_V[1] + diff_beta
    T_f = T_lnn * coef_T[0] + coef_T[1] - diff_beta
    H_f = T_f + V_f
    L_f = T_f - V_f

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

    # Frictions
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(- tau_loss / dq, linewidth=2, label='k_f')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Friction coeffs.')
    plt.ylabel('Something')
    plt.xlabel('sample (n)')
    plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

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
    # plt.legend([r'$\theta_1$', r'$\theta_2$',
    #             r'$\theta^p_1$', r'$\theta^p_2$',
    #             r'$\theta^f_1$', r'$\theta^f_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Losses
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(L_acc_qdd, linewidth=2, label='ddq')
    plt.plot(10 * L_acc_tau, linewidth=2, label='tau')
    plt.plot(100 * L_pot, linewidth=2, label='pot')
    # plt.plot(L_mass, linewidth=2, label='mass')
    # plt.plot(L_kin_pos, linewidth=2, label='kin_pos')
    # plt.plot(L_kin_shape * 100, linewidth=2, label='kin_shape')
    # plt.plot(L_dV_shape * 10000, linewidth=2, label='pot_shape')
    # plt.plot(L_ham, linewidth=2, label='hamiltonian')
    # plt.plot(10000 * L_con, linewidth=2, label='conservative')
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

    # Powers
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(pow_input, linewidth=2, label='Power input.')
    # plt.plot(H_lnn, linewidth=2, label='H. Lnn')
    # plt.plot(H_rec, linewidth=2, label='H. Recon')
    # plt.plot(H_mec, linewidth=2, label='H. Mec')
    # plt.plot(H_loss, linewidth=2, label='H. Loss')
    # plt.plot(H_cor, linewidth=2, label='H. Cor')
    # plt.plot(H_cal, linewidth=2, label='H. Cal')
    # plt.plot(t_sim, H_f, '--', linewidth=3, label='H. Final')
    plt.title('Powers')
    # plt.ylim(0, 2.0)
    # plt.xlim(0, 5)
    plt.ylabel('Power (J/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    # plt.legend([r'$H_{ana}$', r'$H_{nn}$', r'$H_{calib.}$'], loc="best")
    # plt.savefig('media/Model identification/Hamiltonian.png')
    plt.show()

    # Energies
    plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(H_ana, linewidth=2, label='H. Ana')
    # plt.plot(H_lnn, linewidth=2, label='H. Lnn')
    # plt.plot(H_rec, linewidth=2, label='H. Recon')
    plt.plot(H_mec, linewidth=2, label='H. Mec')
    plt.plot(H_loss, linewidth=2, label='H. Loss')
    plt.plot(H_cor, linewidth=2, label='H. Cor')
    # plt.plot(H_cal, linewidth=2, label='H. Cal')
    # plt.plot(H_f, linewidth=3, label='H. Final')
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
    # plt.plot(L_cal, linewidth=2, label='L. Cal')
    plt.plot(L_f, linewidth=2, label='L. Final')
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
    # plt.plot(T_rec, linewidth=2, label='Kin. Recon')
    # plt.plot(V_rec, linewidth=2, label='Pot. Recon')
    # plt.plot(T_cal, linewidth=2, label='Kin. Cal')
    # plt.plot(V_cal, linewidth=2, label='Pot. Cal')
    plt.plot(T_f, linewidth=2, label='Kin. Final')
    plt.plot(V_f, linewidth=2, label='Pot. Final')
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
