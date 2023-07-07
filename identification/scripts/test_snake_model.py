import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
import stable_baselines3.common.save_util as loader

from identification.src import lagranx as lx
from identification.src import trainer
from identification.src import utils
from identification.systems import snake_utils

from identification.hyperparams import settings

import sqlite3

from functools import partial

from scipy.integrate import cumtrapz

import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)


@jax.jit
def calc_V_ana(q):
    # q, theta = jnp.split(angles, 2)
    # K = 1.75
    # x = q - theta
    # return 1/2 * K * jnp.sum(x ** 2)
    K_small = jnp.array([[1.75, 0], [0, 1.75]])
    K = jnp.block([[K_small, -K_small], [-K_small, K_small]])
    return 1 / 2 * jnp.transpose(q) @ K @ q


def build_dynamics(params, train_state):
    # Create basic building blocks
    kinetic = lx.energy_func_model(params, train_state, settings=settings,
                                   output='kinetic')
    potential = lx.energy_func_model(params, train_state, settings=settings,
                                     output='potential')
    friction = lx.energy_func_model(params, train_state, settings=settings,
                                    output='friction')
    inertia = lx.energy_func_model(params, train_state, settings=settings,
                                   output='inertia')

    # TODO: Add settings to the input of this function
    split_tool = snake_utils.build_split_tool(20)

    # Create compiled dynamics
    dyn_builder = partial(lx.inertia_dyn_builder,
                          split_tool=split_tool,
                          kinetic=kinetic,
                          potential=potential,
                          inertia=inertia,
                          friction=friction)
    # dyn_builder = partial(lx.energy_dyn_builder,
    #                       split_tool=split_tool,
    #                       potential=potential,
    #                       kinetic=kinetic,
    #                       friction=friction)
    dyn_builder_compiled = jax.jit(dyn_builder)

    # Vectorize some important calculations
    energy_calcs = jax.vmap(jax.jit(partial(lx.test_calculations,
                                            split_tool=split_tool,
                                            kinetic=kinetic,
                                            potential=potential)))

    # build the integrable EOMs
    eom_compiled = partial(lx.eom_wrapped,
                           potential=potential,
                           inertia=inertia)

    return dyn_builder_compiled, energy_calcs, eom_compiled, split_tool


def handle_data(cursor):
    # Get raw data from database
    query = f'SELECT * FROM {table_name} ' \
            f'LIMIT {samples_num} ' \
            f'OFFSET {offset_num}'
    data_raw = jnp.array(cursor.execute(query).fetchall())

    # Format the samples
    format_samples = jax.vmap(partial(snake_utils.format_sample,
                                      buffer_length=20,
                                      buffer_length_max=20))
    data_formatted = format_samples(data_raw)

    # Break the formatted samples into useful magnitudes
    # TODO: Add settings to the input of this function
    split_tool = snake_utils.build_split_tool(20)
    state, ddq_target = jax.vmap(snake_utils.split_data)(data_formatted)
    q, _, dq, _, _ = jax.vmap(split_tool)(state)

    return (q, dq, ddq_target), state, data_formatted


if __name__ == "__main__":
    # load model
    settings['sys_utils'] = snake_utils
    params = loader.load_from_pkl(path=settings['ckpt_dir_model'], verbose=1)
    train_state = trainer.create_train_state_PowNN(settings, 0,
                                                   params=params)

    # load data
    database = sqlite3.connect('/home/gonz_jm/Documents/thesis_workspace/databases'
                               '/database_points_20buff_command_standard')
    table_name = 'point_2'
    samples_num = 10
    offset_num = 150 + 450
    cursor = database.cursor()

    # build dynamics
    dyn_builder_compiled, \
        energy_calcs, \
        eom_prepared, \
        split_tool = build_dynamics(params, train_state)
    calc_V_ana_vec = jax.vmap(calc_V_ana)

    # format and break up data
    (q, dq, ddq_target), state, data_formatted = handle_data(cursor)

    # calculate forward and inverse dynamics
    dyn_terms = jax.vmap(dyn_builder_compiled)(state)
    ddq_pred = jax.vmap(lx.forward_dynamics)(dyn_terms)
    tau_pred, tau_target, tau_loss = jax.vmap(lx.inverse_dynamics)(ddq=ddq_target,
                                                                   terms=dyn_terms)

    # simulate the trajectory
    if settings['simulate']:
        buffer_length = settings['buffer_length']
        sys_utils = settings['sys_utils']
        num_dof = settings['num_dof']
        simulation = partial(lx.simulate,
                             buffer_length=buffer_length,
                             num_dof=num_dof,
                             samples_num=samples_num,
                             sys_utils=sys_utils,
                             split_tool=split_tool,
                             data_formatted=data_formatted,
                             eom_prepared=eom_prepared
                             )
        q_sim, dq_sim = simulation(state=state, tau=tau_target)
        # buffer_length = settings['buffer_length']
        # num_dof = settings['num_dof']
        # state_long, _ = snake_utils.split_data(data_formatted[0])
        # _, q_buff_input, _, dq_buff_input, _ = split_tool(state[0])
        # q_buff_long, dq_buff_long = jnp.split(state_long[:-4], 2)
        # q_buff_wide = jnp.reshape(q_buff_long, (num_dof, buffer_length))
        # dq_buff_wide = jnp.reshape(dq_buff_long, (num_dof, buffer_length))
        # # t = jnp.linspace(0, samples_num - offset_num) / 100
        # x_input = jnp.concatenate([q[0],
        #                            dq[0]])
        # # integ = jax.jit(odeint, backend='cpu')
        # q_sim = jnp.array([q[0]])
        # dq_sim = jnp.array([dq[0]])
        # for index, tau_cur in enumerate(tau_target):
        #     print(f"Progress: {index / samples_num *  100}")
        #     # print(q_buff_wide)
        #
        #     # prepare the equation
        #     eom_compiled = partial(eom_prepared,
        #                            q_buff=q_buff_input,
        #                            dq_buff=dq_buff_input)
        #
        #     # calculate the enchilada
        #     # out = lx.solve_eom(x_input,
        #     #                    eom_compiled,
        #     #                    jnp.array([0, 1 / 100]),
        #     #                    tau_cur)
        #     out = odeint(eom_compiled,
        #                  x_input,
        #                  jnp.array([0, 1 / 100]),
        #                  (tau_cur,))
        #
        #     # update the values
        #     x_input = out[-1]
        #     q_cur, dq_cur = jnp.split(x_input, 2)
        #     (q_buff_input, dq_buff_input), \
        #         (q_buff_wide, dq_buff_wide) = utils.format_state_sim(q_cur,
        #                                                              q_buff_wide,
        #                                                              dq_cur,
        #                                                              dq_buff_wide)
        #
        #     # save in vectors
        #     q_sim = jnp.append(q_sim, jnp.array([q_cur]), axis=0)
        #     dq_sim = jnp.append(dq_sim, jnp.array([dq_cur]), axis=0)
        # print(q_sim.shape)

    # calculate energies and powers
    T_lnn, V_lnn, powers = energy_calcs(batch=(state, ddq_target),
                                        dyn_terms=dyn_terms)
    pow_V, pow_T, pow_input, pow_f = powers


    # calculate losses
    def loss_split(sample):
        losses = lx.loss_sample(split_tool,
                                dyn_builder_compiled,
                                sample)
        return jnp.split(losses, 3)


    # loss_func = jax.vmap(loss_split)
    # (L_acc_qdd, L_acc_tau, L_pot) = loss_func((state, ddq_target))

    # TODO: Load the timestep from settings
    # Calculate measured energies
    H_mec = cumtrapz(jax.device_get(pow_input), dx=1 / 100, initial=0)
    H_loss = cumtrapz(jax.device_get(pow_f), dx=1 / 100, initial=0)
    H_cor = H_mec + H_loss

    # calculate NN energies
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    # calculate analytical energies
    V_ana = calc_V_ana_vec(q)
    T_cor = H_cor - V_ana

    # calibrate the NN output
    [alpha_V, beta_V], V_cal = utils.calibrate(V_ana, V_lnn)
    print(f'Factors: {alpha_V}, {beta_V}.')
    [alpha_T, beta_T], T_cal = utils.calibrate(T_cor, T_lnn)
    print(f'Factors: {alpha_T}, {beta_T}.')

    # saved coefficients after calibration
    coef_V = [1, 0]
    coef_T = [0.1795465350151062, -0.06468642503023148]
    diff_beta = coef_T[1] - coef_V[1]

    # calculate calibrated energies
    H_cal = T_cal + V_cal - H_loss
    L_cal = T_cal - V_cal

    # use the calibrated kin. to approximate the analytical lagrangian
    H_ana = T_cal + V_ana
    L_ana = T_cal - V_ana

    # V_ana += diff_beta
    V_f = V_lnn * coef_V[0] + coef_V[1]
    T_f = T_lnn * coef_T[0]
    H_f = T_f + V_f - H_loss
    L_f = T_f - V_f

    # TODO: Improve plotting
    # Plotting
    sns.set(style="darkgrid")

    # Positions
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(q, linewidth=2, label='q')
    if settings['simulate']:
        plt.plot(q_sim, linewidth=2, label='q_sim')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Joint positions.')
    plt.ylabel('rad')
    plt.xlabel('sample (n)')
    # plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Speeds
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(dq, linewidth=2, label='dq')
    if settings['simulate']:
        plt.plot(dq_sim, linewidth=2, label='dq_sim')
    plt.legend()
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    plt.title('Joint speeds.')
    plt.ylabel('rad/s')
    plt.xlabel('sample (n)')
    # plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # Frictions
    # plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(- tau_loss / dq, linewidth=2, label='k_f')
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
    # plt.legend([r'$\theta_1$', r'$\theta_2$',
    #             r'$\theta^p_1$', r'$\theta^p_2$',
    #             r'$\theta^f_1$', r'$\theta^f_2$'], loc="best")
    # plt.savefig('media/Model identification/Loss.png')
    plt.show()

    # # Losses
    # plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(L_acc_qdd, linewidth=2, label='ddq')
    # plt.plot(10 * L_acc_tau, linewidth=2, label='tau')
    # plt.plot(100 * L_pot, linewidth=2, label='pot')
    # # plt.plot(L_mass, linewidth=2, label='mass')
    # # plt.plot(L_kin_pos, linewidth=2, label='kin_pos')
    # # plt.plot(L_kin_shape * 100, linewidth=2, label='kin_shape')
    # # plt.plot(L_dV_shape * 10000, linewidth=2, label='pot_shape')
    # # plt.plot(L_ham, linewidth=2, label='hamiltonian')
    # # plt.plot(10000 * L_con, linewidth=2, label='conservative')
    # # plt.plot(L_loss * 10000, linewidth=2, label='loss')
    # plt.legend()
    # # plt.ylim(-5, 200)
    # # plt.xlim(0, 5)
    # plt.title('Losses')
    # # plt.ylabel('Nm')
    # # plt.xlabel('sample (n)')
    # # plt.legend([r'$\theta_1$', r'$\theta_2$',
    # #             r'$\theta^p_1$', r'$\theta^p_2$',
    # #             r'$\theta^f_1$', r'$\theta^f_2$'], loc="best")
    # # plt.savefig('media/Model identification/Loss.png')
    # plt.show()

    # # Powers
    # plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(pow_input, linewidth=2, label='Power input.')
    # # plt.plot(H_lnn, linewidth=2, label='H. Lnn')
    # # plt.plot(H_rec, linewidth=2, label='H. Recon')
    # # plt.plot(H_mec, linewidth=2, label='H. Mec')
    # # plt.plot(H_loss, linewidth=2, label='H. Loss')
    # # plt.plot(H_cor, linewidth=2, label='H. Cor')
    # # plt.plot(H_cal, linewidth=2, label='H. Cal')
    # # plt.plot(t_sim, H_f, '--', linewidth=3, label='H. Final')
    # plt.title('Powers')
    # # plt.ylim(0, 2.0)
    # # plt.xlim(0, 5)
    # plt.ylabel('Power (J/s)')
    # plt.xlabel('Time (s)')
    # plt.legend()
    # # plt.legend([r'$H_{ana}$', r'$H_{nn}$', r'$H_{calib.}$'], loc="best")
    # # plt.savefig('media/Model identification/Hamiltonian.png')
    # plt.show()

    # Energies
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(H_ana, linewidth=2, label='H. Ana')
    # plt.plot(H_lnn, linewidth=2, label='H. Lnn')
    # plt.plot(H_rec, linewidth=2, label='H. Recon')
    plt.plot(H_mec, linewidth=2, label='H. Mec')
    plt.plot(H_loss, linewidth=2, label='H. Loss')
    plt.plot(H_cor, linewidth=2, label='H. Cor')
    plt.plot(H_cal, linewidth=2, label='H. Cal')
    plt.plot(H_f, linewidth=3, label='H. Final')
    plt.title('Hamiltonians')
    # plt.ylim(0, 2.0)
    # plt.xlim(0, 5)
    plt.ylabel('Energy (J)')
    plt.xlabel('Time (s)')
    plt.legend()
    # plt.legend([r'$H_{ana}$', r'$H_{nn}$', r'$H_{calib.}$'], loc="best")
    # plt.savefig('media/Model identification/Hamiltonian.png')
    plt.show()

    # Lagrangians
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(L_ana, linewidth=2, label='L. Ana')
    # plt.plot(L_lnn, linewidth=2, label='L. Lnn')
    # plt.plot(L_rec, linewidth=2, label='L. Recon')
    plt.plot(L_cal, linewidth=2, label='L. Cal')
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
    plt.plot(T_cal, linewidth=2, label='Kin. Cal')
    plt.plot(V_cal, linewidth=2, label='Pot. Cal')
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
