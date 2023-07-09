import warnings

import jax
import jax.numpy as jnp
import stable_baselines3.common.save_util as loader

from identification.src.dynamix import lagranx as lx
from identification.src.learning import trainer
from identification.systems import snake_utils

from identification.hyperparams import settings

import sqlite3

from functools import partial

import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    # load model
    settings["sys_utils"] = snake_utils
    params = loader.load_from_pkl(path=settings["ckpt_dir_model"], verbose=1)
    train_state = trainer.create_train_state_PowNN(settings, 0, params=params)

    # load data
    # TODO: Add this to settings
    database = sqlite3.connect(
        "/home/gonz_jm/Documents/thesis_workspace/databases"
        "/database_points_20buff_command_standard"
    )
    table_name = "point_2"
    samples_num = 10
    offset_num = 150 + 450
    cursor = database.cursor()

    # build dynamics
    dyn_builder_compiled, energy_calcs, eom_prepared, split_tool = lx.build_dynamics(
        settings, params, train_state
    )
    calc_V_ana_vec = jax.vmap(snake_utils.calc_V_ana)

    # format and break up data
    (q, dq, ddq_target), state, data_formatted = trainer.handle_data(
        settings, cursor, table_name, samples_num, offset_num
    )

    # calculate forward and inverse dynamics
    dyn_terms = jax.vmap(dyn_builder_compiled)(state)
    ddq_pred = jax.vmap(lx.forward_dynamics)(dyn_terms)
    tau_pred, tau_target, tau_loss = jax.vmap(lx.inverse_dynamics)(
        ddq=ddq_target, terms=dyn_terms
    )

    # simulate the trajectory
    if settings["simulate"]:
        buffer_length = settings["buffer_length"]
        sys_utils = settings["sys_utils"]
        num_dof = settings["num_dof"]
        simulation = partial(
            lx.simulate,
            buffer_length=buffer_length,
            num_dof=num_dof,
            samples_num=samples_num,
            sys_utils=sys_utils,
            split_tool=split_tool,
            data_formatted=data_formatted,
            eom_prepared=eom_prepared,
        )
        q_sim, dq_sim = simulation(state=state, tau=tau_target)

    # calculate energies and powers
    T_lnn, V_lnn, powers = energy_calcs(batch=(state, ddq_target), dyn_terms=dyn_terms)
    pow_V, pow_T, pow_input, pow_f = powers

    # calculate losses
    loss_split = lambda sample: jnp.split(
        lx.loss_sample(split_tool, dyn_builder_compiled, sample), 3
    )

    loss_func = jax.vmap(loss_split)
    (L_acc_qdd, L_acc_tau, L_pot) = loss_func((state, ddq_target))

    # calculate energies and
    res_ana, res_lnn, (H_mec, H_loss) = lx.calculate_energies(
        calc_V_ana_vec, pow_input, pow_f, T_lnn, V_lnn, q
    )
    (T_ana, V_ana, H_ana, L_ana) = res_ana
    (T_lnn, V_lnn, H_lnn, L_lnn) = res_lnn

    # calibrate the energies
    res_cal, res_final = lx.calibrate_energies(
        settings, V_ana, V_lnn, T_ana, T_lnn, H_loss
    )
    (T_cal, V_cal, H_cal, L_cal) = res_cal
    (T_f, V_f, H_f, L_f) = res_final

    # plot results
    sns.set(style="darkgrid")
    trainer.plot_joint_positions(q, q_sim, settings)
    trainer.plot_joint_speeds(dq, dq_sim, settings)
    trainer.plot_friction_coeffs(tau_loss, dq)
    trainer.plot_accelerations(ddq_target, ddq_pred)
    trainer.plot_motor_torques(tau_target, tau_pred, tau_loss)
    trainer.plot_losses(L_acc_qdd, L_acc_tau, L_pot)
    trainer.plot_powers(pow_input)
    trainer.plot_hamiltonians(H_ana, H_mec, H_loss, H_cal, H_f)
    trainer.plot_lagrangians(L_ana, L_cal, L_f)
    trainer.plot_energies(V_ana, T_cal, V_cal, T_f, V_f)
