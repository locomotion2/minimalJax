import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3.common.save_util as loader
from systems import dpend_model_arne as model
# from src import lagranx as lx

from hyperparams import settings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import seaborn as sns

import warnings

import jax
import jax.numpy as jnp
import stable_baselines3.common.save_util as loader

from identification.src.dynamix import wrappings, optim
from identification.src.dynamix import motionx as mx
from identification.src.dynamix import energiex as ex
from identification.src.dynamix import simulate
from identification.src.training import trainer
from identification.src.training import plotting
from identification.systems import snake_utils
from identification.systems import dpendulum_utils

import identification.identification_utils as utils

from identification.hyperparams import settings

import sqlite3

from functools import partial

import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    # Load
    # params = loader.load_from_pkl(path=settings["ckpt_dir"], verbose=1)
    # train_state = lx.create_train_state(
    #     jax.random.PRNGKey(settings["seed"]), 0, params=params
    # )
    settings_system = settings["system_settings"]
    settings_model = settings["model_settings"]
    settings_system["sys_utils"] = dpendulum_utils
    model_dir = f"{settings_model['base_dir']}/{settings_system['system']}"
    path_energy = f"{model_dir}/{settings_model['ckpt_dir']}"
    params = loader.load_from_pkl(path=path_energy,
                                  verbose=1)
    train_state = trainer.create_train_state(settings_model["goal"],
                                             settings, 0, params=params)

    # setup system
    time_step = 0.01
    N_sim = int(1 / time_step) * 5
    # x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 5, 0], dtype=np.float32)
    # x_0_sim = np.array([0, 3 * np.pi / 4, 5, -10], dtype=np.float32)
    x_0_sim = jnp.array([0, 0, 0, 0], dtype=np.float32)
    t_sim = jnp.arange(N_sim, dtype=np.float32) * time_step  # time steps 0 to N

    # simulate system
    eom_compiled = jax.jit(model.f_analytical)
    x_sim = simulate.solve_analytical(eom_compiled,
                                      x_0_sim,
                                      t_sim)
    dx_sim = jax.vmap(model.f_analytical)(x_sim)
    x_sim = jax.vmap(model.wrap_angle)(x_sim)
    print(
        f"Dynamic system generated, ranges: "
        f"{jnp.amax(x_sim, axis=0)}, "
        f"{jnp.amin(x_sim, axis=0)}"
    )

    # analytic energies from trajectory
    T_ana, V_ana = jax.device_get(jax.vmap(model.analytic_energies)(x_sim))
    H_ana = T_ana + V_ana
    L_ana = T_ana - V_ana

    # build dynamics
    (
        dyn_builder_compiled,
        energy_calcs,
        eom_prepared,
        split_tool,
    ) = wrappings.build_test_funcs(settings, params, train_state)

    # test application function
    buffer_generator = dpendulum_utils.build_state_buffer_gen(N_sim, settings_model['buffer_length_max'])
    x_sim_buff = buffer_generator(x_sim)
    tau = jnp.zeros((N_sim, 2))
    x_sim_compound = jnp.append(x_sim_buff, tau, axis=1)
    # state = list(zip(x_sim_buff, dx_sim))
    energy_call_compiled = wrappings.build_energy_call(settings,
                                                       params,
                                                       train_state)
    T_lnn, V_lnn = jax.device_get(jax.vmap(energy_call_compiled)(x_sim_compound))

    # Learned energies from trajectory
    # T_lnn, V_lnn, T_rec, _, _ = jax.device_get(
    #     jax.vmap(
    #         lx.partial(lx.test_calculations, params=params, train_state=train_state)
    #     )(x_sim)
    # )
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    # Reconstructed energies from trajectory
    # V_rec = V_lnn
    # L_rec = T_rec - V_rec
    # H_rec = T_rec + V_rec

    # Calibrated energies
    # mean_ana = jnp.mean(T_ana)
    # mean_lnn = jnp.mean(T_rec)
    # max_ana = jnp.max(T_ana - mean_ana)
    # min_ana = jnp.min(T_ana - mean_ana)
    # max_lnn = jnp.max(T_rec - mean_lnn)
    # min_lnn = jnp.min(T_rec - mean_lnn)
    # height_ana = (max_ana - min_ana) / 2
    # height_lnn = (max_lnn - min_lnn) / 2
    # alpha = height_ana / height_lnn
    # beta = - mean_lnn * height_ana / height_lnn + mean_ana
    # T_cal = T_rec * alpha + beta#
    [alpha, beta], T_cal = utils.calibrate(T_ana, T_lnn)
    print(f"Factors: {alpha}, {beta}.")

    # mean_ana = jnp.mean(V_ana)
    # mean_lnn = jnp.mean(V_rec)
    # max_ana = jnp.max(V_ana - mean_ana)
    # min_ana = jnp.min(V_ana - mean_ana)
    # max_lnn = jnp.max(V_rec - mean_lnn)
    # min_lnn = jnp.min(V_rec - mean_lnn)
    # alpha = height_ana / height_lnn
    # beta = - mean_lnn * height_ana / height_lnn + mean_ana
    # V_cal = V_rec * alpha + beta
    [alpha, beta], V_cal = utils.calibrate(V_ana, V_lnn)
    print(f"Factors: {alpha}, {beta}.")

    L_cal = T_cal - V_cal
    H_cal = T_cal + V_cal

    # factors previously calibrated
    kin_factors = np.array([1.785508155822754, 0.009507834911346436])
    pot_factors = np.array([1.785508155822754, -23.28985595703125])
    T_f = V_lnn * kin_factors[0] + kin_factors[1]
    V_f = V_lnn * pot_factors[0] + pot_factors[1]
    H_f = T_f + V_f
    L_f = T_f - V_f

    # Losses
    # L_total, L_acc, L_mec = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    # for step in range(N_sim):
    #     data = (x_sim[step, :], dx_sim[step, :])
    #     L_total[step], L_acc[step], L_mec[step] = lx.loss_sample(
    #         data, params=params, train_state=train_state
    #     )

    # Plotting
    sns.set(style="darkgrid")

    # Losses
    # plt.figure(figsize=(8, 4.5), dpi=120)
    # plt.plot(t_sim, L_total, linewidth=2, label="Total Loss")
    # plt.plot(t_sim, L_acc, "--", label="Accuracy")
    # plt.plot(t_sim, L_mec, "--", label="Kinetic")
    # plt.ylim(-0.1, 0.5)
    # plt.xlim(0, 5)
    # plt.title("Objective function losses for Energy-based Model Identification")
    # plt.ylabel("Loss (arb. unit)")
    # plt.xlabel("Time (s)")
    # plt.legend([r"$L_{tot}$", r"$L_{acc}$", r"$L_{mec}$"], loc="best")
    # # plt.savefig('media/Model identification/Loss.png')
    # plt.show()

    # Hamiltonians
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(t_sim, H_ana, linewidth=3, label="H. Analytic")
    plt.plot(t_sim, H_lnn, label='H. Lnn')
    # plt.plot(t_sim, H_rec, label="H. Recon")
    plt.plot(t_sim, H_cal, label='H. Calibrated')
    # plt.plot(t_sim, H_f, "--", linewidth=3, label="H. Final")
    plt.title("Hamiltonians from the Gravitational Double Pendulum Simulation")
    # plt.ylim(0, 2.0)
    plt.xlim(0, 5)
    plt.ylabel("Energy (J)")
    plt.xlabel("Time (s)")
    plt.legend([r"$H_{ana}$", r"$H_{nn}$", r"$H_{calib.}$"], loc="best")
    # plt.savefig('media/Model identification/Hamiltonian.png')
    plt.show()

    # Lagrangians
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(t_sim, L_ana, linewidth=3, label="L. Analytic")
    plt.plot(t_sim, L_lnn, label='L. Lnn')
    # plt.plot(t_sim, L_rec, label="L. Reconstructed")
    plt.plot(t_sim, L_cal, label='L. Calibrated')
    # plt.plot(t_sim, L_f, "--", linewidth=2, label="L. Final")
    plt.title("Lagrangians from the Gravitational Double Pendulum Simulation")
    # plt.ylim(-1.5, 2.0)
    plt.xlim(0, 5)
    plt.ylabel("Energy (J)")
    plt.xlabel("Time (s)")
    plt.legend([r"$l_{ana}$", r"$L_{nn}$", r"$L_{calib.}$"], loc="best")
    # plt.savefig('media/Model identification/Lagrangian.png')
    plt.show()

    # Energies
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(t_sim, T_ana, linewidth=3, label="Kin. Analytic")
    plt.plot(t_sim, V_ana, linewidth=3, label="Pot. Analytic")
    plt.plot(t_sim, T_lnn, label='Kin. Lnn.')
    plt.plot(t_sim, V_lnn, label='Pot. Lnn.')
    # plt.plot(t_sim, T_rec, label="Kin. Reconstructed")
    # plt.plot(t_sim, V_rec, label="Pot. Reconstructed")
    plt.plot(t_sim, T_cal, label='Kin. Calibrated')
    plt.plot(t_sim, V_cal, label='Pot. Calibrated')
    # plt.plot(t_sim, T_f, "--", linewidth=2, label="Kin. Final")
    # plt.plot(t_sim, V_f, "--", linewidth=2, label="Pot. Final")
    plt.title("Energies from the Gravitational Double Pendulum Simulation")
    # plt.ylim(-0.25, 2.0)
    plt.xlim(0, 5)
    plt.ylabel("Energy Level (J)")
    plt.xlabel("Time (s)")
    plt.legend()
    # plt.legend([r"$T_{ana}$", r"$V_{ana}$", r"$T_{nn}$", r"$V_{nn}$"], loc="best")
    # plt.savefig('media/Model identification/Energies.png')
    plt.show()
