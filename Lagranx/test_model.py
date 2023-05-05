from src import lagranx as lx
from src import dpend_model_arne as model
from hyperparams import settings

import jax
import jax.numpy as jnp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3.common.save_util as loader

if __name__ == "__main__":
    # Load
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(jax.random.PRNGKey(settings['seed']), 0, params=params)

    # Test system
    time_step = 0.001
    N_sim = 1000 * 5
    # x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 5, 0], dtype=np.float32)
    x_0_sim = np.array([0, 3 * np.pi / 4, 5, -10], dtype=np.float32)
    # x_0_sim = np.array([0, 0, 0, 0], dtype=np.float32)

    # Simulate system
    t_sim = np.arange(N_sim, dtype=np.float32) * time_step  # time steps 0 to N
    x_sim = lx.solve_analytical(x_0_sim, t_sim)
    xt_sim = jax.vmap(model.f_analytical)(x_sim)  # time derivatives of each state
    x_sim = jax.vmap(model.normalize)(x_sim)
    print(f"Generation successful! Ranges: {jnp.amax(x_sim, axis=0)}, {jnp.amin(x_sim, axis=0)}")

    # Analytic energies from trajectory
    T_ana, V_ana = jax.device_get(jax.vmap(model.analytic_energies)(x_sim))
    H_ana = T_ana + V_ana
    L_ana = T_ana - V_ana

    # Learned energies from trajectory
    T_lnn, V_lnn, T_rec, _, _ = jax.device_get(jax.vmap(lx.partial(lx.learned_energies, params=params, train_state=train_state))(x_sim))
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    # Reconstructed energies from trajectory
    V_rec = V_lnn
    L_rec = T_rec - V_rec
    H_rec = T_rec + V_rec

    mean_ana = jnp.mean(T_ana)
    mean_lnn = jnp.mean(T_rec)
    max_ana = jnp.max(T_ana - mean_ana)
    min_ana = jnp.min(T_ana - mean_ana)
    max_lnn = jnp.max(T_rec - mean_lnn)
    min_lnn = jnp.min(T_rec - mean_lnn)
    height_ana = (max_ana - min_ana) / 2
    height_lnn = (max_lnn - min_lnn) / 2
    alpha = height_ana / height_lnn
    beta = - mean_lnn * height_ana / height_lnn + mean_ana
    T_cal = T_rec * alpha + beta
    print(f'Factors: {alpha}, {beta}.')

    mean_ana = jnp.mean(V_ana)
    mean_lnn = jnp.mean(V_rec)
    max_ana = jnp.max(V_ana - mean_ana)
    min_ana = jnp.min(V_ana - mean_ana)
    max_lnn = jnp.max(V_rec - mean_lnn)
    min_lnn = jnp.min(V_rec - mean_lnn)
    alpha = height_ana / height_lnn
    beta = - mean_lnn * height_ana / height_lnn + mean_ana
    V_cal = V_rec * alpha + beta
    print(f'Factors: {alpha}, {beta}.')

    L_cal = T_cal - V_cal
    H_cal = T_cal + V_cal

    # Energies saved
    kin_factors = np.array([1.865427851676941, 0.038243770599365234])
    pot_factors = np.array([1.865427851676941, -18.905736923217773])

    T_f = T_rec * kin_factors[0] + kin_factors[1]
    V_f = V_rec * pot_factors[0] + pot_factors[1]
    H_f = T_f + V_f
    L_f = T_f - V_f

    # Losses
    L_total, L_acc, L_con, L_kin, L_pot, L_pos =\
        np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    for step in range(N_sim):
        data = (x_sim[step, :], xt_sim[step, :])
        L_total[step], L_acc[step], L_con[step], L_kin[step], L_pot[step] = \
            lx.loss_sample(data, params=params, train_state=train_state)

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, L_total, label='Total Loss')
    plt.plot(t_sim, L_acc, label='Accuracy')
    # plt.plot(t_sim, L_con, label='Hamiltonian')
    plt.plot(t_sim, L_kin, label='Kinetic')
    plt.plot(t_sim, L_pot, label='Potential')
    plt.ylim(-0.1, 1)
    plt.title('Errors')
    plt.ylabel('Loss')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, H_ana, label='H. Analytic')
    # plt.plot(t_sim, H_lnn, label='H. Lnn')
    # plt.plot(t_sim, H_rec, label='H. Recon')
    # plt.plot(t_sim, H_cal, label='H. Calibrated')
    plt.plot(t_sim, H_f, label='H. Final')
    plt.title('Hamiltonians')
    plt.ylim(0, 2.0)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, L_ana, label='L. Analytic')
    # plt.plot(t_sim, L_lnn, label='L. Lnn')
    # plt.plot(t_sim, L_rec, label='L. Reconstructed')
    # plt.plot(t_sim, L_cal, label='L. Calibrated')
    plt.plot(t_sim, L_f, label='L. Final')
    plt.title('Lagrangians')
    plt.ylim(-1, 2.5)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, T_ana, label='Kin. Analytic')
    plt.plot(t_sim, V_ana, label='Pot. Analytic')
    # plt.plot(t_sim, T_lnn, label='Kin. Lnn.')
    # plt.plot(t_sim, V_lnn, label='Pot. Lnn.')
    # plt.plot(t_sim, T_rec, label='Kin. Reconstructed')
    # plt.plot(t_sim, V_rec, label='Pot. Reconstructed')
    # plt.plot(t_sim, T_cal, label='Kin. Calibrated')
    # plt.plot(t_sim, V_cal, label='Pot. Calibrated')
    plt.plot(t_sim, T_f, label='Kin. Final')
    plt.plot(t_sim, V_f, label='Pot. Final')
    plt.title('Energies')
    plt.ylim(-1.5, 1.5)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()
