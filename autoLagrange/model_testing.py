import autoLagrange as al
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import stable_baselines3.common.save_util as loader

if __name__ == "__main__":
    # Load
    seed = 0
    ckpt_dir = 'tmp/flax-checkpointing'
    restored_state, _ = al.create_train_state(jax.random.PRNGKey(seed), 0)
    params = loader.load_from_pkl(path=ckpt_dir, verbose=1)
    train_state = restored_state

    # Test system
    time_step = 0.001
    N_sim = 1000 * 5
    x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)

    # Simulate system
    t_sim = np.arange(N_sim, dtype=np.float32) * time_step  # time steps 0 to N
    x_sim = al.solve_analytical(x_0_sim, t_sim)
    x_sim = jax.vmap(al.normalize_dp)(x_sim)

    # Analytic energies from trajectory
    T_ana, V_ana = jax.device_get(jax.vmap(al.analytic_energies)(x_sim))
    H_ana = T_ana + V_ana
    L_ana = T_ana - V_ana

    # Learned energies from trajectory
    T_lnn, V_lnn = jax.device_get(jax.vmap(al.partial(al.learned_energies, params))(x_sim))
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    # Reconstructed energies from trajectory

    # Calibration
    # x_origin = np.array([0, 0, 0, 0], dtype=np.float32)
    # _, V_origin = jax.device_get(al.analytic_energies(x_origin))
    # V_lnn_origin, _ = jax.device_get(partial(al.learned_energies, params)(x_origin))
    # x_bottom = np.array([np.pi / 2, np.pi / 2, 0, 0], dtype=np.float32)
    # _, V_bottom = jax.device_get(al.analytic_energies(x_origin))
    # V_lnn_bottom, _ = jax.device_get(partial(al.learned_energies, params)(x_bottom))
    # alpha = (V_origin - V_bottom) / (V_lnn_origin - V_lnn_bottom)
    # beta = V_origin - alpha * V_lnn_origin
    # H_rec_0 = jnp.mean(alpha * (T_lnn + V_lnn) + beta)
    #
    # T_rec = alpha * jax.device_get(jax.vmap(partial(al.recon_kin, al.learned_lagrangian(params)))(x_sim)) + beta
    # V_rec = H_rec_0 - T_rec
    # H_rec = V_rec + T_rec
    # L_rec = T_rec - V_rec

    # Calibrated energies from trajectory
    T_rec = jax.device_get(jax.vmap(partial(al.recon_kin, al.learned_lagrangian(params)))(x_sim))
    H_0 = jnp.mean(H_lnn)
    V_rec = H_0 - T_rec
    L_rec = T_rec - V_rec
    H_rec = T_rec + V_rec

    mean_ana = jnp.mean(T_ana)
    mean_lnn = jnp.mean(T_rec)
    max_ana = jnp.max(T_ana - mean_ana)
    max_lnn = jnp.max(T_rec - mean_lnn)
    T_cal = ((T_rec - mean_lnn) / max_lnn) * max_ana + mean_ana
    # T_cal = T_rec * (max_ana / mean_lnn) - (mean_lnn * max_ana / max_lnn - mean_ana)

    mean_ana = jnp.mean(V_ana)
    mean_lnn = jnp.mean(V_rec)
    max_ana = jnp.max(V_ana - mean_ana)
    max_lnn = jnp.max(V_rec - mean_lnn)
    V_cal = ((V_rec - mean_lnn) / max_lnn) * max_ana + mean_ana
    # V_cal = V_rec * (max_ana / mean_lnn) - (mean_lnn * max_ana / max_lnn - mean_ana)

    L_cal = T_cal - V_cal
    H_cal = T_cal + V_cal

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, H_ana, label='H. Analytic')
    # plt.plot(t_sim, H_lnn, label='H. Lnn')
    # plt.plot(t_sim, H_rec, label='H. Recon')
    plt.plot(t_sim, H_cal, label='H. Calibrated')
    plt.title('Hamiltonians')
    plt.ylim(0, 0.1)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, L_ana, label='L. Analytic')
    # plt.plot(t_sim, L_lnn, label='L. Lnn')
    # plt.plot(t_sim, L_rec, label='L. Reconstructed')
    plt.plot(t_sim, L_cal, label='L. Calibrated')
    plt.title('Lagrangians')
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
    plt.plot(t_sim, T_cal, label='Kin. Calibrated')
    plt.plot(t_sim, V_cal, label='Pot. Calibrated')
    plt.title('Energies')
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()