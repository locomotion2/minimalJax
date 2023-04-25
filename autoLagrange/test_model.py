import lagranx as lx
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import stable_baselines3.common.save_util as loader

if __name__ == "__main__":
    # Define all settings
    settings = {'batch_size': 100,
                'test_every': 1,
                'num_batches': 1000,
                'num_epochs': 150,
                'time_step': 0.01,
                'data_size': 1500,
                'starting_point': np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32),
                'data_dir': 'tmp/data',
                'reload': False,
                'ckpt_dir': 'tmp/flax-checkpointing',
                'seed': 0
                }

    # Load
    params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = lx.create_train_state(jax.random.PRNGKey(settings['seed']), 0, params=params)

    # Test system
    time_step = 0.001
    N_sim = 1000 * 5
    x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)
    # x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)

    # Simulate system
    t_sim = np.arange(N_sim, dtype=np.float32) * time_step  # time steps 0 to N
    x_sim = lx.solve_analytical(x_0_sim, t_sim)
    xt_sim = jax.vmap(lx.f_analytical)(x_sim)  # time derivatives of each state
    x_sim = jax.vmap(lx.normalize_dp)(x_sim)

    # Analytic energies from trajectory
    T_ana, V_ana = jax.device_get(jax.vmap(lx.analytic_energies)(x_sim))
    H_ana = T_ana + V_ana
    L_ana = T_ana - V_ana

    # Learned energies from trajectory
    T_lnn, V_lnn, _ = jax.device_get(jax.vmap(lx.partial(lx.learned_energies,
                                                         params=params, train_state=train_state))(x_sim))
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    # Calibrated energies from trajectory
    T_rec = jax.device_get(jax.vmap(partial(lx.kin_energy_lagrangian, lagrangian=lx.learned_lagrangian(params, train_state)))(x_sim))
    H_0 = jnp.mean(H_lnn)
    V_rec = H_0 - T_rec
    L_rec = T_rec - V_rec
    H_rec = T_rec + V_rec

    mean_ana = jnp.mean(T_ana)
    mean_lnn = jnp.mean(T_rec)
    max_ana = jnp.max(T_ana - mean_ana)
    max_lnn = jnp.max(T_rec - mean_lnn)
    T_cal = ((T_rec - mean_lnn) / max_lnn) * max_ana + mean_ana

    mean_ana = jnp.mean(V_ana)
    mean_lnn = jnp.mean(V_rec)
    max_ana = jnp.max(V_ana - mean_ana)
    max_lnn = jnp.max(V_rec - mean_lnn)
    V_cal = ((V_rec - mean_lnn) / max_lnn) * max_ana + mean_ana

    L_cal = T_cal - V_cal
    H_cal = T_cal + V_cal

    # Losses
    L_total, L_acc, L_con, L_kin, L_pot, L_pos = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    for step in range(N_sim):
        data = (x_sim[step, :], xt_sim[step, :])
        L_total[step], L_acc[step], L_con[step], L_kin[step], L_pot[step], L_pos[step] =\
            lx.loss_sample(data, params=params, train_state=train_state)

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, L_total, label='Total Loss')
    plt.plot(t_sim, L_acc, label='Accuracy')
    plt.plot(t_sim, L_con, label='Hamiltonian')
    plt.plot(t_sim, L_kin, label='Kinetic')
    plt.plot(t_sim, L_pot, label='Potential')
    plt.ylim(-0.1, 200)
    plt.title('Errors')
    plt.ylabel('Loss')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, H_ana, label='H. Analytic')
    plt.plot(t_sim, H_lnn, label='H. Lnn')
    plt.plot(t_sim, H_rec, label='H. Recon')
    plt.plot(t_sim, H_cal, label='H. Calibrated')
    plt.title('Hamiltonians')
    plt.ylim(0, 0.1)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, L_ana, label='L. Analytic')
    plt.plot(t_sim, L_lnn, label='L. Lnn')
    plt.plot(t_sim, L_rec, label='L. Reconstructed')
    plt.plot(t_sim, L_cal, label='L. Calibrated')
    plt.title('Lagrangians')
    plt.ylim(-0.5, 2)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, T_ana, label='Kin. Analytic')
    plt.plot(t_sim, V_ana, label='Pot. Analytic')
    plt.plot(t_sim, T_lnn, label='Kin. Lnn.')
    plt.plot(t_sim, V_lnn, label='Pot. Lnn.')
    plt.plot(t_sim, T_rec, label='Kin. Reconstructed')
    plt.plot(t_sim, V_rec, label='Pot. Reconstructed')
    plt.plot(t_sim, T_cal, label='Kin. Calibrated')
    plt.plot(t_sim, V_cal, label='Pot. Calibrated')
    plt.title('Energies')
    plt.ylim(-1, 1)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()