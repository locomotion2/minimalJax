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
