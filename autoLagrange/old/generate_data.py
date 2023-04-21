import autoLagrange as al
import numpy as np
import jax.numpy as jnp
import jax, optax
import matplotlib.pyplot as plt
import stable_baselines3.common.save_util as loader
from copy import deepcopy as copy

if __name__ == "__main__":
    print('Step 2: Generating the dataset generator')
    seed = 0  # needless to say these should be in a config or defined like flags
    total_epochs = 100
    batch_size = 1500
    minibatch_per_batch = 1

    random_key = jax.random.PRNGKey(0)
    time_step = 0.01
    x_0 = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)
    noise = np.random.RandomState(0).randn(x_0.size)
    x_0_test = x_0 + 1e-3 * noise

    data_generator = al.train_test_data_generator_toy(x_0, x_0_test,
                                                      batch_size, minibatch_per_batch, total_epochs,
                                                      time_step=time_step)

    data = data_generator(0)
    data_dir = 'tmp/data'
    loader.save_to_pkl(path=data_dir, obj=data, verbose=1)