import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.patches import Circle

from moviepy.editor import ImageSequenceClip
from functools import partial
import proglog
import importlib
from PIL import Image

# import time
# from google.colab import files
import warnings

import jax

# from jax.experimental.ode import odeint

from src import lagranx as lx
from src import movie_maker as mk
from hyperparams import settings

import stable_baselines3.common.save_util as loader

if __name__ == "__main__":
    # Load
    print("Loading the trained model.")
    params = loader.load_from_pkl(path=settings["ckpt_dir"], verbose=1)
    train_state = lx.create_train_state(
        jax.random.PRNGKey(settings["seed"]), 0, params=params
    )

    # Define parameters
    N = 301
    t_f = 20
    x_0_sim = np.array([0, 0, 0, 0], dtype=np.float32)
    # x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 5, 0], dtype=np.float32)
    # x_0_sim = np.array([0, 3 * np.pi / 4, 5, -10], dtype=np.float32)
    L1, L2 = 0.5, 0.5
    di = 1

    # Simulate systems
    print("Simulating the analytic and learned systems.")
    t_sim = np.arange(N) / N * t_f
    x_sim = jax.device_get(lx.solve_analytical(x_0_sim, t_sim))
    lagrangian = lx.energy_func(params, train_state, output="lagrangian")
    x_model = jax.device_get(lx.solve_lagrangian(x_0_sim, lagrangian, t=t_sim))

    # Making videos
    print("Taking pictures of the analytical system at each time-step.")
    images_ana = mk.make_images(x_sim, N, di, L1, L2)
    PIL_images_ana = [Image.fromarray(im, mode="RGB") for im in images_ana]
    mk.save_movie(PIL_images_ana, "../../media/double_pend_ana.gif")
    print("True (analytical) dynamics gif of the double pendulum saved.")

    print("Taking pictures of the learned system at each time-step.")
    images_model = mk.make_images(x_model, N, di, L1, L2)
    PIL_images_model = [Image.fromarray(im, mode="RGB") for im in images_model]
    mk.save_movie(PIL_images_model, "../../media/double_pend_lnn.gif")
    print("LNN-predicted dynamics gif of the double pendulum saved.")

    # The movie sometimes takes a second before showing up in the file system.
