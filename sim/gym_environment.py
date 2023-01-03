from CONSTANTS import *

import gym
from gym.spaces import Box
import numpy as np
from typing import Callable
from IPython import display
import matplotlib.pyplot as plt

from environments import BaseEnvironment


class BaseGymEnvironment(gym.Env):
    def __init__(self, sim_handle: BaseEnvironment, reward_func: Callable = None, action_scale=None):
        super().__init__()

        if action_scale is None:
            self.action_scale = [1, 1]
        else:
            self.action_scale = action_scale
        self.action_space = Box(low=0, high=1, shape=(OUTPUT_SIZE,))
        self.observation_space = Box(low=-1, high=1, shape=(INPUT_SIZE,))

        self.sim = sim_handle
        self.reward_func = reward_func
        self.E_d = 0

        self.rng = np.random.default_rng()

    def new_target_energy(self):
        self.E_d = self.rng.uniform(0, MAX_ENERGY / 2)

    def gather_data(self):
        p = self.sim.get_positions()
        E = self.sim.get_energies()
        state = {'positions': p, 'Energy_des': self.E_d, 'Energies': E}
        obs = np.concatenate(p, self.E_d / MAX_ENERGY)

        return state, obs

    def step(self, action):
        # Format and take action
        action = np.multiply(self.action_scale, action)
        self.sim.step(action)

        # Extract data for learning
        state, obs = self.gather_data()
        reward = self.reward_func(state)
        done = self.sim.is_done()
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.sim.restart()
        self.new_target_energy()
        _, obs = self.gather_data()

        return obs

    def render(self, mode="human"):
        pass
        # display.clear_output(wait=True)
        # plt.clf()
        # # plt.figure(figsize=(10,7))
        #
        # plt.subplot(2, 1, 1)
        # plt.plot(t[0:i + 1], rr[0:i + 1], 'b--', linewidth=3)
        # plt.ylabel(r'$RR$')
        # plt.legend(['Reflux ratio'], loc='best')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(t[0:i + 1], sp[0:i + 1], 'k.-', linewidth=1)
        # plt.plot(t[0:i + 1], xd[0:i + 1], 'r-', linewidth=3)
        # plt.ylabel(r'$x_d\;(mol/L)$')
        # plt.legend(['Starting composition', 'Distillate composition'], loc='best')
        # plt.xlabel('Time (hr)')
        #
        # plt.draw()
        # plt.show()