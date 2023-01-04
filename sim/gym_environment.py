from sim.CONSTANTS import *

import gym
from gym.spaces import Box
import numpy as np
from typing import Callable

from sim.environments import BaseEnvironment


class BaseGymEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, reward_func: Callable = None, action_scale=None):
        super().__init__()

        if action_scale is None:
            self.action_scale = [1, 1]
        else:
            self.action_scale = action_scale

        if reward_func is None:
            self.reward_func = self.default_func
        else:
            self.reward_func = reward_func

        self.action_space = Box(low=0, high=1, shape=(OUTPUT_SIZE,))
        self.observation_space = Box(low=-1, high=1, shape=(INPUT_SIZE,))

        self.sim = BaseEnvironment(delta_t=0.01, t_final=2)
        self.E_d = 0
        self.rng = np.random.default_rng()

    def default_func(self, state: dict):
        E_d = state['Energy_des']
        E_k, E_p = state['Energies']
        E_t = E_k + E_p

        r_E = float(np.exp(-((E_t - E_d) / 0.3) ** 2))

        return r_E * 0.1

    def new_target_energy(self):
        self.E_d = self.rng.uniform(0, MAX_ENERGY / 2)

    def gather_data(self):
        p = self.sim.get_positions()
        E = self.sim.get_energies()
        state = {'positions': p, 'Energy_des': self.E_d, 'Energies': E}
        obs = np.concatenate([p, [self.E_d / MAX_ENERGY]])

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
        if mode == 'rgb_array':
            return None  # return RGB frame suitable for video
        elif mode == 'human':
            self.sim.render()
        else:
            super(BaseGymEnvironment, self).render(mode=mode)  # just raise an exception
