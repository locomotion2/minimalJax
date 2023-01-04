from sim.CONSTANTS import *

import gym
from gym.spaces import Box
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

from sim.environments import BaseEnvironment


class BaseGymEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, reward_func: Callable = None, action_scale=None, render: bool = False):
        super().__init__()

        if action_scale is None:
            self.action_scale = [10, 2]
        else:
            self.action_scale = action_scale

        if reward_func is None:
            self.reward_func = self.default_func
        else:
            self.reward_func = reward_func

        self.action_space = Box(low=-1, high=1, shape=(OUTPUT_SIZE,))
        self.observation_space = Box(low=-1, high=1, shape=(INPUT_SIZE,))

        self.visualize = render
        if self.visualize:
            # display.clear_output(wait=True)
            plt.figure(2)
            plt.clf()
            plt.ion()
            plt.show()

        self.sim = BaseEnvironment(delta_t=0.01, t_final=2, render=render)
        self.E_d = 0
        self.r_traj = np.asarray([0])
        self.E_t_traj = np.asarray([0])
        self.rng = np.random.default_rng()

    def default_func(self, state: dict):
        E_d = state['Energy_des']
        E_k, E_p = state['Energies']
        E_t = E_k + E_p
        r_E = float(np.exp(-((E_t - E_d) / 0.3) ** 2))

        tau = state['Torque']
        r_tau = float(np.exp(-(tau / 0.3) ** 2))

        r_t = 0.8 * r_E + 0.2 * r_tau

        if self.visualize:
            self.plot_reward(r_t, E_t, E_d)

        return r_t

    def new_target_energy(self):
        # self.E_d = self.rng.uniform(0, MAX_ENERGY / 2)
        self.E_d = 0.2

    def gather_data(self):
        p = self.sim.get_positions()
        E = self.sim.get_energies()
        tau = self.sim.controller.get_force_traj()[-1]
        state = {'Positions': p, 'Energy_des': self.E_d, 'Energies': E, 'Torque': tau}
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

    def plot_reward(self, r, E_t, E_d):
        self.r_traj = np.append(self.r_traj, r)
        self.E_t_traj = np.append(self.E_t_traj, E_t)
        t_traj_model = self.sim.model.get_temporal_traj()
        E_d_traj = np.ones(np.shape(t_traj_model)) * E_d

        plt.figure(2)
        ax = plt.subplot(2, 1, 1)
        ax.set_ylim([0, 1])
        plt.plot(t_traj_model, self.r_traj, 'y--', linewidth=1)
        plt.ylabel(r'Reward')
        plt.xlabel('Time (s)')
        # plt.legend(['Pend. traj.'], loc='best')

        ax = plt.subplot(2, 1, 2)
        # ax.set_ylim([0, 1])
        plt.plot(t_traj_model, self.E_t_traj, 'b--', linewidth=1)
        plt.plot(t_traj_model, E_d_traj, 'g--', linewidth=1)
        plt.ylabel(r'Energy (J)')
        plt.xlabel('Time (s)')
        plt.legend([r'$E_t$', r'$E_d$'], loc='best')

        plt.draw()
        plt.pause(0.0001)

    def render(self, mode="human"):
        if mode == 'rgb_array':
            return None  # return RGB frame suitable for video
        elif mode == 'human':
            self.sim.plot()
        else:
            super(BaseGymEnvironment, self).render(mode=mode)  # just raise an exception
