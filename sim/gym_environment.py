from sim.CONSTANTS import *

import gym
from gym.spaces import Box
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

from sim.environments import BaseEnvironment
from sim.curricula import UniformGrowthCurriculum, BaseCurriculum


class BaseGymEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, reward_func: Callable = None, curriculum: BaseCurriculum = None, action_scale=None,
                 render: bool = False, energy_command: float = None):
        super().__init__()

        # Handle inputs
        if action_scale is None:
            self.action_scale = ACTION_SCALE
        else:
            self.action_scale = action_scale

        if reward_func is None:
            self.reward_func = self.default_func
        else:
            self.reward_func = reward_func

        if curriculum is None:
            self.curriculum = UniformGrowthCurriculum()
        else:
            self.curriculum = curriculum

        if energy_command is None:
            self.inference = False
            self.E_d = 0
        else:
            self.inference = True
            self.E_d = energy_command

        # Build environment
        self.action_space = Box(low=-1, high=1, shape=(OUTPUT_SIZE,))
        self.observation_space = Box(low=-1, high=1, shape=(INPUT_SIZE,))
        self.sim = BaseEnvironment(delta_t=0.01, t_final=2, render=render)

        # Tracking / help variables
        self.r_epi = 0
        self.cur_step = 0
        self.r_traj = np.asarray([0])
        self.E_t_traj = np.asarray([0])

        # Rendering
        self.visualize = render
        if self.visualize:
            # display.clear_output(wait=True)
            plt.figure('Reward', figsize=FIG_SIZE)
            plt.clf()
            plt.ion()
            plt.show()

    def default_func(self, state: dict):
        # Energy reward
        E_d = state['Energy_des']
        E_k, E_p = state['Energies']
        E_t = E_k + E_p
        r_E = float(np.exp(-((E_t - E_d) / 0.1) ** 2))

        # Force punishment
        tau = state['Torque']
        r_tau = float(np.exp(-(tau / 0.1) ** 2))

        # Total reward
        r_step = 0.8 * r_E + 0.2 * r_tau

        # Tracking data
        self.r_traj = np.append(self.r_traj, r_step)
        self.E_t_traj = np.append(self.E_t_traj, E_t)

        # Plotting
        if self.visualize and self.cur_step % VIZ_RATE == 0:
            self.sim.plot()
            self.plot_reward()

        # Tracking the episode
        self.r_epi += r_step

        return r_step

    def new_target_energy(self):
        if not self.inference:
            if self.curriculum.check_criterion(self.r_epi, self.E_d):
                self.curriculum.grow()
            self.E_d = self.curriculum.sample()

    def gather_data(self):
        p = self.sim.get_positions()

        E = self.sim.get_energies()
        tau = self.sim.controller.get_force_traj()[-1]
        state = {'Positions': p, 'Energy_des': self.E_d, 'Energies': E, 'Torque': tau}
        obs = np.concatenate([p, [self.E_d / MAX_ENERGY]])

        return state, obs

    def step(self, action):
        # Format and take action
        # action[0] += 1
        # print(action)
        action = np.multiply(self.action_scale, action)  # TODO: Format in its own function
        self.sim.step(action)

        # Extract data for learning
        state, obs = self.gather_data()
        reward = self.reward_func(state)
        done = self.sim.is_done()
        info = {}  # TODO: Add some debugging info

        # March forwards
        self.cur_step += 1

        return obs, reward, done, info

    def reset(self):
        self.sim.restart()
        self.new_target_energy()
        self.r_epi = 0
        self.cur_step = 0
        self.r_traj = np.asarray([0])
        self.E_t_traj = np.asarray([0])
        _, obs = self.gather_data()

        return obs

    def plot_reward(self):
        t_traj = self.sim.model.get_temporal_traj()
        E_d_traj = np.ones(np.shape(t_traj)) * self.E_d

        plt.figure('Reward')

        # Reward vs time
        ax = plt.subplot(2, 1, 1)
        ax.clear()
        ax.set_ylim([0, 1])
        plt.plot(t_traj, self.r_traj, 'y--', linewidth=1)
        plt.ylabel(r'Reward')
        plt.xlabel('Time (s)')
        # plt.legend(['Pend. traj.'], loc='best')

        # Energies vs time
        ax = plt.subplot(2, 1, 2)
        ax.clear()
        # ax.set_ylim([0, 1])
        plt.plot(t_traj, self.E_t_traj, 'b--', linewidth=1)
        plt.plot(t_traj, E_d_traj, 'g--', linewidth=1)
        plt.ylabel(r'Energy (J)')
        plt.xlabel('Time (s)')
        plt.legend([r'$E_t$', r'$E_d$'], loc='best')

        plt.draw()
        plt.pause(0.00001)

    def render(self, mode="human"):
        if mode == 'rgb_array':
            return None  # return RGB frame suitable for video
        elif mode == 'human':
            pass
            # self.sim.plot()
            # self.plot_reward()
        else:
            super(BaseGymEnvironment, self).render(mode=mode)  # just raise an exception
