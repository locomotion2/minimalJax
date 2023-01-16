from sim.CONSTANTS import *

import gym
from gym.spaces import Box
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib
# import pyautogui as pg

from sim.environments import BaseEnvironment
from sim.curricula import UniformGrowthCurriculum, BaseCurriculum


def gaus(value: float, width: float = 0.3):
    return float(np.exp(-(value / width) ** 2))


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class BaseGymEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, reward_func: Callable = None, curriculum: BaseCurriculum = None, action_scale=None,
                 render: bool = False, energy_command: float = None, starting_range: list = None,
                 mode: str = 'equilibrium', solve: bool = False):
        super().__init__()

        # Handle inputs
        self.solve = solve

        if mode in {'speed', 'position', 'equilibrium'}:
            self.mode = mode
        else:
            raise Exception('Selected initial condition mode is unknown.')

        if action_scale is None:
            self.action_scale = ACTION_SCALE
        else:
            self.action_scale = action_scale

        if starting_range is None:  # This implies that a curriculum is not wanted
            var_range = [0, 1]
        else:
            var_range = starting_range

        if reward_func is None:
            self.reward_func = self.default_func
        else:
            self.reward_func = reward_func

        if curriculum is None:
            self.curriculum = UniformGrowthCurriculum(starting_range=var_range)
        else:
            self.curriculum = curriculum

        if energy_command is None:
            self.inference = False
            self.E_d = 0
        else:
            self.inference = True
            self.E_d = energy_command

        # Rendering
        self.visualize = render
        if self.visualize:
            # display.clear_output(wait=True)
            # matplotlib.use("TkAgg")
            # width, height = pg.size()
            fig_reward = plt.figure('Reward', figsize=FIG_SIZE)
            move_figure(fig_reward, 0, 0)
            plt.clf()
            # plt.ion()
            # plt.show()

            fig_system = plt.figure('System', figsize=FIG_SIZE)
            move_figure(fig_system, 1920/2, 0)
            plt.clf()
            # plt.ion()
            # plt.show()

        # Build environment
        self.action_space = Box(low=-1, high=1, shape=(OUTPUT_SIZE,))
        self.observation_space = Box(low=-1, high=1, shape=(INPUT_SIZE,))
        self.sim = BaseEnvironment(delta_t_learning=ACTUAL_TIMESTEP, delta_t_system=MIN_TIMESTEP,
                                   t_final=ACTUAL_FINAL_TIME, mode=self.mode, solve=self.solve)

        # Tracking / help variables
        self.r_epi = 0
        self.cur_step = 0
        self.r_traj = np.asarray([self.r_epi])
        self.E_t_traj = np.asarray([0])

    def default_func(self, state: dict):
        # Force punishment
        tau = state['Torque']
        cost_torque = gaus(tau, 0.2)

        # Energy rewards
        E_d = state['Energy_des']
        E_k, E_p = state['Energies']
        E_t = E_k + E_p
        cost_E_t = gaus(E_t - E_d, 0.25)
        cost_E_k = gaus(E_k - E_d, 0.3) * gaus(E_p, 0.3)
        cost_E_p = gaus(E_p - E_d, 0.1) * gaus(E_k, 0.3)

        # Total reward
        costs = np.asarray([cost_E_t, cost_E_k, cost_E_p, cost_torque])
        weights = np.asarray([0.5, 0.0, 0.0, 0.5])
        cost_step = np.dot(costs, weights)

        # Tracking data
        self.r_traj = np.append(self.r_traj, cost_step)
        self.E_t_traj = np.append(self.E_t_traj, E_t)

        # Plotting
        if self.visualize and self.cur_step % VIZ_RATE == 0:
            try:
                self.sim.plot()
                self.plot_reward()
            except KeyboardInterrupt:
                raise KeyboardInterrupt

        # Tracking the episode
        self.r_epi += cost_step

        return cost_step

    def new_target_energy(self):
        if not self.inference:
            if self.curriculum.check_criterion(self.r_epi, self.E_d):
                self.curriculum.grow()
            self.E_d = self.curriculum.sample()

    def gather_data(self):
        # Model data
        p_model, v_model = self.sim.get_state_model()
        E = self.sim.get_energies()

        # CPG data
        p_gen = self.sim.get_state_generator()  # TODO: Test to feed in the CPG position as well

        tau = self.sim.controller.get_force_traj()[-1]
        state = {'Pos_model': p_model, 'Vel_model': v_model, 'Pos_gen': p_gen, 'Energy_des': self.E_d,
                 'Energies': E, 'Torque': tau}
        obs = np.concatenate([p_model, np.asarray(v_model) / MAX_SPEED,
                              np.asarray(p_gen) / ACTION_SCALE[1], [self.E_d / MAX_ENERGY]])

        return state, obs

    def step(self, action):
        try:
            # Format and take action
            # action[0] += 1  #  Enable this for tuning
            # action[1] = 0.5  # TODO: Train without this
            action = np.multiply(self.action_scale, action)  # TODO: Format in its own function
            self.sim.step(action)

            # Extract data for learning
            state, obs = self.gather_data()
            reward = self.reward_func(state)
            done = self.sim.is_done()
            info = {}  # TODO: Add some debugging info

            # March on
            self.cur_step += 1

            # Handle episode end
            if done:
                info['TimeLimit.truncated'] = True  # Tell the RL that the episode has limited episode duration

            return obs, reward, done, info

        except KeyboardInterrupt:
            del self.sim
            print("Closing the program due to Keyboard interrupt.")
            self.close()

        return None

    def reset(self):
        self.new_target_energy()
        self.sim.restart(self.E_d)

        self.r_epi = 0
        self.cur_step = 0
        self.r_traj = np.asarray([self.r_epi])
        self.E_t_traj = np.asarray([self.E_d])
        _, obs = self.gather_data()

        return obs

    def plot_reward(self):
        try:
            t_traj = self.sim.model.get_temporal_traj()
            E_d_traj = np.ones(np.shape(t_traj)) * self.E_d

            plt.figure('Reward')

            # Reward vs time
            ax = plt.subplot(2, 1, 1)
            ax.clear()
            plt.axis([0, self.sim.t_final, 0, 1])
            plt.plot(t_traj, self.r_traj, 'y--', linewidth=1)
            plt.ylabel(r'Reward')
            plt.xlabel('Time (s)')
            # plt.legend(['Pend. traj.'], loc='best')

            # Energies vs time
            ax = plt.subplot(2, 1, 2)
            ax.clear()
            ax.set_xlim([0, self.sim.t_final])
            ax.set_ylim([0, 1.5])
            plt.plot(t_traj, self.E_t_traj, 'b--', linewidth=1)
            plt.plot(t_traj, E_d_traj, 'g--', linewidth=1)
            plt.ylabel(r'Energy (J)')
            plt.xlabel('Time (s)')
            plt.legend([r'$E_t$', r'$E_d$'], loc='best')

            plt.draw()
            plt.pause(0.0001)

        except KeyboardInterrupt:
            plt.close(fig=plt.figure('System'))
            plt.close(fig=plt.figure('Reward'))
            raise KeyboardInterrupt

    def render(self, mode="human"):
        if mode == 'rgb_array':
            return
            # width, height = pg.size()
            # return np.asarray(pg.screenshot(region=(0, 0, width, height)))
        elif mode == 'human':
            pass
        else:
            super(BaseGymEnvironment, self).render(mode=mode)  # just raise an exception

