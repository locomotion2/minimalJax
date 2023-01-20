import pandas as pd

from sim.CONSTANTS import *

import gym
from gym.spaces import Box
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# import pandas as pd
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

        if mode in {'speed', 'position', 'equilibrium', 'random', 'random_des'}:
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

            sns.set()
            figure = plt.figure('Vizualization', figsize=FIG_SIZE)
            move_figure(figure, 0, 0)

        # Build environment
        self.action_space = Box(low=-1, high=1, shape=(OUTPUT_SIZE,))
        self.observation_space = Box(low=-1, high=1, shape=(INPUT_SIZE,))
        self.sim = BaseEnvironment(delta_t_learning=ACTUAL_TIMESTEP, delta_t_system=MIN_TIMESTEP,
                                   t_final=ACTUAL_FINAL_TIME, mode=self.mode, solve=self.solve)

        # Tracking / help variables
        self.r_epi = 0
        self.cur_step = 0
        self.r_traj = np.asarray([self.r_epi])

    def default_func(self, state: dict):
        # Cartesian reward
        pos_model = state['Pos_model']
        pos_gen = state['Pos_gen']
        dist = np.linalg.norm(pos_model - pos_gen)
        cost_cart = gaus(dist, 0.1)

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
        costs = np.asarray([cost_E_t, cost_E_k, cost_E_p, cost_torque, cost_cart])
        weights = np.asarray([0.4, 0.0, 0.0, 0.3, 0.3])
        cost_step = np.dot(costs, weights)

        return cost_step

    def tracking(self, reward: float):
        # Tracking data
        self.r_traj = np.append(self.r_traj, reward)

        # Tracking the episode
        self.r_epi += reward

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
            action = np.multiply(self.action_scale, action)  # TODO: Format in its own function
            self.sim.step(action)

            # Extract data for learning
            state, obs = self.gather_data()
            reward = self.reward_func(state)
            done = self.sim.is_done()
            info = {}  # TODO: Add some debugging info

            # Track variables
            self.tracking(reward)

            # March on
            self.cur_step += 1

            # Handle episode end
            if done:
                if self.visualize:
                    self.plot()
                info['TimeLimit.truncated'] = True  # Tell the RL that the episode has limited episode duration

            return obs, reward, done, info

        except KeyboardInterrupt:
            del self.sim
            print("Closing the program due to Keyboard interrupt.")
            self.close()

        return None

    def reset(self):
        # Reset system
        self.new_target_energy()
        self.sim.restart(self.E_d)

        # Gather data from the new episode
        state, obs = self.gather_data()
        reward = self.reward_func(state)

        # Reset tracking
        self.r_traj = np.asarray([reward])  # TODO: The plotting looks off, the starting value is weird

        # Reset help variables
        self.r_epi = reward
        self.cur_step = 0

        return obs

    def plot(self):
        try:
            figure = plt.figure('Vizualization')
            active_lines = [plt.Line2D] * 3
            index = 1

            # Prepare the underlying system to plot and unpack data
            data = self.sim.prepare_plot()
            [system_data, sim_data, rl_data, controller_data, energy_data] = data

            # Config. pos against time
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.plot('time', 'des_traj', 'g--', data=system_data, linewidth=3, alpha=0.5)
            plt.plot('time', 'cur_traj', 'b--', data=system_data, linewidth=1)
            plt.ylabel(r'$Angle (rad)$')
            plt.xlabel('Time (s)')
            plt.legend(['Des. traj.', 'Sys. traj.'], loc='best')
            plt.axis([0, self.sim.t_final, 180, -180])
            index += 1

            # Pendulum simulation and CPG path
            [sim_model_data, sim_CPG_data, sim_CPG_traj_data] = sim_data
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            active_lines[0], = plt.plot('x_model', 'y_model', 'bo-', linewidth=2, data=sim_model_data)
            active_lines[1], = plt.plot('x_CPG', 'y_CPG', 'o', linewidth=10, alpha=0.6, color='orange', data=sim_CPG_data)
            plt.plot('x_traj_CPG', 'y_traj_CPG', 'g*-', linewidth=1, alpha=0.2, data=sim_CPG_traj_data)
            plt.ylabel(r'$Y-Pos. (m)$')
            plt.xlabel(r'$X-Pos. (m)$')
            plt.legend(['Pendulum', 'CPG', 'Path'], loc='best')
            window = 1.5
            plt.axis([-window, window, -window, window])
            index += 1

            # RL-params
            [rl_param_data, rl_traj_data] = rl_data
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.plot('mu_traj', 'omega_traj', '*-', linewidth=1, alpha=0.4, color='fuchsia', data=rl_traj_data)
            active_lines[2], = plt.plot('mu', 'omega', 'o', linewidth=2, alpha=0.7, color='purple', data=rl_param_data)
            plt.ylabel(r'$\omega\,(hz)$')
            plt.xlabel(r'$\mu^2\,(m)$')
            [h_window, v_window] = ACTION_SCALE
            v_window = v_window ** 2
            plt.axis([-h_window, h_window, 0, v_window])
            index += 1

            # PID controller
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.axis([0, self.sim.t_final, -MAX_TORQUE, MAX_TORQUE])
            plt.plot('time', 'torque', '--', linewidth=2, color='orange', data=controller_data)
            plt.plot('time', 'e_P', 'b--', linewidth=1, alpha=0.7, data=controller_data)
            plt.plot('time', 'e_I', 'k--', linewidth=1, alpha=0.3, data=controller_data)
            plt.plot('time', 'e_D', 'g--', linewidth=1, alpha=0.7, data=controller_data)
            plt.ylabel(r'$Force (Nm)$')
            plt.xlabel('Time (s)')
            plt.legend(['Cont. Out', '$e_P$', '$e_I$', '$e_D$'], loc='best')
            index += 1

            # Reward vs time
            time_data = system_data.loc[:, 'time']
            reward_data = pd.DataFrame({'reward': self.r_traj})
            reward_data = pd.concat([time_data, reward_data], axis=1)
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.axis([0, self.sim.t_final, 0, 1])
            plt.plot('time', 'reward', '--', linewidth=3, color='gold', data=reward_data)
            plt.ylabel(r'$Reward$')
            plt.xlabel(r'$Time (s)$')
            plt.legend(['Step reward'], loc='best')
            index += 1

            # Energies vs time
            E_d_traj = np.ones(len(time_data)) * self.E_d
            energy_data = pd.concat([time_data, pd.DataFrame({'energy_des': E_d_traj}), energy_data], axis=1)
            plt.subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.axis([0, self.sim.t_final, 0, 1.5])  # TODO: Set to constants
            plt.plot('time', 'energy', 'b--', linewidth=1, data=energy_data)
            plt.plot('time', 'energy_des', 'g--', linewidth=1, data=energy_data)
            plt.ylabel(r'$Energy (J)$')
            plt.xlabel(r'$Time (s)$')
            plt.legend([r'$E_t$', r'$E_d$'], loc='best')
            index += 1
            
            # Show the plots
            figure.canvas.draw()
            plt.pause(0.0001)

            # Animate plots
            self.animate(figure, active_lines)

            # Leave the plot open
            plt.ioff()  # TODO: Look at what this actually does
            plt.show()

        except KeyboardInterrupt:
            plt.close(fig=plt.figure('System'))
            plt.close(fig=plt.figure('Reward'))
            raise KeyboardInterrupt

    def animate(self, figure, active_lines):
        # Helping variables
        time = 0
        x_0 = [0, 0]
        y_0 = [-LINE_DIST, LINE_DIST]
        
        # Types of plots for animation
        passive_plots_indeces = (0, 3, 4, 5)
        
        # Define the lines that will be animated
        passive_plots = [figure.get_axes()[i] for i in passive_plots_indeces]
        plot_num = len(passive_plots)
        passive_lines = [plt.Line2D] * plot_num
        
        # Initialize the passive lines
        for i in range(plot_num):
            passive_lines[i], = passive_plots[i].plot(x_0, y_0, 'r', alpha=0.5)

        def line_animation_step(time: float = 0, line: [plt.Line2D] = None):
            x = [time, time]
            y = [-LINE_DIST, LINE_DIST]
            line.set_ydata(y)
            line.set_xdata(x)
            
        # Main animation loop
        step = 0
        while time < ACTUAL_FINAL_TIME:
            # Update passive line positions
            for i in range(plot_num):
                line_animation_step(time, passive_lines[i])

            # Animate the complex plots
            self.sim.animate(step, active_lines)

            # Update vars and figure
            time += ACTUAL_TIMESTEP * VIZ_RATE
            step += VIZ_RATE
            figure.canvas.draw()
            figure.canvas.flush_events()

    def render(self, mode="human"):
        if mode == 'rgb_array':
            return
            # width, height = pg.size()
            # return np.asarray(pg.screenshot(region=(0, 0, width, height)))
        elif mode == 'human':
            pass
        else:
            super(BaseGymEnvironment, self).render(mode=mode)  # just raise an exception
