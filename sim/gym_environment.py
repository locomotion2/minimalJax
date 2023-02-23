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

from sim.environments import DoublePendulumCPGEnv, DoublePendulumDirectEnv, PendulumCPGEnv, PendulumDirectEnv
from sim.curricula import UniformGrowthCurriculum, BaseCurriculum
from sim.reward_functions import default_func


def p_norm(x, p):
    return np.power(np.sum(np.power(x, p)), 1 / p)


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

    def __init__(self, **kwargs):
        super().__init__()

        # Handle inputs
        params = {
            'delta_t_learning': ACTUAL_TIMESTEP,
            'delta_t_system': MIN_TIMESTEP,
            'solve': kwargs.get('solve', False)
        }

        mode = kwargs.get('mode', 'random_des')
        valid_modes = {'speed', 'position', 'equilibrium', 'random', 'random_des'}
        if mode not in valid_modes:
            raise ValueError(f"Selected initial condition mode {mode} is unknown. Valid modes: {valid_modes}")
        params['mode'] = mode

        self.final_time = kwargs.get('final_time', FINAL_TIME)
        params['t_final'] = self.final_time
        params['starting_range'] = kwargs.get('starting_range', [0, 1.2])

        self.reward_func = kwargs.get('reward_func', default_func)

        curriculum = kwargs.get('curriculum', None)
        if curriculum is None:
            self.curriculum = UniformGrowthCurriculum(starting_range=params['starting_range'])
        else:
            self.curriculum = curriculum

        energy_command = kwargs.get('energy_command', None)
        if energy_command is None:
            self.inference = False
            self.E_d = 0
        else:
            self.inference = True
            self.E_d = energy_command

        generator = kwargs.get('generator', None)
        self.system = kwargs.get('system', None)
        if generator is None or generator == 'CPG':
            if self.system is None or self.system == 'DoublePendulum':
                self.system = DoublePendulumCPGEnv
                params['state_size'] = 4
                params['num_dof'] = 2
                input_size = INPUT_SIZE_DPEND
            elif self.system == 'Pendulum':
                self.system = PendulumCPGEnv
                params['state_size'] = 2
                params['num_dof'] = 1
                input_size = INPUT_SIZE_PEND
            else:
                raise NotImplementedError(f"System {self.system} not implemented")
            self.action_scale = kwargs.get('action_scale', ACTION_SCALE_CPG)
            output_size = OUTPUT_SIZE_CPG
        elif generator == 'direct':
            if self.system is None or self.system == 'DoublePendulum':
                self.system = DoublePendulumDirectEnv
                params['state_size'] = 4
                params['num_dof'] = 2

                input_size = INPUT_SIZE_DPEND
            elif self.system == 'Pendulum':
                self.system = PendulumDirectEnv
                params['state_size'] = 2
                params['num_dof'] = 1
                input_size = INPUT_SIZE_PEND
            else:
                raise NotImplementedError(f"System {self.system} not implemented")
            self.action_scale = kwargs.get('action_scale', ACTION_SCALE_DIRECT)
            output_size = OUTPUT_SIZE_DIRECT
        else:
            raise NotImplementedError(f"Generator {generator} not implemented")

        # Rendering
        self.visualize = kwargs.get('render', False)
        if self.visualize:
            sns.set()
            figure = plt.figure('Visualization', figsize=FIG_SIZE)
            move_figure(figure, 0, 0)

        # Build environment
        self.action_space = Box(low=-1, high=1, shape=(output_size,))
        self.observation_space = Box(low=-1, high=1, shape=(input_size,))
        self.sim = self.system(params=params)

        # Tracking / help variables
        self.cur_step = 0
        self.num_dof = params['num_dof']
        self.r_epi = 0
        self.r_num = 3
        self.r_traj = np.asarray([np.asarray([self.r_epi] * (self.r_num + 1))])

    def tracking(self, reward: float, costs: list):
        # Tracking data
        self.r_traj = np.append(self.r_traj, [np.concatenate([[reward], costs], axis=0)], axis=0)

        # Tracking the episode
        self.r_epi += reward

    def new_target_energy(self):
        if not self.inference:
            if self.curriculum.check_criterion(self.r_epi, self.E_d):
                self.curriculum.grow()
            self.E_d = self.curriculum.sample()

    def gather_data(self):
        # Model data
        p_model, v_model = self.sim.get_cartesian_state_model()
        q_model, dq_model = self.sim.get_joint_state_model()
        E_model = self.sim.get_energies()

        # CPG data
        # p_gen = self.sim.get_cartesian_state_generator()
        q_gen, dq_gen = self.sim.get_joint_state_generator()
        params_gen = self.sim.get_params_generator()
        if self.num_dof == 1:
            q_gen = np.asarray([q_gen[0]])
            dq_gen = np.asarray([dq_gen[0]])

        # Controller data
        tau = self.sim.controller.get_force()

        # Build data packages
        state = {'Pos_model': p_model, 'Vel_model': v_model, 'Joint_pos': q_model, 'Joint_vel': dq_model,
                 'Pos_gen': q_gen, 'Vel_gen': dq_gen, 'Params_gen': params_gen,
                 'Energy_des': self.E_d, 'Energies': E_model, 'Torque': tau}
        obs = np.concatenate([np.asarray(q_model) / self.action_scale[1], np.asarray(dq_model) / MAX_SPEED,
                              np.asarray(q_gen) / self.action_scale[1], np.asarray(dq_gen) / MAX_SPEED,
                              [self.E_d / MAX_ENERGY]])

        # obs_temp = obs.copy()
        # obs_temp[np.abs(obs_temp) < 1] = 0
        # # debug_print('observations', obs[np.abs(obs) > 1])
        # debug_print('observations', obs_temp)

        return state, obs

    def step(self, action):
        try:
            # Format and take action
            # action[0] += 1  # Enable this for tuning
            action = np.multiply(self.action_scale, action)  # Scale up the action
            self.sim.step({'action': action, 'E_d': self.E_d, 'inference': self.inference})

            # Extract data for learning
            state, obs = self.gather_data()
            reward, costs = self.reward_func(state)
            done = self.sim.is_done()
            info = {}  # TODO: Add some debugging info

            # Track variables
            self.tracking(reward, costs)

            # March on
            self.cur_step += 1

            # Handle episode end
            if done:
                if self.visualize:
                    self.plot()
                info['TimeLimit.truncated'] = True  # Tell the RL that the episode has limited episode duration

            return obs, reward, done, info

        except KeyboardInterrupt:
            # del self.sim
            # self.close()
            print("Closing the program due to Keyboard interrupt.")
            raise KeyboardInterrupt

    def reset(self):

        # Reset system
        self.new_target_energy()
        self.sim.restart({'E_d': self.E_d})

        # Gather data from the new episode
        state, obs = self.gather_data()
        reward, costs = self.reward_func(state)

        # Reset tracking
        self.r_traj = np.append(self.r_traj, [np.concatenate([[reward], costs], axis=0)],
                                axis=0)  # TODO: The plotting looks off, the starting value is weird

        # Reset help variables
        self.r_epi = reward
        self.cur_step = 0

        return obs

    def plot(self):
        try:
            figure = plt.figure('Visualization')
            active_lines = [plt.Line2D] * 5
            index = 1

            # Prepare the underlying system to plot and unpack data
            data = self.sim.prepare_plot()
            [system_data, sim_data, sim_data_joints, rl_data, controller_data, energy_data] = data

            # Perform some calculations
            [rl_param_data, rl_traj_data] = rl_data
            [sim_model_data, sim_CPG_data, sim_CPG_traj_data] = sim_data
            [sim_model_data_joints, sim_model_traj_data_joints, sim_CPG_data_joints,
             sim_CPG_traj_data_joints] = sim_data_joints

            omega_traj = rl_traj_data['omega_traj'].to_numpy()
            # omega_avg = p_norm(omega_traj, 50)
            omega_avg = np.abs(omega_traj).mean()  # Todo: Find a better approximation, ask Antonin
            omega_var = np.abs(omega_traj).var()
            print(f'Omega: {omega_avg}')
            print(f'Omega Variance: {omega_var}')
            omega_avg = omega_avg + omega_var
            period = 2 * np.pi / omega_avg
            print(f'Period: {period}')
            x_omega = np.asarray([omega_avg, omega_avg])
            x_period = np.asarray([period, period])
            n_lines = int(np.ceil(self.sim.t_final / period))
            y_0 = [-LINE_DIST, LINE_DIST]

            param_gen_traj = rl_traj_data['gen'].to_numpy()
            # time_points = system_data['time'].to_numpy()
            # time_points = np.multiply(time_points, param_gen_traj)
            # time_points = time_points[time_points != 0]
            cpg_traj_x = sim_CPG_traj_data['x_traj_CPG'].to_numpy()
            cpg_traj_y = sim_CPG_traj_data['y_traj_CPG'].to_numpy()
            temp = np.multiply(cpg_traj_x, param_gen_traj)
            cpg_traj_x = cpg_traj_x[temp != 0]
            cpg_traj_y = cpg_traj_y[temp != 0]
            debug_print('zeros in param', np.size(param_gen_traj[param_gen_traj != 0]))
            debug_print('zeros in cpg', np.size(cpg_traj_x))

            cpg_joint_traj_x = sim_CPG_traj_data_joints['x_traj_CPG'].to_numpy()
            cpg_joint_traj_y = sim_CPG_traj_data_joints['y_traj_CPG'].to_numpy()
            temp = np.multiply(cpg_joint_traj_x, param_gen_traj)
            cpg_joint_traj_x = cpg_joint_traj_x[temp != 0]
            cpg_joint_traj_y = cpg_joint_traj_y[temp != 0]

            # Config. pos against time
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Joint pos. of sys. and CPG in time')
            legend_entries = []
            for i in range(int((len(system_data.columns) - 1) / 2)):
                plt.plot('time', f'des_traj_{i}', '--', data=system_data, linewidth=3, alpha=0.5)
                plt.plot('time', f'cur_traj_{i}', '--', data=system_data, linewidth=1)
                legend_entries.append(f'Des. traj. #{i}')
                legend_entries.append(f'Sys. traj. #{i}')
            for n in range(n_lines):
                plt.plot(x_period * n, y_0, '--', alpha=0.3)
            plt.ylabel(r'$Angle (rad)$')
            plt.xlabel('Time (s)')
            plt.legend(legend_entries, loc='best')
            plt.axis([0, self.sim.t_final, 2 * 180, 2 * -180])
            index += 1

            # Pendulum simulation and CPG path
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('System sim. and CPG in cart. coordinates')
            plt.plot('x_traj_CPG', 'y_traj_CPG', 'g*-', linewidth=1, alpha=0.3, data=sim_CPG_traj_data)
            active_lines[0], = plt.plot('x_model', 'y_model', 'bo-', linewidth=2, data=sim_model_data)
            active_lines[1], = plt.plot('x_CPG', 'y_CPG', 'o', linewidth=10, alpha=0.6, color='orange',
                                        data=sim_CPG_data)
            plt.plot(cpg_traj_x, cpg_traj_y, 'o', linewidth=2, alpha=0.2, color='purple')
            plt.ylabel(r'$Y-Pos. (m)$')
            plt.xlabel(r'$X-Pos. (m)$')
            plt.legend(['CPG Path', 'Pendulum', 'Oscillator', 'Pot. Gen. Pts.'], loc='best')
            window = 1.5
            plt.axis([-window, window, -window, window])
            index += 1

            # Pendulum simulation and CPG path in joint coords
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('System sim. and CPG in joint. coordinates')
            plt.plot('x_traj_CPG', 'y_traj_CPG', 'g*-', linewidth=1, alpha=0.3, data=sim_CPG_traj_data_joints)
            plt.plot('x_traj_model', 'y_traj_model', '*-', linewidth=1, alpha=0.4, color='lightblue',
                     data=sim_model_traj_data_joints)
            active_lines[2], = plt.plot('x_model', 'y_model', 'bo', linewidth=10, data=sim_model_data_joints)
            active_lines[3], = plt.plot('x_CPG', 'y_CPG', 'o', linewidth=10, alpha=0.6, color='orange',
                                        data=sim_CPG_data_joints)
            plt.plot(cpg_joint_traj_x, cpg_joint_traj_y, 'o', linewidth=2, alpha=0.2, color='purple')
            plt.ylabel(r'$q_2 (rad)$')
            plt.xlabel(r'$q_1 (rad)$')
            plt.legend(['CPG Path', 'Model Path', 'Pendulum', 'Oscillator'], loc='best')
            window = 180
            plt.axis([-window, window, -window, window])
            index += 1

            # RL-params
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Output of the RL-algorithm, param. of the CPG')
            plt.plot('omega_traj', 'mu_traj', '*-', linewidth=1, alpha=0.2, color='fuchsia', data=rl_traj_data)
            active_lines[4], = plt.plot('omega', 'mu', 'o', linewidth=2, alpha=0.7, color='purple', data=rl_param_data)
            plt.plot(x_omega, y_0, '--', alpha=0.5, color='purple')
            plt.plot(-x_omega, y_0, '--', alpha=0.5, color='purple')
            plt.xlabel(r'$\omega\,(hz)$')
            plt.ylabel(r'$\mu^2\,(m^2)$')
            [h_window, v_window] = self.action_scale[0:2]
            # v_window = v_window ** 2
            plt.axis([-h_window, h_window, -v_window * 0.5, v_window * 1.5])
            index += 1

            # PID controller
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Output of the PID controller and sys. torque')
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
            reward_data = pd.DataFrame(
                {'reward': self.r_traj[:, 0], 'cos_E': self.r_traj[:, 1], 'cos_tau': self.r_traj[:, 2],
                 'cos_pos': self.r_traj[:, 3]})
            reward_data = pd.concat([time_data, reward_data], axis=1)
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Reward breakdown of the episode')
            plt.axis([0, self.sim.t_final, 0, 1.1])
            plt.plot('time', 'reward', '-', linewidth=3, color='gold', data=reward_data)
            plt.plot('time', 'cos_E', '--', linewidth=1, alpha=0.5, color='black', data=reward_data)
            plt.plot('time', 'cos_tau', '--', linewidth=1, alpha=0.5, color='orange', data=reward_data)
            plt.plot('time', 'cos_pos', '--', linewidth=1, alpha=0.5, color='blue', data=reward_data)
            plt.ylabel(r'$Reward$')
            plt.xlabel(r'$Time (s)$')
            plt.legend(['Step reward', 'E. cost', r'$\tau\,cost$', 'Pos. cost'], loc='best')
            index += 1

            # Energies vs time
            E_d_traj = np.ones(len(time_data)) * self.E_d
            energy_data = pd.concat([time_data, pd.DataFrame({'energy_des': E_d_traj}), energy_data], axis=1)
            plt.subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Energy trajectory in time')
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
        passive_plots_indeces = (0, 4, 5, 6)

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
        while time < self.final_time:
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
