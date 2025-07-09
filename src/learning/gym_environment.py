import pandas as pd

from src.CONSTANTS import *

import gymnasium as gym
from gymnasium.spaces import Box
import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from PIL import ImageGrab

from src.environments.dpendulum import DoublePendulumCPGEnv, DoublePendulumDirectEnv
from src.environments.pendulum import PendulumCPGEnv, PendulumDirectEnv
from src.learning.curricula import UniformGrowthCurriculum
from src.learning.reward_functions import default_func


def p_norm(x, p):
    return jnp.power(jnp.sum(jnp.power(x, p)), 1 / p)


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    import matplotlib
    backend = matplotlib.get_backend()
    window = f.canvas.manager.window
    if backend == 'TkAgg':
        window.wm_geometry(f"+{x}+{y}")
    elif backend == 'WXAgg':
        window.SetPosition((x, y))
    else:
        # Qt backends: Try move() if available, else setGeometry
        if hasattr(window, "move"):
            window.move(x, y)
        elif hasattr(window, "setGeometry"):
            # width and height in pixels
            width, height = int(f.get_figwidth() * f.dpi), int(f.get_figheight() * f.dpi)
            window.setGeometry(x, y, width, height)



class BaseGymEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

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
            raise ValueError(
                f"Selected initial condition mode {mode} is unknown. Valid modes: {valid_modes}")
        params['mode'] = mode

        self.final_time = kwargs.get('final_time', FINAL_TIME)
        params['t_final'] = self.final_time
        params['starting_range'] = kwargs.get('starting_range', [0, 1.5])
        self.render_mode = kwargs.get('render_mode')

        self.reward_func = kwargs.get('reward_func', default_func)

        curriculum = kwargs.get('curriculum', None)
        if curriculum is None:
            self.curriculum = UniformGrowthCurriculum(
                min_difficulty=params['starting_range'][0],
                max_difficulty=params['starting_range'][1]
            )
        else:
            self.curriculum = curriculum

        energy_command = kwargs.get('energy_command', None)
        self.energy_step = kwargs.get('energy_step', False)
        if energy_command is None:
            self.inference = False
            self.E_d = 0
        else:
            self.inference = True
            self.E_d = energy_command

        energy_observer = kwargs.get('energy_observer', None)
        if energy_observer is None or energy_observer == 'model':
            self.energy_observer = None

        generator = kwargs.get('generator', None)
        self.system = kwargs.get('system', None)
        if generator is None or generator == 'CPG':
            if self.system is None or self.system == 'DoublePendulum':
                self.system = DoublePendulumCPGEnv
                params['num_dof'] = 2
            elif self.system == 'Pendulum':
                self.system = PendulumCPGEnv
                params['num_dof'] = 1
            else:
                raise NotImplementedError(f"System {self.system} not implemented")
            self.action_scale = jnp.asarray(kwargs.get('action_scale', ACTION_SCALE_CPG))
            
            # --- CORRECTED SECTION ---
            # The original code attempted to modify a JAX array in-place, which is not allowed.
            # It now correctly uses the .at[].set() method to update the array immutably.
            omega_scale = jnp.asarray([self.action_scale[0]] * params['num_dof'])
            omega_scale = omega_scale.at[-1].set(self.action_scale[-1])
            self.action_scale = omega_scale
            # --- END OF CORRECTION ---
            
            output_size = params['num_dof']
        elif generator == 'direct':
            if self.system is None or self.system == 'DoublePendulum':
                self.system = DoublePendulumDirectEnv
                params['num_dof'] = 2
            elif self.system == 'Pendulum':
                self.system = PendulumDirectEnv
                params['num_dof'] = 1
            else:
                raise NotImplementedError(f"System {self.system} not implemented")
            self.action_scale = jnp.asarray(
                kwargs.get('action_scale', ACTION_SCALE_DIRECT))
            q_scale = jnp.asarray([self.action_scale[0]] * params['num_dof'])
            q_d_scale = jnp.asarray([self.action_scale[1]] * params['num_dof'])
            self.action_scale = jnp.concatenate([q_scale, q_d_scale])
            output_size = params['num_dof'] * 2
        else:
            raise NotImplementedError(f"Generator {generator} not implemented")
        params['state_size'] = params['num_dof'] * 2
        input_size = params['state_size'] * 2 + 1

        # Rendering
        self.visualize = kwargs.get('render', False)
        self.record = kwargs.get('record', False)
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
        self.r_traj = jnp.asarray([jnp.asarray([self.r_epi] * (self.r_num + 1))])
        self.E_l_traj = jnp.asarray([jnp.asarray([0])])

    def tracking(self, reward: float, costs: list, energies: tuple):
        # Tracking data
        self.r_traj = jnp.append(self.r_traj,
                                [jnp.concatenate([[reward], costs], axis=0)], axis=0)
        self.E_l_traj = jnp.append(self.E_l_traj, [energies[0] + energies[1]])

        # Tracking the episode
        self.r_epi += reward

    def new_target_energy(self):
        if not self.inference:
            # You might need to define a specific success condition for your task.
            # Here, we assume a simple condition where any positive episode reward is a success.
            MIN_SCORE_FOR_SUCCESS = 0.0  # Placeholder: Adjust this threshold as needed.
            success_rate = 1.0 if self.r_epi > MIN_SCORE_FOR_SUCCESS else 0.0
            self.curriculum.update(success_rate)
            self.E_d = self.curriculum.get_difficulty()

    def gather_data(self):
        # Model data
        p_model, v_model = self.sim.get_cartesian_state_model()
        q_model, dq_model = self.sim.get_joint_state_model()
        if self.energy_observer is not None:
            E_model = self.energy_observer.get_energies(q_model, dq_model)
        else:
            E_model = self.sim.get_energies()

        # CPG data
        q_gen, dq_gen = self.sim.get_joint_state_generator()
        params_gen = self.sim.get_params_generator()

        # Controller data
        tau = self.sim.controller.get_force()

        # Check if energy step desired
        if self.sim.is_time_energy_step() and self.energy_step:
            self.E_d /= 2

        # Build data packages
        state = {'Pos_model': p_model, 'Vel_model': v_model, 'Joint_pos': q_model,
                 'Joint_vel': dq_model,
                 'Pos_gen': q_gen, 'Vel_gen': dq_gen, 'Params_gen': params_gen,
                 'Energy_des': self.E_d, 'Energies': E_model, 'Torque': tau}
        # TODO: maybe add a constant MAX_POSTITION
        obs = jnp.concatenate([q_model / self.action_scale[-1], dq_model / MAX_SPEED,
                              q_gen / self.action_scale[-1], dq_gen / MAX_SPEED,
                              jnp.asarray([self.E_d / MAX_ENERGY])])

        return state, obs

    def step(self, action):
        try:
            # Format and take action
            # action[0] += 1  # Enable this for tuning
            action = jnp.multiply(self.action_scale, action)  # Scale up the action
            self.sim.step(
                {'action': action, 'E_d': self.E_d, 'inference': self.inference})

            # Extract data for training
            state, obs = self.gather_data()
            reward, costs = self.reward_func(state)
            terminated = self.sim.is_done()
            info = {}  # TODO: Add some debugging info

            # Track variables
            self.tracking(reward, costs, state['Energies'])

            # March on
            self.cur_step += 1

            # Handle episode end
            if terminated:
                if self.visualize:
                    self.plot()
                # info['TimeLimit.truncated'] = True  # Tell the RL that the episode has limited episode duration

            truncated = terminated

            return obs.astype(jnp.float32), reward, terminated, truncated, info

        except KeyboardInterrupt:
            print("Closing the program due to Keyboard interrupt.")
            raise KeyboardInterrupt

    def reset(self, seed=None, options=None):

        # Reset system
        self.new_target_energy()
        self.sim.restart({'E_d': self.E_d})

        # Gather data from the new episode
        state, obs = self.gather_data()
        reward, costs = self.reward_func(state)

        # Reset tracking
        self.r_traj = jnp.append(self.r_traj,
                                [jnp.concatenate([[reward], costs], axis=0)],
                                axis=0)  # TODO: The plotting looks off, the starting value is weird

        # Reset help variables
        self.r_epi = reward
        self.cur_step = 0

        return obs.astype(jnp.float32), {}

    def plot(self):
        try:
            figure = plt.figure('Visualization')
            active_lines = [plt.Line2D] * 5
            index = 1

            # Prepare the underlying system to plot and unpack data
            data = self.sim.prepare_plot()
            [system_data, sim_data, sim_data_joints, rl_data, controller_data,
             energy_data] = data

            # Perform some calculations
            [rl_param_data, rl_traj_data] = rl_data
            [sim_model_data, sim_CPG_data, sim_CPG_traj_data] = sim_data
            [sim_model_data_joints, sim_model_traj_data_joints, sim_CPG_data_joints,
             sim_CPG_traj_data_joints] = sim_data_joints

            omega_traj = rl_traj_data['omega_traj'].to_numpy()
            omega_avg = jnp.abs(
                omega_traj).mean()
            omega_var = jnp.abs(omega_traj).var()
            print(f'Omega: {omega_avg}')
            print(f'Omega Variance: {omega_var}')
            omega_avg = omega_avg + omega_var
            period = 2 * jnp.pi / omega_avg
            print(f'Period: {period}')
            x_omega = jnp.asarray([omega_avg, omega_avg])
            x_period = jnp.asarray([period, period])
            n_lines = int(jnp.ceil(self.sim.t_final / period))
            y_0 = [-LINE_DIST, LINE_DIST]

            param_gen_traj = rl_traj_data['gen'].to_numpy()
            cpg_traj_x = sim_CPG_traj_data['x_traj_CPG'].to_numpy()
            cpg_traj_y = sim_CPG_traj_data['y_traj_CPG'].to_numpy()
            temp = jnp.multiply(cpg_traj_x, param_gen_traj)
            cpg_traj_x = cpg_traj_x[temp != 0]
            cpg_traj_y = cpg_traj_y[temp != 0]

            cpg_joint_traj_x = sim_CPG_traj_data_joints['x_traj_CPG'].to_numpy()
            cpg_joint_traj_y = sim_CPG_traj_data_joints['y_traj_CPG'].to_numpy()
            temp = jnp.multiply(cpg_joint_traj_x, param_gen_traj)
            cpg_joint_traj_x = cpg_joint_traj_x[temp != 0]
            cpg_joint_traj_y = cpg_joint_traj_y[temp != 0]

            # Config. pos against time
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Joint pos. of sys. and CPG through time')
            legend_entries = []
            for i in range(int((len(system_data.columns) - 1) / 2)):
                plt.plot('time', f'des_traj_{i}', '--', data=system_data, linewidth=2,
                         alpha=1)
                plt.plot('time', f'cur_traj_{i}', linewidth=3, alpha=0.5,
                         data=system_data)
                legend_entries.append(f'Des. path. q{i}')
                legend_entries.append(f'Sys. path. q{i}')
            for n in range(n_lines):
                plt.plot(x_period * n, y_0, '--', alpha=0.3)
            plt.ylabel(r'$Angle\,(rad)$')
            plt.xlabel('Time (s)')
            plt.legend(legend_entries, loc='best')
            plt.axis([0, self.sim.t_final, 2 * 180, 2 * -180])
            index += 1

            # Pendulum simulation and CPG path
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('System src. and CPG in cart. coordinates')
            plt.plot('x_traj_CPG', 'y_traj_CPG', 'g*-', linewidth=0.5, alpha=0.3,
                     data=sim_CPG_traj_data)
            active_lines[0], = plt.plot('x_model', 'y_model', 'bo-', linewidth=2,
                                        data=sim_model_data)
            active_lines[1], = plt.plot('x_CPG', 'y_CPG', 'o', linewidth=10, alpha=0.6,
                                        color='orange',
                                        data=sim_CPG_data)
            plt.plot(cpg_traj_x, cpg_traj_y, 'o', linewidth=2, alpha=0.2,
                     color='purple')
            plt.ylabel(r'$Vert.\,Pos.\,(m)$')
            plt.xlabel(r'$Hor.\,Pos.\,(m)$')
            plt.legend(['CPG Path', 'Pendulum Sim.', r'$CPG\,q_{des}$', 'Pot. Gen. '
                                                                        'Pts.'],
                       loc='best')
            window = 1.3
            plt.axis([-window, window, -window, window])
            index += 1

            # Pendulum simulation and CPG path in joint coords
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('System src. and CPG in joint. coordinates')
            plt.plot('x_traj_CPG', 'y_traj_CPG', 'g*-', linewidth=1, alpha=0.5,
                     data=sim_CPG_traj_data_joints)
            plt.plot('x_traj_model', 'y_traj_model', 'b*-', linewidth=1, alpha=0.3,
                     data=sim_model_traj_data_joints)

            active_lines[3], = plt.plot('x_CPG', 'y_CPG', 'o', linewidth=10, alpha=0.6,
                                        color='orange',
                                        data=sim_CPG_data_joints)
            active_lines[2], = plt.plot('x_model', 'y_model', 'o', linewidth=10,
                                        color='lightblue', alpha=0.6,
                                        data=sim_model_data_joints)
            plt.plot(cpg_joint_traj_x, cpg_joint_traj_y, 'o', linewidth=2, alpha=0.2,
                     color='purple')
            plt.ylabel(r'$q_2\,(rad)$')
            plt.xlabel(r'$q_1\,(rad)$')
            plt.legend(['CPG Path', 'Model Path', r'$CPG\,q_{des}$', 'Pendulum Sim.'],
                       loc='best')
            window = 200
            plt.axis([-window, window, -window, window])
            index += 1

            # RL-params
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Output of the RL-agent, param. of the CPG')
            plt.plot('omega_traj', 'mu_traj', linewidth=1, alpha=0.4,
                     color='fuchsia', data=rl_traj_data)
            active_lines[4], = plt.plot('omega', 'mu', 'o', linewidth=2, alpha=0.7,
                                        color='purple', data=rl_param_data)
            plt.plot(x_omega, y_0, '--', alpha=0.5, color='purple')
            plt.plot(-x_omega, y_0, '--', alpha=0.5, color='purple')
            plt.xlabel(r'$\omega\,(hz)$')
            plt.ylabel(r'$\mu^2\,(m^2)$')
            h_window = self.action_scale[0]
            v_window = self.action_scale[-1]
            plt.axis([-h_window, h_window, -v_window * 0.5, v_window * 1.5])
            index += 1

            # PID controller
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Output of the PID controller and sys. torque')
            plt.axis([0, self.sim.t_final, -MAX_TORQUE, MAX_TORQUE])
            plt.plot('time', 'torque', linewidth=3, color='orange',
                     data=controller_data)
            plt.plot('time', 'e_P', 'b--', linewidth=2, alpha=0.5, data=controller_data)
            plt.plot('time', 'e_I', 'k--', linewidth=2, alpha=0.3, data=controller_data)
            plt.plot('time', 'e_D', 'g--', linewidth=2, alpha=0.5, data=controller_data)
            plt.ylabel(r'$Force (Nm)$')
            plt.xlabel('Time (s)')
            plt.legend([r'$e_{tot}=\tau$', '$e_P$', '$e_I$', '$e_D$'], loc='best')
            index += 1

            # Reward vs time
            time_data = system_data.loc[:, 'time']
            reward_data = pd.DataFrame(
                {'reward': self.r_traj[:, 0], 'cos_E': self.r_traj[:, 1],
                 'cos_tau': self.r_traj[:, 2],
                 'cos_pos': self.r_traj[:, 3]})
            reward_data = pd.concat([time_data, reward_data], axis=1)
            figure.add_subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Reward breakdown of the episode')
            plt.axis([0, self.sim.t_final, 0, 1.1])
            plt.plot('time', 'cos_E', '--', linewidth=2, alpha=0.2, color='black',
                     data=reward_data)
            plt.plot('time', 'cos_tau', '--', linewidth=2, alpha=0.2, color='brown',
                     data=reward_data)
            plt.plot('time', 'cos_pos', '--', linewidth=2, alpha=0.2, color='blue',
                     data=reward_data)
            plt.plot('time', 'reward', '-', linewidth=3, color='gold', data=reward_data)
            plt.ylabel(r'Reward')
            plt.xlabel(r'Time (s)')
            plt.legend(['E. track.', r'$\tau$ track.', 'Pos. track.', 'Step reward'],
                       loc='best')
            index += 1

            # Energies vs time
            samples = len(time_data)
            E_d_traj = jnp.ones(samples) * self.E_d
            if self.energy_step:
                E_d_traj[0:int(jnp.floor(samples / 2))] *= 2

            energy_data = pd.concat([time_data,
                                     pd.DataFrame({'energy_des': E_d_traj,
                                                   'energy_model': self.E_l_traj}),
                                     energy_data], axis=1)
            plt.subplot(FIG_COORDS[0], FIG_COORDS[1], index)
            plt.title('Energy trajectory through time')
            plt.plot('time', 'energy_des', 'g', linewidth=3, data=energy_data)
            plt.plot('time', 'energy', 'b', linewidth=3, data=energy_data, alpha=0.5)
            plt.plot('time', 'energy_model', 'r--', linewidth=2, data=energy_data,
                     alpha=0.7)
            plt.ylabel(r'Energy (J)')
            plt.xlabel(r'Time (s)')
            plt.legend([r'$E_{ana}$', r'$E_{des}$', r'$E_{NN}$'], loc='best')
            plt.axis([0, self.sim.t_final, 0, 1.3])  # TODO: Set to constants
            index += 1

            # Show the plots
            figure.canvas.draw()
            figure.tight_layout()
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

        if self.record:
            # Testing the video recording
            screen_size = (1920, 1080)
            fps = 10

            # Create a video writer object
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter("media/experiment recording.avi", fourcc, fps,
                                  screen_size)

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

            if self.record:
                img = ImageGrab.grab(bbox=(0, 0, screen_size[0], screen_size[1]))

                # Convert the screenshot to a numpy array
                img_np = jnp.array(img)

                # Convert the color space from RGB to BGR
                frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Write the current frame to the video
                out.write(frame)

        if self.record:
            # Release the video writer and close the video file
            out.release()
            cv2.destroyAllWindows()

    def render(self, mode="human"):
        if mode == 'rgb_array':
            return
            # width, height = pg.size()
            # return jnp.asarray(pg.screenshot(region=(0, 0, width, height)))
        elif mode == 'human':
            pass
        else:
            super(BaseGymEnvironment, self).render(mode=mode)  # just raise an exception