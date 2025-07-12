import pandas as pd
import numpy as np # Using numpy for Gym spaces
from src.CONSTANTS import *

import gymnasium as gym
from gymnasium.spaces import Box
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import ImageGrab

from src.environments.dpendulum import DoublePendulumCPGEnv, DoublePendulumDirectEnv
from src.environments.pendulum import PendulumCPGEnv, PendulumDirectEnv
from src.learning.curricula import UniformGrowthCurriculum
from src.learning.reward_functions import default_func


# --- Helper Functions (Not for JIT) ---
def move_figure(f, x, y):
    """Move matplotlib figure's upper left corner to pixel (x, y)"""
    import matplotlib
    backend = matplotlib.get_backend()
    if backend is None: return # Return if no backend
    window = f.canvas.manager.window
    if backend == 'TkAgg':
        window.wm_geometry(f"+{x}+{y}")
    elif backend == 'WXAgg':
        window.SetPosition((x, y))
    else:
        if hasattr(window, "move"):
            window.move(x, y)
        elif hasattr(window, "setGeometry"):
            width, height = int(f.get_figwidth() * f.dpi), int(f.get_figheight() * f.dpi)
            window.setGeometry(x, y, width, height)


class BaseGymEnvironment(gym.Env):
    """
    A JAX-accelerated Gymnasium Environment.

    This class uses the "stateful wrapper, stateless core" pattern.
    - The public methods (`step`, `reset`) are stateful (they modify `self.state`)
      and compatible with the standard Gymnasium API.
    - The internal, performance-critical logic is in `_jitted_step`, a pure,
      stateless function that is JIT-compiled by JAX.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, **kwargs):
        """
        Initializes the environment. This method is stateful and runs only once.
        """
        super().__init__()

        # --- Environment Configuration ---
        params = {
            'delta_t_learning': ACTUAL_TIMESTEP,
            'delta_t_system': MIN_TIMESTEP,
            'solve': kwargs.get('solve', False)
        }
        self.mode = kwargs.get('mode', 'random_des')
        self.final_time = kwargs.get('final_time', FINAL_TIME)
        params.update({
            'mode': self.mode,
            't_final': self.final_time,
            'starting_range': kwargs.get('starting_range', [0, 1.5])
        })
        self.render_mode = kwargs.get('render_mode')
        self.reward_func = kwargs.get('reward_func', default_func)

        # --- Curriculum Learning Setup ---
        self.curriculum = kwargs.get('curriculum') or UniformGrowthCurriculum(
            min_difficulty=params['starting_range'][0],
            max_difficulty=params['starting_range'][1]
        )

        # --- Energy and Inference Setup ---
        self.energy_step = kwargs.get('energy_step', False)
        self.inference = 'energy_command' in kwargs
        self.E_d = kwargs.get('energy_command', 0.0)
        self.energy_observer = kwargs.get('energy_observer')

        # --- System and Action Space Configuration ---
        generator = kwargs.get('generator', 'CPG')
        system_name = kwargs.get('system', 'DoublePendulum')
        
        system_map_cpg = {'DoublePendulum': DoublePendulumCPGEnv, 'Pendulum': PendulumCPGEnv}
        system_map_direct = {'DoublePendulum': DoublePendulumDirectEnv, 'Pendulum': PendulumDirectEnv}

        if generator == 'CPG':
            system_class = system_map_cpg[system_name]
            self.action_scale = jnp.asarray(kwargs.get('action_scale', ACTION_SCALE_CPG))
        elif generator == 'direct':
            system_class = system_map_direct[system_name]
            self.action_scale = jnp.asarray(kwargs.get('action_scale', ACTION_SCALE_DIRECT))
        else:
            raise NotImplementedError(f"Generator {generator} not implemented")

        params['num_dof'] = 2 if system_name == 'DoublePendulum' else 1
        params['state_size'] = params['num_dof'] * 2
        input_size = params['state_size'] * 2 + 1
        output_size = len(self.action_scale)

        # --- Rendering and Visualization (Not JIT-compatible) ---
        self.visualize = kwargs.get('render', False)
        self.record = kwargs.get('record', False)
        if self.visualize:
            sns.set()
            move_figure(plt.figure('Visualization', figsize=FIG_SIZE), 0, 0)

        # --- Gymnasium API Setup ---
        self.action_space = Box(low=-1.0, high=1.0, shape=(output_size,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(input_size,), dtype=np.float32)
        
        # Instantiate the underlying simulation
        self.sim = system_class(params=params)

        # --- State and Tracking Initialization ---
        self.r_num = 3
        self.key = jax.random.PRNGKey(0)
        # The 'state' is a PyTree that will be passed to JIT functions
        self.state = {} 
        self.r_traj = jnp.empty((0, self.r_num + 1))
        self.E_l_traj = jnp.empty((0, 1))
        self.r_epi = 0.0

    @staticmethod
    @partial(jit, static_argnums=(3,4,5))
    def _jitted_step(state, action, final_time, sim, reward_func, action_scale):
        """
        A pure, JIT-compiled function for one simulation step.
        This is the performance-critical core.
        - It is stateless: all inputs are passed as arguments in the `state` PyTree.
        - It is pure: it has no side-effects and returns the new state PyTree.
        """
        # Unpack the state PyTree
        sim_state = state['sim_state']
        t = state['t']
        
        # Scale the action
        scaled_action = action * action_scale

        # Step the underlying JAX-based simulation.
        # Crucially, sim.step must also be a pure function that takes the
        # state as input and returns the new state.
        next_sim_state = sim.step(sim_state, scaled_action)
        
        # Calculate reward and costs from the new state
        reward, costs = reward_func(next_sim_state)
        
        # Update time and check for termination
        t_next = t + 1
        terminated = (t_next >= final_time)
        
        # Pack and return the new state PyTree and other results
        next_state = {'sim_state': next_sim_state, 't': t_next}
        return next_state, reward, costs, terminated

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        The public, stateful step function that interfaces with the Gym API.
        This method acts as a "wrapper" around the JIT-compiled core.
        """
        action_jax = jnp.asarray(action)
        
        # Call the fast, pure, JIT-compiled step function
        next_state, reward_jax, costs_jax, terminated_jax = self._jitted_step(
            self.state, action_jax, self.final_time, self.sim, self.reward_func, self.action_scale
        )
        
        # This is the stateful part: update the environment's state.
        # This side-effect happens outside the JIT-compiled function.
        self.state = next_state
        
        # Tracking and observation logic also happens outside the JIT-ed path.
        self.tracking(reward_jax, costs_jax, self.state['sim_state']['Energies'])
        
        obs = self._state_to_obs(self.state['sim_state'])
        reward = float(reward_jax)
        terminated = bool(terminated_jax)
        info = {'costs': np.array(costs_jax, dtype=np.float32)}
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)

        # Update the target energy based on the curriculum
        self.new_target_energy()

        # Restart the underlying simulation
        self.sim.restart({'E_d': self.E_d})

        # Get the initial state from the simulation
        initial_sim_state = self.sim.get_state_and_update()

        # Initialize the full environment state
        self.state = {
            'sim_state': initial_sim_state,
            't': 0,
        }

        # Get initial observation and info dict for Gym
        obs = self._state_to_obs(self.state['sim_state'])
        _, costs = self.reward_func(self.state['sim_state']) # We only need costs for the info dict
        info = {'costs': np.array(costs)}

        # Reset episode-specific trackers
        self.r_epi = 0.0
        self.r_traj = jnp.empty((0, self.r_num + 1))
        self.E_l_traj = jnp.empty((0, 1))

        return obs, info

    def _state_to_obs(self, sim_state: dict) -> np.ndarray:
        """Converts the simulation state PyTree to a NumPy observation array."""
        obs_jax = jnp.concatenate([
            sim_state['Joint_pos'],
            sim_state['Joint_vel'],
            sim_state['Pos_gen'],
            sim_state['Vel_gen'],
            jnp.atleast_1d(sim_state['Energy_des'])
        ])
        return np.asarray(obs_jax)

    def tracking(self, reward: jax.Array, costs: jax.Array, energies: tuple):
        """
        Tracks data over an episode. This is stateful and not JIT-compiled.
        """
        new_reward_row = jnp.expand_dims(jnp.concatenate([jnp.atleast_1d(reward), costs]), 0)
        self.r_traj = jnp.concatenate([self.r_traj, new_reward_row])

        total_energy = jnp.atleast_1d(energies[0] + energies[1])
        new_energy_row = jnp.expand_dims(total_energy, 0)
        self.E_l_traj = jnp.concatenate([self.E_l_traj, new_energy_row])
        
        self.r_epi += reward

    def new_target_energy(self):
        """Updates the target energy. This is stateful."""
        if not self.inference:
            MIN_SCORE_FOR_SUCCESS = 0.0
            success_rate = 1.0 if self.r_epi > MIN_SCORE_FOR_SUCCESS else 0.0
            self.curriculum.update(success_rate)
            difficulty = self.curriculum.get_difficulty()
            self.E_d = 0.0 if difficulty is None else difficulty

    def close(self):
        """Closes any open resources (e.g., plotting windows)."""
        if self.visualize:
            plt.close('all')

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