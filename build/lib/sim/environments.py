from sim.CONSTANTS import *

from sim.controllers import PID_pos_vel_damping, PID_pos_vel_tracking_num, PID_pos_vel_tracking_modeled
from sim.models import CPG, Pendulum

from IPython import display
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

import numpy as np


class BaseEnvironment:
    def __init__(self, delta_t_learning: float = MIN_TIMESTEP, delta_t_system: float = MIN_TIMESTEP,
                 t_final: float = FINAL_TIME, mode: str = 'equilibrium', solve: bool = False):
        self.delta_t_system = delta_t_system
        self.delta_t_learning = delta_t_learning
        self.t_final = t_final
        self.t_elapsed = 0
        self.solve = solve
        self.mode = mode

        # Define parameters
        pendulum_params = {'delta_t': self.delta_t_system,
                           'state_size': 2,
                           'num_dof': 1
                           }
        controller_params = {'delta_t': self.delta_t_system,
                             'gains': [0.5, 0.0, 0.5]
                             }
        generator_params = {'delta_t': self.delta_t_learning,
                            'state_size': 2,
                            'num_dof': 2
                            }

        # Build components
        self.model = Pendulum(params=pendulum_params)
        self.controller = PID_pos_vel_tracking_modeled(params=controller_params)
        self.generator = CPG(params=generator_params)

        # sns.set()

    def step(self, action: np.ndarray):
        # Get RL params
        omega = action[0]
        mu = action[1]

        # Obtain solution from model to compare results
        [q_d_sol, dq_d_sol] = self.model.solve(self.t_elapsed)  # TODO: Plot next to current traj

        if not self.solve:
            # Generate next point in path
            self.generator.step({'omega': omega, 'mu': mu})
            self.generator.update_trajectories()
            coils = self.generator.detect_coiling()  # This needs the trajectories to be up-to-date
            [p, v] = self.generator.get_cartesian_state()

            # Run through inverse kins
            [q_d, dq_d] = self.model.inverse_kins({'pos': p, 'speed': v, 'coils': coils})
        else:
            [q_d, dq_d] = [q_d_sol, dq_d_sol]

        # Run the controller at a higher rate
        relative_time = 0
        tau = 0  # Container for the last resulting torque
        while relative_time < self.delta_t_learning:
            # Get the current model state in joint coords
            [q_cur, dq_cur] = self.model.get_joint_state()

            # Run through controller, get force
            step_inputs = {'q_d': q_d, 'dq_d': dq_d, 'q_cur': q_cur, 'dq_cur': dq_cur}
            tau = self.controller.input(inputs=step_inputs)

            # Apply controller action and update model
            self.model.step({'tau': tau})

            # Update the time
            relative_time += self.delta_t_system

        # Save latest trajectory for plotting
        self.model.update_trajectories()
        self.controller.update_trajectories(q_d, tau)
        self.t_elapsed += relative_time

    def get_cartesian_state_model(self):
        return self.model.get_cartesian_state()

    def get_joint_state_model(self):
        return self.model.get_joint_state()

    def get_state_generator(self):
        [p, v] = self.generator.get_cartesian_state()
        return p

    def get_energies(self):
        return self.model.get_energies()

    def is_done(self):
        return self.model.get_time() >= self.t_final

    def restart(self, E_d: float = 0):
        self.model.restart({'mode': self.mode, 'E_d': E_d})
        self.controller.restart()
        self.generator.restart({'x_0': [0, -1]})
        self.t_elapsed = 0

        # display.clear_output(wait=True)
        # plt.clf()

    def animate(self, step: int = 0, lines: [plt.Line2D] = None):
        p_traj_model = self.model.get_cartesian_traj()
        [px_model, py_model] = p_traj_model[step]
        lines[0].set_xdata([0, px_model])
        lines[0].set_ydata([0, py_model])

        p_traj_CPG = self.generator.get_state_traj()
        [px_CPG, py_CPG] = p_traj_CPG[step]
        lines[1].set_xdata(px_CPG)
        lines[1].set_ydata(py_CPG)

        param_traj = self.generator.get_parametric_traj()
        param_x_traj = param_traj[:, 0]
        param_y_traj = param_traj[:, 1] ** 2
        param_x = param_x_traj[step]
        param_y = param_y_traj[step]
        lines[2].set_xdata(param_x)
        lines[2].set_ydata(param_y)

    def prepare_plot(self):  # TODO: Implement easy closing, transfer methods to underlying classes
        try:
            # Config. pos against time
            t_traj = self.model.get_temporal_traj()
            x_traj_model = self.model.get_state_traj()
            q_traj_model = (((x_traj_model[:, 0] + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi
            q_d_traj = (((self.controller.get_desired_traj() + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi
            system_data = pd.DataFrame({'time': t_traj, 'des_traj': q_d_traj, 'cur_traj': q_traj_model})

            # Pendulum simulation and CPG path
            p_traj_model = self.model.get_cartesian_traj()
            [px_model, py_model] = p_traj_model[0]
            p_traj_CPG = self.generator.get_state_traj()
            [px_CPG, py_CPG] = p_traj_CPG[0]
            x_traj_CPG = self.generator.get_state_traj()
            px_traj = x_traj_CPG[:, 0]
            py_traj = x_traj_CPG[:, 1]
            sim_model_data = pd.DataFrame({'x_model': [0, px_model], 'y_model': [0, py_model]})
            sim_CPG_data = pd.DataFrame({'x_CPG': px_CPG, 'y_CPG': py_CPG}, index=[0])
            sim_CPG_traj_data = pd.DataFrame({'x_traj_CPG': px_traj, 'y_traj_CPG': py_traj})

            # RL-params
            param_traj = self.generator.get_parametric_traj()
            param_x_traj = param_traj[:, 0]
            param_y_traj = param_traj[:, 1] ** 2
            param_x = param_x_traj[0]
            param_y = param_y_traj[0]
            rl_param_data = pd.DataFrame({'mu': param_x, 'omega': param_y}, index=[0])
            rl_traj_data = pd.DataFrame({'mu_traj': param_x_traj, 'omega_traj': param_y_traj})

            # PID controller
            tau_traj = self.controller.get_force_traj()
            e_traj = self.controller.get_error_traj()
            e_P = e_traj[:, 0]
            e_I = e_traj[:, 1]
            e_D = e_traj[:, 2]
            controller_data = pd.DataFrame({'time': t_traj, 'torque': tau_traj, 'e_P': e_P, 'e_I': e_I, 'e_D': e_D})

            # Energies Plot
            E_traj = self.model.get_energy_traj()
            energy_data = pd.DataFrame({'time': t_traj, 'energy': E_traj})

            return [system_data, [sim_model_data, sim_CPG_data, sim_CPG_traj_data],
                    [rl_param_data, rl_traj_data], controller_data, energy_data]

        except KeyboardInterrupt:
            plt.close(fig=plt.figure('System'))
            plt.close(fig=plt.figure('Reward'))
            raise KeyboardInterrupt
