from sim.CONSTANTS import *

from sim.controllers import PID_pos_vel_damping, PID_pos_vel_tracking_num, PID_pos_vel_tracking_modeled
from sim.models import CPG, Pendulum

from IPython import display
import matplotlib.pyplot as plt

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

    def get_state_model(self):
        return self.model.get_cartesian_state()

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

    def plot(self):  # TODO: Implement easy closing, transfer methods to underlying classes
        try:
            plt.figure('System')

            # Model
            # Config. pos against time
            t_traj = self.model.get_temporal_traj()
            x_traj_model = self.model.get_state_traj()
            q_traj_model = ((x_traj_model[:, 0] + np.pi) % (2 * np.pi)) - np.pi
            q_d_traj = ((self.controller.get_desired_traj() + np.pi) % (2 * np.pi)) - np.pi
            ax = plt.subplot(2, 2, 1)
            ax.clear()
            plt.plot(t_traj, q_d_traj * 180 / np.pi, 'g--', linewidth=3, alpha=0.5)
            plt.plot(t_traj, q_traj_model * 180 / np.pi, 'b--', linewidth=1)
            plt.ylabel(r'$Angle (rad)$')
            plt.xlabel('Time (s)')
            plt.legend(['Des. traj.', 'Sys. traj.'], loc='best')
            plt.axis([0, self.t_final, 180, -180])

            # Pendulum simulation and CPG path
            ax = plt.subplot(2, 2, 2)
            ax.clear()
            [px_model, py_model] = self.model.get_cartesian_state()[0]
            x_traj_CPG = self.generator.get_state_traj()
            px_traj = x_traj_CPG[:, 0]
            py_traj = x_traj_CPG[:, 1]
            plt.plot([0, px_model], [0, py_model], 'b*-', linewidth=1)
            plt.plot(px_traj, py_traj, 'g*-', linewidth=1, alpha=0.2)
            plt.ylabel(r'$Y-Pos. (m)$')
            plt.xlabel(r'$X-Pos. (m)$')
            plt.legend(['Pendulum', 'CPG'], loc='best')
            window = 1.5
            plt.axis([-window, window, -window, window])

            # RL-params
            ax = plt.subplot(2, 2, 3)
            ax.clear()
            param_traj = self.generator.get_parametric_traj()
            param_x_traj = param_traj[:, 0]
            param_y_traj = param_traj[:, 1]**2
            plt.plot(param_x_traj, param_y_traj, 'o--', linewidth=2, alpha=0.4, color='purple')
            plt.ylabel(r'$\omega\,(hz)$')
            plt.xlabel(r'$\mu^2\,(m)$')
            # plt.legend(['Pendulum', 'CPG'], loc='best')
            # window = 1.5
            [h_window, v_window] = ACTION_SCALE
            v_window = v_window**2
            plt.axis([-h_window, h_window, 0, v_window])

            # PID controller
            tau_traj = self.controller.get_force_traj()
            e_traj = self.controller.get_error_traj()
            e_P = e_traj[:, 0]
            e_I = e_traj[:, 1]
            e_D = e_traj[:, 2]
            ax = plt.subplot(2, 2, 4)
            ax.clear()
            plt.plot(t_traj, tau_traj, '--', linewidth=2, color='orange')
            plt.plot(t_traj, e_P, 'b--', linewidth=1, alpha=0.7)
            plt.plot(t_traj, e_I, 'k--', linewidth=1, alpha=0.3)
            plt.plot(t_traj, e_D, 'g--', linewidth=1, alpha=0.7)
            plt.ylabel(r'$Force (Nm)$')
            plt.xlabel('Time (s)')
            plt.legend(['Cont. Out', '$e_P$', '$e_I$', '$e_D$'], loc='best')
            ax.set_xlim([0, self.t_final])

            # Show and wait
            plt.draw()
            # plt.pause(0.00001)

        except KeyboardInterrupt:
            plt.close(fig=plt.figure('System'))
            plt.close(fig=plt.figure('Reward'))
            raise KeyboardInterrupt
