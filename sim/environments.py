from sim.controllers import PID
from sim.models import CPG, Pendulum

from IPython import display
import matplotlib.pyplot as plt

import numpy as np


class BaseEnvironment:
    def __init__(self, delta_t: float = 0.001, t_final: float = 5, render: bool = False):
        self.delta_t = delta_t
        self.t_final = t_final
        self.render = render

        self.model = Pendulum(delta_t=delta_t)
        self.controller = PID(delta_t=delta_t, P=15, D=-0.75, I=0.1)
        self.generator = CPG(delta_t=delta_t, x_0=[0, -1])

        if render:
            # display.clear_output(wait=True)
            plt.figure(1)
            plt.clf()
            plt.ion()
            plt.show()
            # plt.figure(figsize=(10, 7))

    def step(self, action: np.ndarray):
        # Get params
        omega = action[0]
        mu = action[1]

        # Generate next point in traj
        self.generator.step(omega, mu)
        p_traj = self.generator.get_cartesian_pos()

        # Run through inverse kins and through controller
        q_d = self.model.inverse_kins(p_traj, self.generator.coils)
        q_cur = self.model.get_config_pos()
        dq_cur = self.model.get_config_speed()
        tau = self.controller.input(q_d, q_cur, dq_cur)

        # Apply controller action and update model
        self.model.step(tau)

        if self.render:
            self.plot()

    def get_positions(self):
        return self.model.get_cartesian_pos()

    def get_energies(self):
        return self.model.get_energies()

    def is_done(self):
        return self.model.get_time() >= self.t_final

    def restart(self):  # TODO: Add restart methods to this classes, do not recreate
        self.model = Pendulum(delta_t=self.delta_t)
        self.controller = PID(delta_t=self.delta_t, P=self.controller.P, D=self.controller.D, I=self.controller.I)
        self.generator = CPG(delta_t=self.delta_t, x_0=[0, -1])

        # display.clear_output(wait=True)
        plt.clf()

    def plot(self):
        try:
            plt.figure(1)

            # Model
            t_traj_model = self.model.get_temporal_traj()
            x_traj_model = self.model.get_state_traj()
            q_traj_model = x_traj_model[:, 0] % (2*np.pi)
            q_d_traj = self.controller.get_desired_traj() % (2*np.pi)
            plt.subplot(2, 2, 1)
            plt.plot(t_traj_model, q_traj_model * 180 / np.pi, 'b--', linewidth=1)
            plt.plot(t_traj_model, q_d_traj * 180 / np.pi, 'g--', linewidth=1)
            plt.ylabel(r'$Angle (rad)$')
            plt.xlabel('Time (s)')
            plt.legend(['Sys. traj.', 'Des. traj.'], loc='best')

            ax = plt.subplot(2, 2, 2)
            ax.clear()
            # px_model = self.model.l * np.sin(q_traj_model)
            # py_model = -self.model.l * np.cos(q_traj_model)
            [px_model, py_model] = self.model.get_cartesian_pos()
            plt.plot([0, px_model], [0, py_model], 'k*-', linewidth=2)
            plt.ylabel(r'$Vert. Pos. (m)$')
            plt.xlabel('Hor. Pos. (m)')
            plt.legend(['Pendulum'], loc='best')
            plt.axis([-1.1, 1.1, -1.1, 1.1])

            # CPG
            x_traj_CPG = self.generator.get_state_traj()
            # px_traj = x_traj_CPG[:, 0]
            # py_traj = x_traj_CPG[:, 1]
            [px_traj, py_traj] = self.generator.get_cartesian_pos()
            plt.subplot(2, 2, 3)
            plt.plot(px_traj, py_traj, 'k.-', linewidth=1)
            plt.ylabel(r'$Vert. Pos. (m)$')
            plt.xlabel('Hor. Pos. (m)')
            plt.legend(['CPG'], loc='best')
            plt.axis([-1.1, 1.1, -1.1, 1.1])

            # PID controller
            tau_traj = self.controller.get_force_traj()
            e_traj = self.controller.get_error_traj()
            e_P = e_traj[:, 0]
            e_I = e_traj[:, 1]
            e_D = e_traj[:, 2]
            plt.subplot(2, 2, 4)
            plt.plot(t_traj_model, tau_traj, 'r--', linewidth=1)
            plt.plot(t_traj_model, e_P, 'b--', linewidth=1)
            plt.plot(t_traj_model, e_I, 'k--', linewidth=1)
            plt.plot(t_traj_model, e_D, 'g--', linewidth=1)
            plt.ylabel(r'Force (Nm)$')
            plt.xlabel('Time (s)')
            plt.legend(['Cont. Out', '$e_P$', '$e_I$', '$e_D$'], loc='best')

            # RL

            # Show and wait
            plt.draw()
            plt.pause(0.0001)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
