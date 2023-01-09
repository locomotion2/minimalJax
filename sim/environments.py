from sim.CONSTANTS import *

from sim.controllers import PID, PIDvel
from sim.models import CPG, Pendulum

from IPython import display
import matplotlib.pyplot as plt

import numpy as np


class BaseEnvironment:
    def __init__(self, delta_t: float = MIN_TIMESTEP, t_final: float = FINAL_TIME):
        self.delta_t = delta_t
        self.t_final = t_final

        self.model = Pendulum(delta_t=self.delta_t)
        self.controller = PID(delta_t=self.delta_t, gains=[15, 0.1, -0.75])
        # self.controller = PIDvel(delta_t=self.delta_t, gains=[15, 0.1, 0.3])
        self.generator = CPG(delta_t=self.delta_t, x_0=[0, -1])

    def step(self, action: np.ndarray):
        # Get RL params
        omega = action[0]
        mu = action[1]

        # Generate next point in path
        self.generator.step(omega, mu)
        p = self.generator.get_cartesian_pos()

        # Run through inverse kins and get the current state from model
        q_d = self.model.inverse_kins(p, self.generator.coils)
        q_cur = self.model.get_config_pos()
        dq_cur = self.model.get_config_speed()

        # Run this through controller, get force
        tau = self.controller.input(q_d, q_cur, dq_cur)

        # Apply controller action and update model
        self.model.step(tau)

    def get_state_model(self):
        return [self.model.get_cartesian_pos(), self.model.get_cartesian_vel()]

    def get_state_generator(self):
        return self.generator.get_cartesian_pos()

    def get_energies(self):
        return self.model.get_energies()

    def is_done(self):
        return self.model.get_time() >= self.t_final

    def restart(self):
        self.model.restart()
        self.controller.restart()
        self.generator.restart()

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
            plt.plot(t_traj, q_traj_model * 180 / np.pi, 'b--', linewidth=1)
            plt.plot(t_traj, q_d_traj * 180 / np.pi, 'g--', linewidth=1)
            plt.ylabel(r'$Angle (rad)$')
            plt.xlabel('Time (s)')
            plt.legend(['Sys. traj.', 'Des. traj.'], loc='best')
            plt.axis([0, self.t_final, 180, -180])

            # Pendulum simulation
            ax = plt.subplot(2, 2, 2)
            ax.clear()
            [px_model, py_model] = self.model.get_cartesian_pos()
            plt.plot([0, px_model], [0, py_model], 'k*-', linewidth=2)
            plt.ylabel(r'$Vert. Pos. (m)$')
            plt.xlabel('Hor. Pos. (m)')
            # plt.legend(['Pendulum'], loc='best')
            plt.axis([-1.1, 1.1, -1.1, 1.1])

            # CPG
            # [px_traj, py_traj] = self.generator.get_cartesian_pos()
            x_traj_CPG = self.generator.get_state_traj()
            px_traj = x_traj_CPG[:, 0]
            py_traj = x_traj_CPG[:, 1]
            ax = plt.subplot(2, 2, 3)
            ax.clear()
            plt.plot(px_traj, py_traj, 'k*-', linewidth=1)
            plt.ylabel(r'$Vert. Pos. (m)$')
            plt.xlabel('Hor. Pos. (m)')
            # plt.legend(['CPG'], loc='best')
            plt.axis([-2.1, 2.1, -2.1, 2.1])

            # PID controller
            tau_traj = self.controller.get_force_traj()
            e_traj = self.controller.get_error_traj()
            e_P = e_traj[:, 0]
            e_I = e_traj[:, 1]
            e_D = e_traj[:, 2]
            ax = plt.subplot(2, 2, 4)
            ax.clear()
            plt.plot(t_traj, tau_traj, 'y--', linewidth=2)
            plt.plot(t_traj, e_P, 'b--', linewidth=1)
            plt.plot(t_traj, e_I, 'k--', linewidth=1)
            plt.plot(t_traj, e_D, 'g--', linewidth=1)
            plt.ylabel(r'$Force (Nm)$')
            plt.xlabel('Time (s)')
            plt.legend(['Cont. Out', '$e_P$', '$e_I$', '$e_D$'], loc='best')
            ax.set_xlim([0, self.t_final])

            # RL

            # Show and wait
            plt.draw()
            plt.pause(0.00001)
        except KeyboardInterrupt:
            plt.close(fig=plt.figure('System'))
