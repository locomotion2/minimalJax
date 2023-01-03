from CONSTANTS import *
from controllers import PID
from models import CPG, Pendulum

import numpy as np

class BaseEnvironment:
    def __init__(self, delta_t: float = 0.001, t_final: float = 10):
        self.delta_t = delta_t
        self.t_final = t_final

        self.model = Pendulum(delta_t=delta_t)
        self.controller = PID(delta_t=delta_t)
        self.generator = CPG(delta_t=delta_t, x_0=[0, -1])

    def step(self, action: np.ndarray):
        # Get params
        omega = action[0]
        mu = action[1]

        # Generate next point in traj
        self.generator.step(omega, mu)
        p_traj = self.generator.get_cartesian_pos()

        # Run through inverse kins and through controller
        q_d = self.model.inverse_kins(p_traj)
        q_cur = self.model.get_config_pos()
        tau = self.controller.input(q_d, q_cur)

        # Apply controller action and update model
        self.model.step(tau)

    def get_positions(self):
        return self.model.get_cartesian_pos()

    def get_energies(self):
        return self.model.get_energies()

    def is_done(self):
        return self.model.get_time() >= self.t_final

    def restart(self):
        self.model = Pendulum(delta_t=self.delta_t)
        self.controller = PID(delta_t=self.delta_t)
        self.generator = CPG(delta_t=self.delta_t, x_0=[0, -1])
