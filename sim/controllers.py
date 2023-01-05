from sim.CONSTANTS import *

import numpy as np


class ConstantOutput:
    def __init__(self, value: float = 0):
        self.value = value

    def input(self, q_d: float, q_cur: float):
        return self.value


class PID:
    def __init__(self, delta_t: float = MIN_TIMESTEP, gains: list = None):
        self.delta_t = delta_t
        if gains is None:
            self.gains = [1, 0, 0]
        else:
            self.gains = gains

        self.q_d_prev = 0
        self.e_P_accum = 0

        self.tau_traj = np.asarray([0])
        self.q_d_traj = np.asarray([0])
        self.e_traj = np.asarray([np.asarray([0, 0, 0])])

    def restart(self):
        self.q_d_prev = 0
        self.e_P_accum = 0

        self.tau_traj = np.asarray([0])
        self.q_d_traj = np.asarray([0])
        self.e_traj = np.asarray([np.asarray([0, 0, 0])])

    def get_force_traj(self):
        return self.tau_traj

    def get_desired_traj(self):
        return self.q_d_traj

    def get_error_traj(self):
        return self.e_traj

    def input(self, q_d: float, q_cur: float, dq_cur: float):
        # Calc. P error
        e_P = q_d - q_cur

        # Calc. I error
        self.e_P_accum += e_P
        e_I = self.e_P_accum

        # Calc. D error
        dq_d = (self.q_d_prev - q_d) / self.delta_t
        self.q_d_prev = q_d
        e_D = dq_cur

        # Calc. Force
        [P, I, D] = self.gains
        e = np.asarray([P * e_P, I * e_I, D * e_D])
        tau = e.sum()

        # Track in vectors
        self.e_traj = np.append(self.e_traj, [e], axis=0)
        self.tau_traj = np.append(self.tau_traj, tau)
        self.q_d_traj = np.append(self.q_d_traj, q_d)

        return tau
