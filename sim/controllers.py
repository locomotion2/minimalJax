from sim.CONSTANTS import *

import numpy as np
from abc import ABC, abstractmethod


class BaseController(ABC):
    def __init__(self, params: dict = None):
        delta_t = params['delta_t']
        if delta_t is None:
            self.delta_t = MIN_TIMESTEP
        else:
            self.delta_t = delta_t

        self.n_dof = params.get('num_dof', 2)

    @abstractmethod
    def restart(self):
        pass


class ConstantOutput(BaseController):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.value = params['value']

    def input(self, inputs: dict = None):
        return self.value

    def restart(self):
        pass


class basePIDController(BaseController):
    def __init__(self, params: dict = None):
        super().__init__(params)
        gains_outer = params['gains_outer']
        gains_eigen = params['gains_eigen']
        gains_mode = params['mode']
        if gains_outer is None:
            self.gains_outer = np.asarray([1, 0, 0])
            self.gains_eigen = np.asarray([1, 0, 0])
        else:
            self.gains_outer = np.asarray(gains_outer)
            self.gains_eigen = np.asarray(gains_eigen)
        if gains_mode is None:
            self.mode = True
        elif gains_mode == 'maximal':
            self.mode = True
        elif gains_mode == 'minimal':
            self.mode = False
        else:
            raise NotImplementedError
        self.gains_cur = self.gains_eigen

        # Target variables
        self.q_d = [0] * self.n_dof
        self.dq_d = [0] * self.n_dof

        # D and I variables
        self.q_d_prev = [0] * self.n_dof
        self.e_P_accum = [0] * self.n_dof

        # Storage and help variables
        self.e_cur = [0] * self.n_dof
        self.tau_last = [0] * self.n_dof
        self.tau_traj = np.asarray([0] * self.n_dof)
        self.q_d_traj = np.asarray([np.asarray([0] * self.n_dof)])
        self.e_traj = np.asarray([np.asarray([0, 0, 0])])

    def restart(self):
        # Target variables
        self.q_d = [0] * self.n_dof
        self.dq_d = [0] * self.n_dof

        # D and I variables
        self.q_d_prev = [0] * self.n_dof
        self.e_P_accum = [0] * self.n_dof

        # Storage and help variables
        self.e_cur = [0] * self.n_dof
        self.e_traj = np.asarray([np.asarray([0, 0, 0])])
        self.tau_traj = np.asarray([np.asarray([0])])
        self.q_d_traj = np.asarray([np.asarray(self.q_d)])

    def get_force(self):
        return self.tau_last

    def get_force_traj(self):
        return self.tau_traj.flatten()

    def get_desired_traj(self):
        return self.q_d_traj

    def get_error_traj(self):
        return self.e_traj

    @abstractmethod
    def input(self, t: float, q: float, dq: float):
        pass

    def set_target(self, q_d: float, dq_d: float, params: dict = None):
        self.q_d = q_d
        self.dq_d = dq_d

        if not params.get('inference', False):
            cost_energy = gaus(params.get('E') - params.get('E_d'), 0.05)
            if self.mode:
                self.gains_cur = np.maximum(cost_energy * self.gains_eigen, self.gains_outer)
            else:
                self.gains_cur = np.minimum(1/np.maximum(0.001, cost_energy) * self.gains_eigen, self.gains_outer)

    def update_trajectories(self, q_d):
        # Tracking vectors
        error = self.e_cur
        error = np.linalg.norm(error, axis=1)
        tau = self.tau_last
        tau = np.linalg.norm(tau)
        self.e_traj = np.append(self.e_traj, [error], axis=0)  # TODO: This needs to change for the larger dims
        self.tau_traj = np.append(self.tau_traj, [np.asarray([tau])], axis=0)
        self.q_d_traj = np.append(self.q_d_traj, [q_d], axis=0)


class PID_pos_vel_tracking_modeled(basePIDController):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def input(self, t, q, dq):
        # Handle inputs
        q_cur = q
        dq_cur = dq

        # Calc. P error
        e_P = self.q_d - q_cur

        # Calc. I error
        self.e_P_accum += e_P
        e_I = self.e_P_accum

        # Calc. D error
        self.q_d_prev = self.q_d
        e_D = self.dq_d - dq_cur
        e_D = e_D.flatten()

        # Calc. Force
        [P, I, D] = self.gains_cur
        self.e_cur = np.asarray([P * e_P, I * e_I, D * e_D])
        # self.e_cur = np.asarray([e_P, e_I, e_D]) @ self.gains
        tau = self.e_cur.sum(axis=0).flatten()  # TODO: This can be an issue
        self.tau_last = tau  # Update the variable to eventually get last

        return tau
