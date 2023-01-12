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

    @abstractmethod
    def input(self, inputs: dict = None):
        pass

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
        gains = params['gains']
        if gains is None:
            self.gains = [1, 0, 0]
        else:
            self.gains = gains

        self.q_d_prev = 0
        self.e_P_accum = 0

        # Storage and help variables
        self.e_cur = 0
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

    @abstractmethod
    def input(self, inputs: dict = None):
        pass

    def update_trajectories(self, q_d, tau):
        # Tracking vectors
        self.e_traj = np.append(self.e_traj, [self.e_cur], axis=0)
        self.tau_traj = np.append(self.tau_traj, tau)
        self.q_d_traj = np.append(self.q_d_traj, q_d)


class PID_pos_vel_damping(basePIDController):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def input(self, inputs: dict = None):
        q_d = inputs['q_d']
        q_cur = inputs['q_cur']
        dq_cur = inputs['dq_cur']

        # Calc. P error
        e_P = q_d - q_cur

        # Calc. I error
        self.e_P_accum += e_P
        e_I = self.e_P_accum

        # Calc. D error
        # dq_d = (self.q_d_prev - q_d) / self.delta_t
        self.q_d_prev = q_d
        e_D = dq_cur

        # Calc. Force
        [P, I, D] = self.gains
        self.e_cur = np.asarray([P * e_P, I * e_I, D * e_D])
        tau = self.e_cur.sum()

        return tau


class PID_pos_vel_tracking_num(basePIDController):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def input(self, inputs: dict = None):
        q_d = inputs['q_d']
        q_cur = inputs['q_cur']
        dq_cur = inputs['dq_cur']

        # Calc. P error
        e_P = q_d - q_cur

        # Calc. I error
        self.e_P_accum += e_P
        e_I = self.e_P_accum

        # Calc. D error
        dq_d = (q_d - self.q_d_prev) / self.delta_t
        self.q_d_prev = q_d
        e_D = dq_d - dq_cur

        # Calc. Force
        [P, I, D] = self.gains
        self.e_cur = np.asarray([P * e_P, I * e_I, D * e_D])
        tau = self.e_cur.sum()

        return tau

class PID_pos_vel_tracking_modeled(basePIDController):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def input(self, inputs: dict = None):
        q_d = inputs['q_d']
        dq_d = inputs['dq_d']
        q_cur = inputs['q_cur']
        dq_cur = inputs['dq_cur']

        # Calc. P error
        e_P = q_d - q_cur

        # Calc. I error
        self.e_P_accum += e_P
        e_I = self.e_P_accum

        # Calc. D error
        # dq_d = (q_d - self.q_d_prev) / self.delta_t
        self.q_d_prev = q_d
        e_D = dq_d - dq_cur

        # Calc. Force
        [P, I, D] = self.gains
        self.e_cur = np.asarray([P * e_P, I * e_I, D * e_D])
        tau = self.e_cur.sum()

        return tau