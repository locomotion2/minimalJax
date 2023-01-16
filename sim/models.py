from sim.CONSTANTS import *

import numpy as np
from scipy.integrate import odeint
from abc import ABC, abstractmethod


# TODO: Look into abstract classes, see how to implement methods with variable params

def project(u: np.ndarray, v: np.ndarray):
    v_norm = np.sqrt(sum(v ** 2))
    return (np.dot(u, v) / v_norm ** 2) * v


class baseModel(ABC):
    def __init__(self, params: dict = None):
        # Load params
        self.t_0 = default(params, 't_0', 0)
        self.delta_t = default(params, 'delta_t', MIN_TIMESTEP)
        self.state_size = params['state_size']
        self.num_dof = params['num_dof']

        # Initialize variables
        self.x_0 = [0] * self.state_size
        self.x_cur = np.asarray(self.x_0)
        self.x_traj = np.asarray([self.x_cur])

        # TODO: This is currently unused
        self.q_0 = np.asarray([0] * self.num_dof)
        self.q_cur = np.asarray(self.q_0)
        self.q_traj = np.asarray([self.q_cur])

        self.t_cur = self.t_0
        self.t_traj = self.t_cur

    @abstractmethod
    def select_initial(self, params: dict = None):
        pass

    def restart(self, params: dict = None):
        # Handle inputs
        self.t_0 = default(params, 't_0', self.t_0)

        # Set up initial state conditions
        self.select_initial(params)

        # Set up time and tracking
        self.t_cur = self.t_0
        self.t_traj = self.t_cur

    @abstractmethod
    def eqs_motion(self, x, t, u):
        pass

    @abstractmethod
    def inverse_kins(self, params: dict = None):
        pass

    def step(self, params: dict = None):
        ts = [self.t_cur, self.t_cur + self.delta_t]
        xs = odeint(self.eqs_motion, self.x_cur, ts, args=(params,))  # TODO: Check how fast this is

        # Update vars
        self.x_cur = np.asarray(xs[-1])
        self.t_cur += self.delta_t

    def update_trajectories(self):
        self.x_traj = np.append(self.x_traj, [self.x_cur], axis=0)
        self.t_traj = np.append(self.t_traj, self.t_cur)

    @abstractmethod
    def solve(self, t):
        pass

    @abstractmethod
    def get_cartesian_state(self):
        pass

    @abstractmethod
    def get_joint_state(self):
        pass

    def get_time(self):
        return self.t_cur

    @abstractmethod
    def get_energies(self):
        pass

    def get_state_traj(self):
        return self.x_traj

    def get_temporal_traj(self):
        return self.t_traj


class CPG(baseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.omega_cur = 0
        self.mu_cur = 0
        self.coils = 0

    def restart(self, params: dict = None):
        super().restart(params)
        self.coils = 0

    def eqs_motion(self, x, t, params):
        self.mu_cur = params['mu']
        self.omega_cur = params['omega']

        rho = x[0] ** 2 + x[1] ** 2
        circleDist = self.mu_cur ** 2 - rho

        dx1 = -x[1] * self.omega_cur + x[0] * circleDist
        dx2 = x[0] * self.omega_cur + x[1] * circleDist

        return [dx1, dx2]

    def get_cartesian_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}

        p = self.x_cur
        v = self.eqs_motion(self.x_cur, 0, params)

        return [p, v]

    def get_state_traj(self):
        return self.x_traj

    def get_temporal_traj(self):
        return self.t_traj

    def detect_coiling(self):
        x_new = self.x_cur
        x_old = self.x_traj[-2]

        new_angle = np.arctan2(x_new[1], x_new[0])
        old_angle = np.arctan2(x_old[1], x_old[0])
        if (-np.pi / 2 > new_angle < 0) and (0 < old_angle > np.pi / 2):
            self.coils += 1
        elif (-np.pi / 2 > old_angle < 0) and (0 < new_angle > np.pi / 2):
            self.coils -= 1

        return self.coils

    def select_initial(self, params: dict = None):
        self.x_0 = default(params, 'x_0', self.x_0)
        self.x_cur = np.asarray(self.x_0)
        self.x_traj = np.asarray([self.x_cur])

        return self.x_0

    def solve(self, t):
        pass

    def inverse_kins(self, params: dict = None):
        pass

    def get_joint_state(self):
        pass

    def get_energies(self):
        pass


class Pendulum(baseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.l = default(params, 'l', 1)
        self.m = default(params, 'm', 0.1)

    def select_initial(self, params: dict = None):
        # Handle inputs
        mode = default(params, 'mode', 'equilibrium')
        E_d = default(params, 'E_d', 0)

        # Initialize initial conditions
        q_0 = 0
        dq_0 = 0

        # Choose based on case
        if mode == 'speed':
            dq_0 = (1 / self.l) * np.sqrt(2 * E_d / self.m)
        elif mode == 'position':
            q_0 = np.arccos(1 + E_d / (self.m * self.l * g))

        # Initialize tracking arrays
        self.x_0 = [q_0, dq_0]
        self.x_cur = np.asarray(self.x_0)
        self.x_traj = np.asarray([self.x_cur])

        return self.x_0

    def eqs_motion(self, x, t, params):
        tau = params['tau']

        x1 = x[0]
        x2 = x[1]

        dx1 = x2
        dx2 = g / self.l * np.sin(x1) + 1 / (self.m * self.l ** 2) * tau

        return [dx1, dx2]

    def solve(self, t: float):
        [q0, dq0] = self.x_0
        omega_star = np.sqrt(np.abs(g) / self.l)
        q = q0 * np.cos(omega_star * t)
        dq = - omega_star * q0 * np.sin(omega_star * t)

        return [q, dq]

    def inverse_kins(self, params: dict = None):
        # Load params
        p = params['pos']
        v = params['speed']
        coils = params['coils']

        # Calculate angular position
        theta = np.arctan2(p[1], p[0])  # Finds angle in range [-pi, pi]
        theta_corrected = theta + np.pi / 2 + coils * 2 * np.pi  # This offsets the origin and accounts for the coiling of the CPG

        # Calculate angular speed
        circular_tangential = np.asarray([np.cos(theta_corrected), np.sin(theta_corrected)])
        v_proj = project(v, circular_tangential)  # This finds the speed in the pendulum circle
        omega_abs = np.sqrt(sum(v_proj ** 2)) / self.l
        omega_sign = np.sign(v_proj[1] / np.sin(theta_corrected))
        omega = omega_sign * omega_abs

        return [theta_corrected, omega]

    def get_joint_state(self):
        return self.x_cur

    def get_cartesian_state(self):
        q = self.x_cur[0]
        dq = self.x_cur[1]

        p = [self.l * np.sin(q), - self.l * np.cos(q)]
        v = [self.l * dq * np.cos(q), self.l * dq * np.sin(q)]

        return [p, v]

    def get_energies(self):
        return [1 / 2 * self.m * (self.l * self.x_cur[1]) ** 2,
                self.m * -g * self.l * (1 - np.cos(self.x_cur[0]))]
