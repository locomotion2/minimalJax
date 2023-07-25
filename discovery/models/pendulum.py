from discovery.CONSTANTS import *

import numpy as np

from discovery.models.base_models import BaseModel

import discovery.discovery_utils as sutils

class Pendulum(BaseModel):
    def __init__(self, params: dict = None):
        if params.get('not_inherited', True):
            self.l = params.get('l', 1)
            self.m = params.get('m', 0.1)
            self.k_f = params.get('k_f', 0.0)

        self.rng = np.random.default_rng()
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']

            x1 = x[0]
            x2 = x[1]
            tau = controller(t, x1, x2)

            dx1 = np.asarray([x2])
            dx2 = g / np.linalg.norm(self.l) * np.cos(x1) - \
                  self.k_f * x2 + \
                  tau / (np.linalg.norm(self.m) * np.linalg.norm(self.l) ** 2)

            return np.asarray([dx1, dx2]).flatten()

        return eqs_motion

    def select_initial(self, params: dict = None):

        def inverse_kinetic(E: float = 0):
            return np.asarray([(1 / self.l) * np.sqrt(2 * E / self.m)])

        def inverse_potential(E: float = 0):
            return np.asarray([np.arcsin(- E / (self.m * self.l * g) - 1)])

        # Handle inputs
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0)

        # Choose energies based on mode
        alpha = self.rng.uniform(0, 1)
        beta = self.rng.uniform(0, 1)
        E_k = 0
        E_p = 0
        if mode == 'speed':
            E_k = E_d
            E_p = 0
        elif mode == 'position':
            E_k = 0
            E_p = E_d
        elif mode == 'random_des':
            E_k = alpha * E_d
            E_p = (1 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20  # TODO: Change into it's own constant
            E_k = alpha * E_rand
            E_p = (1 - alpha) * E_rand

        # Calculate starting positions
        q_0 = inverse_potential(E_p)
        dq_0 = inverse_kinetic(E_k)

        # Initialize tracking arrays
        self.x_0 = np.asarray([q_0, dq_0]).flatten()
        self.x_cur = self.x_0

        self.q_0 = q_0
        self.q_cur = self.q_0

        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = self.p_0

        return q_0

    def solve(self, t: float):  # Todo: This is not up to date
        [q0, _] = self.x_0
        omega_star = np.sqrt(np.abs(g) / self.l)
        q = q0 * np.cos(omega_star * t)
        dq = - omega_star * q0 * np.sin(omega_star * t)

        return np.asarray([q, dq])

    def inverse_kins(self, params: dict = None):  # Todo: this is not up to date
        # Load params
        p = params['pos']
        v = params['speed']
        coils = params['coils']

        # Calculate angular position
        theta = np.arctan2(p[1], p[0])  # Finds angle in range [-pi, pi]
        theta_corrected = theta + np.pi / 2 + coils * 2 * np.pi  # This offsets the origin and accounts for the coiling of the CPG

        # Calculate angular speed
        circular_tangential = np.asarray(
            [np.cos(theta_corrected), np.sin(theta_corrected)])
        v_proj = sutils.project(v, circular_tangential)  # This finds the speed in the
        # pendulum circle
        omega_abs = np.sqrt(sum(v_proj ** 2)) / self.l
        omega_sign = np.sign(v_proj[1] / np.sin(theta_corrected))
        omega = omega_sign * omega_abs

        return np.asarray([theta_corrected, omega]).flatten()

    def forward_kins(self,
                     params: dict = None):  # TODO: Expand this in the future to calculate the speeds as well
        q = params['joints']
        p = np.asarray([self.l * np.cos(q), self.l * np.sin(q)])
        return p.flatten()

    def get_joint_state(self):
        q = np.asarray(self.x_cur[0:self.num_dof]).flatten()
        dq = np.asarray(self.x_cur[self.num_dof:]).flatten()
        return [q, dq]

    def get_cartesian_state(self):
        q = self.x_cur[0]
        dq = self.x_cur[1]

        # Todo: This needs to be replaced by forward kins
        p = np.asarray([self.l * np.cos(q), self.l * np.sin(q)]).flatten()
        v = np.asarray([- self.l * dq * np.sin(q), self.l * dq * np.cos(q)]).flatten()
        return [p, v]

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = np.asarray(self.x_cur[0:self.num_dof])

        # Calculate the link positions
        p = self.forward_kins({'joints': q})

        return p

    def get_energies(self):
        return np.asarray([1 / 2 * self.m * (self.l * self.x_cur[1]) ** 2,
                           - self.m * g * self.l * (1 + np.sin(self.x_cur[0]))])
