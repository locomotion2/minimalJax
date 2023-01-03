from CONSTANTS import *

import numpy as np
from scipy.integrate import odeint


class CPG:
    def __init__(self, t_0: float = 0, delta_t: float = 0.001, x_0=None):
        self.delta_t = delta_t
        self.t_cur = t_0
        if x_0 is None:
            self.x_cur = [0, 0]
        else:
            self.x_cur = x_0

    def eqs_motion(self, x, t, omega, mu):
        rho = x[0] ** 2 + x[1] ** 2
        circleDist = mu ** 2 - rho

        dx1 = -x[1] * omega + x[0] * circleDist
        dx2 = x[0] * omega + x[1] * circleDist

        return [dx1, dx2]

    def get_cartesian_pos(self):
        return self.x_cur

    def step(self, omega, mu):
        ts = [self.t_cur, self.t_cur + self.delta_t]
        xs = odeint(self.eqs_motion, self.x_cur, ts, args=[omega, mu])

        # Update vars
        self.x_cur = xs[-1]
        self.t_cur += self.delta_t


class Pendulum:
    def __init__(self, t_0: float = 0, delta_t: float = 0.001, q_0: float = 0, dq_0: float = 0,
                 l: float = 1, m: float = 0.1):
        self.delta_t = delta_t
        self.l = l
        self.m = m
        self.x_cur = [q_0, dq_0]
        self.t_cur = t_0

    def eqs_motion(self, x, t, tau):
        x1 = x[0]
        x2 = x[1]

        dx1 = x2
        dx2 = g / self.l * np.sin(x1) + 1 / (self.m * self.l ** 2) * tau

        return [dx1, dx2]

    def get_cartesian_pos(self):
        return [self.l * np.cos(self.x_cur[0]), self.l * np.sin(self.x_cur[0])]

    def get_config_pos(self):
        return self.x_cur[0]

    def inverse_kins(self, p):
        return np.arctan2(p[0], p[1])

    def get_energies(self):
        return [1/2 * self.m * (self.l * self.x_cur[1])**2,
                self.m * g * self.l * (1 - np.cos(self.x_cur[0]))]

    def get_time(self) -> float:
        return self.t_cur

    def config(self):
        pass

    def step(self, tau):
        ts = [self.t_cur, self.t_cur + self.delta_t]
        xs = odeint(self.eqs_motion, self.x_cur, ts, args=[tau])

        # Update vars
        self.x_cur = xs[-1]
        self.t_cur += self.delta_t
