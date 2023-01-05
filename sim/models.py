from sim.CONSTANTS import *

import numpy as np
from scipy.integrate import odeint

# TODO: Look into abstract classes, see how to implement methods with variable params

class CPG:
    def __init__(self, t_0: float = 0, x_0: list = None, delta_t: float = MIN_TIMESTEP):
        self.t_0 = t_0
        self.delta_t = delta_t
        self.t_cur = t_0
        if x_0 is None:
            self.x_0 = np.asarray([0, 0])
        else:
            self.x_0 = x_0
        self.x_cur = self.x_0

        self.x_traj = np.asarray([self.x_cur])
        self.t_traj = self.t_cur
        self.coils = 0

    def restart(self, t_0: float = None, x_0: list = None):
        if t_0 is not None:
            self.t_0 = t_0

        if x_0 is not None:
            self.x_0 = np.asarray(x_0)

        self.t_cur = self.t_0
        self.x_cur = self.x_0

        self.x_traj = np.asarray([self.x_cur])
        self.t_traj = self.t_cur
        self.coils = 0

    def eqs_motion(self, x, t, omega, mu):
        rho = x[0] ** 2 + x[1] ** 2
        circleDist = mu ** 2 - rho

        dx1 = -x[1] * omega + x[0] * circleDist
        dx2 = x[0] * omega + x[1] * circleDist

        return [dx1, dx2]

    def get_cartesian_pos(self):
        return self.x_cur

    def get_state_traj(self):
        return self.x_traj

    def get_temporal_traj(self):
        return self.t_traj

    def step(self, omega, mu):
        ts = [self.t_cur, self.t_cur + self.delta_t]
        xs = odeint(self.eqs_motion, self.x_cur, ts, args=(omega, mu))

        # Detect coiling
        new_angle = np.arctan2(xs[-1][1], xs[-1][0])
        old_angle = np.arctan2(self.x_cur[1], self.x_cur[0])
        if (-np.pi / 2 > new_angle < 0) and (0 < old_angle > np.pi / 2):
            self.coils += 1
        elif (-np.pi / 2 > old_angle < 0) and (0 < new_angle > np.pi / 2):
            self.coils -= 1

        # Update vars
        self.x_cur = np.asarray(xs[-1])
        self.t_cur += self.delta_t
        self.x_traj = np.append(self.x_traj, [self.x_cur], axis=0)
        self.t_traj = np.append(self.t_traj, self.t_cur)


class Pendulum:
    def __init__(self, t_0: float = 0, q_0: float = 0, dq_0: float = 0,
                 delta_t: float = MIN_TIMESTEP, l: float = 1, m: float = 0.1):
        self.t_0 = t_0
        self.q_0 = q_0
        self.dq_0 = dq_0

        self.delta_t = delta_t
        self.l = l
        self.m = m

        self.x_cur = np.asarray([q_0, dq_0])
        self.t_cur = t_0

        self.x_traj = np.asarray([self.x_cur])
        self.t_traj = self.t_cur

    def restart(self, t_0: float = None, q_0: float = None, dq_0: float = None):
        if t_0 is not None:
            self.t_0 = t_0

        if q_0 is not None:
            self.q_0 = q_0

        if dq_0 is not None:
            self.dq_0 = dq_0

        self.x_cur = np.asarray([self.q_0, self.dq_0])
        self.t_cur = self.t_0

        self.x_traj = np.asarray([self.x_cur])
        self.t_traj = self.t_cur

    def eqs_motion(self, x, t, tau):
        x1 = x[0]
        x2 = x[1]

        dx1 = x2
        dx2 = g / self.l * np.sin(x1) + 1 / (self.m * self.l ** 2) * tau

        return [dx1, dx2]

    def get_cartesian_pos(self):
        return [self.l * np.sin(self.x_cur[0]), - self.l * np.cos(self.x_cur[0])]

    def get_config_pos(self):
        return self.x_cur[0]

    def get_config_speed(self):
        return self.x_cur[1]

    def inverse_kins(self, p, coils):
        res = np.arctan2(p[1], p[0])
        return res + np.pi / 2 + coils * 2 * np.pi

    def get_energies(self):
        return [1 / 2 * self.m * (self.l * self.x_cur[1]) ** 2,
                self.m * -g * self.l * (1 - np.cos(self.x_cur[0]))]

    def get_time(self) -> float:
        return self.t_cur

    def get_state_traj(self):
        return self.x_traj

    def get_temporal_traj(self):
        return self.t_traj

    def step(self, tau):
        ts = [self.t_cur, self.t_cur + self.delta_t]
        xs = odeint(self.eqs_motion, self.x_cur, ts, args=(tau,))

        # Update vars
        self.x_cur = np.asarray(xs[-1])
        self.t_cur += self.delta_t
        self.x_traj = np.append(self.x_traj, [self.x_cur], axis=0)
        self.t_traj = np.append(self.t_traj, self.t_cur)
