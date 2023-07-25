from discovery.CONSTANTS import *

import numpy as np

from scipy.integrate import odeint, solve_ivp

import discovery.discovery_utils as sutils
from discovery.models.pendulum import Pendulum


class ArnesDoublePendulum:
    def __init__(self, l, m, g, k, qr, kf):
        self.dof = 2
        self._kf = kf
        self._l = l
        self._m = m
        self._g = g
        self._k = k
        self._q_rest = qr
        self._p = l, m, g, k, qr, kf
        self._has_inverse_dynamics = True
        self._eq = None
        self._U0 = None

    @property
    def params(self):
        return self._p

    @property
    def equilibrium(self):
        if self._eq is None:
            self._eq = self.compute_equilibrium()
        return self._eq

    @property
    def U0(self):
        if self._U0 is None:
            self._U0 = self._potential(self.equilibrium)
        return self._U0

    def sim(self, q0, dq0,
            t_max,
            controllers=None,
            dt=None,
            dense=True,
            return_sol=False,
            verbose=False,
            wrap=False,
            **kwargs
            ):

        sol = solve_ivp(
            fun=self.create_dynamics(controllers),
            t_span=(0, t_max),
            y0=np.r_[q0, dq0],
            method='LSODA',
            dense_output=dense,
            t_eval=np.arange(0, t_max, dt) if dt and dense else None,
            max_step=dt if dt else np.inf,
            **kwargs,
        )

        if verbose:
            print(f"Solve IVP finished with '{sol.message}'.")
            print(
                f"In total {len(sol.t)} time points were evaluated and the rhs was evaluated {sol.nfev} times.")

        traj = sol.y.T
        n = self.dof
        q = np.arctan2(np.sin(traj[:, 0:n]), np.cos(traj[:, 0:n])) if wrap else traj[:,
                                                                                0:n]
        if return_sol:
            return sol.t, q, traj[:, n:], sol
        else:
            return sol.t, q, traj[:, n:]

    def compute_equilibrium(self, q0=None):
        if q0 is None:
            q0 = np.zeros(self.dof)

        def viscous_damping(t, q, dq):
            return -2 * self.mass_matrix(q) @ dq

        _, q, _, sol = self.sim(
            q0=q0,
            dq0=np.zeros(self.dof),
            dt=1e-2,
            controllers=[viscous_damping],
            events=self.create_convergence_check(),
            t_max=50.0,
            return_sol=True,
        )

        if sol.status == 1:  # Termination event occured
            return q[-1, :]
        else:
            raise ValueError("No equilibrium found.")

    def create_convergence_check(self, eps=1e-3, terminal=True):
        n = self.dof

        def convergence_check(t, y):
            q, dq = y[0:n], y[n:]
            if t < 1:
                return np.inf
            else:
                return np.linalg.norm(dq) - eps

        convergence_check.terminal = terminal
        return convergence_check

    def mass_matrix(self, q):
        l, m, g, k, qr, kf = self.params
        return np.array([[l[0] ** 2 * m[0] + m[1] * (
                    l[0] ** 2 + 2 * l[0] * l[1] * np.cos(q[1]) + l[1] ** 2),
                          1.0 * l[1] * m[1] * (l[0] * np.cos(q[1]) + l[1])],
                         [1.0 * l[1] * m[1] * (l[0] * np.cos(q[1]) + l[1]),
                          1.0 * l[1] ** 2 * m[1]]])

    def gravity(self, q):
        l, m, g, k, qr, kf = self.params
        return np.array(
            [[g * (l[0] * m[0] * np.cos(q[0]) + m[1] * (
                        l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1])))],
             [g * l[1] * m[1] * np.cos(q[0] + q[1])]]).flatten()

    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr, kf = self.params
        return np.array([[-2.0 * dq[0] * dq[1] * l[0] * l[1] * m[1] * np.sin(
            q[1]) - 1.0 * dq[1] ** 2 * l[0] * l[
                              1] * m[1] * np.sin(q[1]) + 1.0 * k[0] * q[0] - 1.0 * k[
                              0] * qr[0]], [
                             1.0 * dq[0] ** 2 * l[0] * l[1] * m[1] * np.sin(
                                 q[1]) + 1.0 * k[1] * q[1] - 1.0 * k[
                                 1] * qr[1]]]).flatten()

    def kinetic_energy(self, q, dq):
        if q.ndim == dq.ndim == 1:
            return self._kinetic_energy(q, dq)
        elif q.ndim == dq.ndim == 2:
            return self._kinetic_energy(q.T, dq.T)
        else:
            raise ValueError

    def potential_energy(self, q, absolute=False):
        if q.ndim == 1:
            return self._potential(q) - (0 if absolute else self.U0)
        elif q.ndim == 2:
            return self._potential(q.T) - (0 if absolute else self.U0)
        else:
            raise ValueError

    def forward_kinematics(self, q):
        if q.ndim == 1:
            return self._fkin(q).flatten()  # Warning recently added .flatten() here
        elif q.ndim == 2:
            return self._fkin(q.T).T.squeeze()

    def forward_kinematics_for_each_link(self, q):
        if q.ndim == 1:
            return sutils.hom2xyphi(self._link_positions(q).reshape((-1, 3, 3)))
        elif q.ndim == 2:
            out = np.empty((q.shape[0], self.dof, 3))
            for i in range(q.shape[0]):
                out[i, :, :] = sutils.hom2xyphi(self._link_positions(q[i]).reshape((-1,
                                                                                    3,
                                                                                    3)))
            return out
        else:
            raise ValueError

    def bad_invkin(self, cart, q0=None, K=1.0, tol=1e-3, max_steps=100):
        if q0 is not None:
            q = q0
        else:
            q = np.random.uniform(-np.pi, np.pi, self.dof)

        if cart.size == 3:
            f = lambda x: self.forward_kinematics(x)
            if self.dof == 3:
                A = lambda x: np.linalg.inv(self.jacobian(x))
            else:
                A = lambda x: np.linalg.pinv(self.jacobian(x))
        elif cart.size == 2:
            f = lambda x: self.forward_kinematics(x)[0:2]
            A = lambda x: np.linalg.pinv(self.jacobian(x)[0:2, :])
        elif cart.size == 1:
            f = lambda x: self.forward_kinematics(x)[0:1]
            A = lambda x: np.linalg.pinv(self.jacobian(x)[0:1, :])
        else:
            raise ValueError("Illegal length of desired task-space pose")

        step = 0
        while True:
            if step > max_steps:
                print(f"No invkin solution found for {cart}, trying again")
                q = np.random.uniform(-np.pi, np.pi, self.dof)
                step = 0

            e = cart - f(q)
            if np.linalg.norm(e) < tol:
                break
            try:
                inc = A(q) @ np.squeeze(K * e)
            except ValueError:
                inc = (A(q) * np.squeeze(K * e))[:, 0]
            q += inc
            step += 1

        # for _ in range(max_steps):
        #     e = cart - f(q)
        #     if np.linalg.norm(e) < tol:
        #         break
        #     try:
        #         inc = A(q) @ np.squeeze(K * e)
        #     except ValueError:
        #         inc = (A(q) * np.squeeze(K * e))[:, 0]
        #     q += inc
        # else:
        #     print(f"No invkin solution found for {cart}, trying again")
        #     q = np.random.uniform(-np.pi, np.pi, self.dof)
        #     # raise AssertionError(f"No invkin solution found for {cart}.")

        return np.arctan2(np.sin(q), np.cos(q))

    def _potential(self, q):
        l, m, g, k, qr, kf = self.params
        return g * l[0] * m[0] * np.sin(q[0]) + g * m[1] * (
                l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1])) + 0.5 * k[0] * (
                    -q[0] + qr[0]) ** 2 + 0.5 * \
            k[1] * (-q[1] + qr[1]) ** 2

    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr, kf = self.params
        ddq_cons = np.array([[1.0 * (
                0.5 * dq[0] ** 2 * l[0] ** 2 * l[1] * m[1] * np.sin(2 * q[1]) + 1.0 *
                dq[0] ** 2 * l[0] * l[
                    1] ** 2 * m[1] * np.sin(q[1]) + 2.0 * dq[0] * dq[1] * l[0] * l[
                    1] ** 2 * m[1] * np.sin(
            q[1]) + 1.0 * dq[1] ** 2 * l[0] * l[1] ** 2 * m[1] * np.sin(
            q[1]) - 1.0 * g * l[0] * l[1] * m[
                    0] * np.cos(q[0]) - 0.5 * g * l[0] * l[1] * m[1] * np.cos(
            q[0]) + 0.5 * g * l[0] * l[1] *
                m[1] * np.cos(q[0] + 2 * q[1]) - 1.0 * k[0] * l[1] * q[0] + 1.0 * k[0] *
                l[1] * qr[0] + 1.0 * k[
                    1] * l[0] * q[1] * np.cos(q[1]) - 1.0 * k[1] * l[0] * qr[
                    1] * np.cos(q[1]) + 1.0 * k[1] *
                l[1] * q[1] - 1.0 * k[1] * l[1] * qr[1] - 1.0 * l[0] * tau_in[
                    1] * np.cos(q[1]) + 1.0 * l[1] *
                tau_in[0] - 1.0 * l[1] * tau_in[1]) / (l[0] ** 2 * l[1] * (
                    m[0] + m[1] * np.sin(q[1]) ** 2))], [
                                 2.0 * (-1.0 * dq[0] ** 2 * l[0] ** 3 * l[1] * m[0] * m[
                                     1] * np.sin(q[1]) - 1.0 * dq[
                                            0] ** 2 * l[0] ** 3 * l[1] * m[
                                            1] ** 2 * np.sin(q[1]) - 1.0 * dq[0] ** 2 *
                                        l[
                                            0] ** 2 * l[1] ** 2 * m[1] ** 2 * np.sin(
                                             2 * q[1]) - 1.0 * dq[0] ** 2 * l[
                                            0] * l[1] ** 3 * m[1] ** 2 * np.sin(
                                             q[1]) - 1.0 * dq[0] * dq[1] * l[
                                            0] ** 2 * l[1] ** 2 * m[1] ** 2 * np.sin(
                                             2 * q[1]) - 2.0 * dq[0] * dq[1] *
                                        l[0] * l[1] ** 3 * m[1] ** 2 * np.sin(
                                             q[1]) - 0.5 * dq[1] ** 2 * l[0] ** 2 *
                                        l[1] ** 2 * m[1] ** 2 * np.sin(2 * q[1]) - 1.0 *
                                        dq[1] ** 2 * l[0] * l[
                                            1] ** 3 * m[1] ** 2 * np.sin(
                                             q[1]) + 0.5 * g * l[0] ** 2 * l[1] * m[0] *
                                        m[1] * np.cos(q[0] - q[1]) - 0.5 * g * l[
                                            0] ** 2 * l[1] * m[0] * m[
                                            1] * np.cos(q[0] + q[1]) + 0.5 * g * l[
                                            0] ** 2 * l[1] * m[
                                            1] ** 2 * np.cos(q[0] - q[1]) - 0.5 * g * l[
                                            0] ** 2 * l[1] * m[
                                            1] ** 2 * np.cos(q[0] + q[1]) + 1.0 * g * l[
                                            0] * l[1] ** 2 * m[0] * m[
                                            1] * np.cos(q[0]) + 0.5 * g * l[0] * l[
                                            1] ** 2 * m[1] ** 2 * np.cos(
                                             q[0]) - 0.5 * g * l[0] * l[1] ** 2 * m[
                                            1] ** 2 * np.cos(
                                             q[0] + 2 * q[1]) + 1.0 * k[0] * l[0] * l[
                                            1] * m[1] * q[0] * np.cos(
                                             q[1]) - 1.0 * k[0] * l[0] * l[1] * m[1] *
                                        qr[0] * np.cos(q[1]) + 1.0 * k[
                                            0] * l[1] ** 2 * m[1] * q[0] - 1.0 * k[0] *
                                        l[1] ** 2 * m[1] * qr[0] - 1.0 *
                                        k[1] * l[0] ** 2 * m[0] * q[1] + 1.0 * k[1] * l[
                                            0] ** 2 * m[0] * qr[1] - 1.0 * k[
                                            1] * l[0] ** 2 * m[1] * q[1] + 1.0 * k[1] *
                                        l[0] ** 2 * m[1] * qr[1] - 2.0 *
                                        k[1] * l[0] * l[1] * m[1] * q[1] * np.cos(
                                             q[1]) + 2.0 * k[1] * l[0] * l[1] *
                                        m[1] * qr[1] * np.cos(q[1]) - 1.0 * k[1] * l[
                                            1] ** 2 * m[1] * q[1] + 1.0 * k[
                                            1] * l[1] ** 2 * m[1] * qr[1] + 1.0 * l[
                                            0] ** 2 * m[0] * tau_in[1] + 1.0 * l[
                                            0] ** 2 * m[1] * tau_in[1] - 1.0 * l[0] * l[
                                            1] * m[1] * tau_in[
                                            0] * np.cos(q[1]) + 2.0 * l[0] * l[1] * m[
                                            1] * tau_in[1] * np.cos(
                                             q[1]) - 1.0 * l[1] ** 2 * m[1] * tau_in[
                                            0] + 1.0 * l[1] ** 2 * m[1] *
                                        tau_in[1]) / (l[0] ** 2 * l[1] ** 2 * m[1] * (
                                         2 * m[0] - m[1] * np.cos(2 * q[1]) + m[1]))]])
        ddq = ddq_cons - kf * np.asarray([[dq[0]], [dq[1]]])
        return ddq

    def _kinetic_energy(self, q, dq):
        l, m, g, k, qr, kf = self.params
        return 0.5 * dq[0] ** 2 * l[0] ** 2 * m[0] + 0.5 * m[1] * (
                dq[0] ** 2 * l[0] ** 2 + 2 * dq[0] ** 2 * l[0] * l[1] * np.cos(q[1]) +
                dq[0] ** 2 * l[
                    1] ** 2 + 2 * dq[0] * dq[1] * l[0] * l[1] * np.cos(q[1]) + 2 * dq[
                    0] * dq[1] * l[1] ** 2 + dq[
                    1] ** 2 * l[1] ** 2)

    def _energy(self, q, dq):
        l, m, g, k, qr, kf = self.params
        return 0.5 * dq[0] ** 2 * l[0] ** 2 * m[0] + g * l[0] * m[0] * np.sin(
            q[0]) + g * m[1] * (
                l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1])) + 0.5 * k[0] * (
                    -q[0] + qr[0]) ** 2 + 0.5 * \
            k[1] * (-q[1] + qr[1]) ** 2 + 0.5 * m[1] * (
                    dq[0] ** 2 * l[0] ** 2 + 2 * dq[0] ** 2 * l[0] * l[1] * np.cos(
                q[1]) + dq[0] ** 2 * l[
                        1] ** 2 + 2 * dq[0] * dq[1] * l[0] * l[1] * np.cos(q[1]) + 2 *
                    dq[0] * dq[1] * l[1] ** 2 + dq[
                        1] ** 2 * l[1] ** 2)

    def _link_positions(self, q):
        l, m, g, k, qr, kf = self.params
        return np.array([[np.cos(q[0]), -np.sin(q[0]), l[0] * np.cos(q[0])],
                         [np.sin(q[0]), np.cos(q[0]), l[0] * np.sin(q[0])], [0, 0, 1],
                         [np.cos(q[0] + q[1]), -np.sin(q[0] + q[1]),
                          l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1])],
                         [np.sin(q[0] + q[1]), np.cos(q[0] + q[1]),
                          l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1])],
                         [0, 0, 1]]).reshape((2, 3, 3))

    def _fkin(self, q):
        l, m, g, k, qr, kf = self.params
        return np.array([[l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1])],
                         [l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1])],
                         [np.arctan2(np.sin(q[0] + q[1]), np.cos(q[0] + q[1]))]])

    def endeffector_pose(self, q):
        l, m, g, k, qr, kf = self.params
        return np.array(
            [[np.cos(q[0] + q[1]), -np.sin(q[0] + q[1]),
              l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1])],
             [np.sin(q[0] + q[1]), np.cos(q[0] + q[1]),
              l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1])],
             [0, 0, 1]])

    def jacobian(self, q):
        l, m, g, k, qr, kf = self.params
        return np.array([[-l[0] * np.sin(q[0]) - l[1] * np.sin(q[0] + q[1]),
                          -l[1] * np.sin(q[0] + q[1])],
                         [l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1]),
                          l[1] * np.cos(q[0] + q[1])],
                         [1, 1]])

    def create_dynamics(self, controllers=None):
        n = self.dof
        if controllers is None:
            controllers = []

        def ode_precomputed(t, y):
            q, dq = y[0:n], y[n:]
            tau = sum((c(t, q, dq) for c in controllers), np.zeros(self.dof))
            ddq = self._ddq(q, dq, tau).flatten()
            return np.r_[dq, ddq]

        return ode_precomputed


class DoublePendulum(Pendulum):
    def __init__(self, params: dict = None):
        self.l = params.get('l', (0.5, 0.5))
        self.m = params.get('m', (0.05, 0.05))
        self.k_f = params.get('k_f', 0.0)
        self.transition = np.asarray([0, 0])
        self.q_des_prev = None
        self.backend = ArnesDoublePendulum(
            l=self.l,
            m=self.m,
            g=-g,
            k=(0, 0),
            qr=(0, 0),
            kf=self.k_f
        )
        params['not_inherited'] = False
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']
            eqs_backend = self.backend.create_dynamics(
                controllers=[controller])  # get eqs of motion
            return eqs_backend(t, x)

        return eqs_motion

    def select_initial(self, params: dict = None):

        def inverse_kinetic(q, E: float = 0):
            # Randomize vector direction
            dq0 = self.rng.uniform(-1, 1, 2)

            # Scale up so that it produces the desired energy
            E_norm = self.backend.kinetic_energy(q, dq0)
            dq0 = dq0 / E_norm * E

            def dEk(q, dq):
                l = self.l
                m = self.m
                dE = np.asarray([[m[0] * dq[0] * l[0] ** 2 + m[1] * dq[0] * l[0] ** 2
                                  + m[1] * 2 * dq[0] * l[0] * l[1] * np.cos(q[1]) + m[
                                      1] * dq[0] * l[1] ** 2
                                  + m[1] * dq[1] * l[0] * l[1] * np.cos(q[1]) + m[1] *
                                  dq[1] * l[1] ** 2,
                                  m[1] * dq[0] * l[0] * l[1] * np.cos(q[1])
                                  + m[1] * dq[0] * l[1] ** 2 + m[1] * dq[1] * l[
                                      1] ** 2]])

                return dE

            error_func = lambda x: self.backend.kinetic_energy(q, x)
            jacobian_func = lambda x: dEk(q, x)

            return sutils.inverse_grad_desc(E, error_func, jacobian_func,
                                            name='inv. in. energy',
                                            q0=dq0, max_steps=1000, max_tries=10)

        def inverse_potential(E: float = 0, q0=None):
            def dEp(q):
                l = self.l
                m = self.m
                dE = np.asarray([[-g * m[0] * l[0] * np.cos(q[0]) - g * m[1] * l[
                    0] * np.cos(q[0]) - g * m[1] * l[1] *
                                  np.cos(q[0] + q[1]),
                                  - g * m[1] * l[1] * np.cos(q[0] + q[1])]])
                return dE

            q = sutils.inverse_grad_desc(E, self.backend.potential_energy, dEp,
                                         name='inv. pot. energy',
                                         q0=q0, max_steps=1000, max_tries=10)

            return np.arctan2(np.sin(q), np.cos(q))

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
        q_0 = inverse_potential(E=E_p)
        dq_0 = inverse_kinetic(E=E_k, q=q_0)

        # Initialize current positional values
        self.x_0 = np.append(q_0, dq_0)
        self.x_cur = self.x_0

        self.q_0 = q_0
        self.q_cur = self.q_0

        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = self.p_0

        # Returns the initial position in joint coordinates
        return q_0

    def solve(self, t: float):
        raise NotImplementedError

    def detect_transition(self, q_des):
        new_angle = q_des
        old_angle = new_angle
        if self.q_des_prev is not None:
            old_angle = self.q_des_prev
        self.q_des_prev = q_des

        def is_II(angle):
            return np.asarray([int(np.pi / 2 < angle[0] < np.pi),
                               int(np.pi / 2 < angle[1] < np.pi)])

        def is_III(angle):
            return np.asarray([int(-np.pi / 2 > angle[0] > -np.pi),
                               int(-np.pi / 2 > angle[1] > -np.pi)])

        self.transition += is_III(old_angle) * is_II(new_angle) * np.asarray([-1, -1]) \
                           + is_II(old_angle) * is_III(new_angle) * np.asarray([1, 1])

        return self.transition

    def inverse_kins(self, params: dict = None):
        p = params['pos']
        v = params['speed']
        # coils = params['coils']

        # Check reachability
        r_0 = np.linalg.norm(p)
        if r_0 > np.sum(self.l):
            p = p / r_0 * np.sum(self.l)

        # Calculate joint position
        q0 = self.x_cur[0:self.num_dof]
        q = self.backend.bad_invkin(p, q0=q0, K=1.0, tol=1e-2, max_steps=100)

        #  Correct angle based on transitions between quadrants
        trans = self.detect_transition(q)
        q = q + trans * 2 * np.pi  # This accounts for the non-uniqueness of the inv-kinematics

        # Calculate the joint speed
        dq = np.linalg.pinv(self.backend.jacobian(q))[:, 0:-1] @ v

        return np.asarray([q, dq])

    # TODO: Expand this in the future to calculate the speeds as wel
    def forward_kins(self,
                     params: dict = None):
        q = params['joints']
        return self.backend.forward_kinematics(q)[0:-1]

    def get_cartesian_state(self):
        # Get joint positions and speeds
        q = np.asarray(self.x_cur[:self.num_dof])
        dq = np.asarray(self.x_cur[self.num_dof:])

        # Calculate the cart. positions and speeds
        p = np.asarray(self.backend.forward_kinematics(q)[0:-1])
        v = np.asarray((self.backend.jacobian(q) @ dq)[0:-1])

        return np.asarray([p, v])

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = np.asarray(self.x_cur[0:self.num_dof])

        # Calculate the link positions
        p = self.backend.forward_kinematics_for_each_link(q)
        p = np.asarray(p[:, 0:-1])

        return p

    def get_energies(self):
        # Get joint positions and speeds
        q = np.asarray(self.x_cur[0:self.num_dof])
        dq = np.asarray(self.x_cur[self.num_dof:])

        # Calculate energies
        E_pot = np.asarray([self.backend.potential_energy(q, absolute=False)])
        E_kin = np.asarray([self.backend.kinetic_energy(q, dq)])

        return np.asarray([E_pot, E_kin])
