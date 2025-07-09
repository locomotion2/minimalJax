from src.CONSTANTS import *

import numpy as np # Keep for solve_ivp for now
import jax
import jax.numpy as jnp
from scipy.integrate import odeint, solve_ivp

import src.discovery_utils as sutils
from src.models.pendulum import Pendulum


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
        # Initialize a JAX PRNG key for potential future use or if any jax.random operations are added.
        # For minimal changes, a fixed key is used here. For proper randomness, this key should be managed externally.
        self._jax_key = jax.random.PRNGKey(0)


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

        # Use jnp.concatenate for initial state
        sol = solve_ivp(
            fun=self.create_dynamics(controllers),
            t_span=(0, t_max),
            y0=jnp.concatenate([q0, dq0]),
            method='LSODA',
            dense_output=dense,
            t_eval=jnp.arange(0, t_max, dt) if dt and dense else None,
            max_step=dt if dt else jnp.inf,
            **kwargs,
        )

        if verbose:
            print(f"Solve IVP finished with '{sol.message}'.")
            print(
                f"In total {len(sol.t)} time points were evaluated and the rhs was evaluated {sol.nfev} times.")

        traj = sol.y.T
        n = self.dof
        # Use jnp.arctan2, jnp.sin, jnp.cos
        q = jnp.arctan2(jnp.sin(traj[:, 0:n]), jnp.cos(traj[:, 0:n])) if wrap else traj[:,
                                                                                0:n]
        if return_sol:
            return sol.t, q, traj[:, n:], sol
        else:
            return sol.t, q, traj[:, n:]

    def compute_equilibrium(self, q0=None):
        if q0 is None:
            q0 = jnp.zeros(self.dof) # Use jnp.zeros

        def viscous_damping(t, q, dq):
            return -2 * self.mass_matrix(q) @ dq

        _, q, _, sol = self.sim(
            q0=q0,
            dq0=jnp.zeros(self.dof), # Use jnp.zeros
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
                return jnp.inf # Use jnp.inf
            else:
                return jnp.linalg.norm(dq) - eps # Use jnp.linalg.norm

        convergence_check.terminal = terminal
        return convergence_check

    def mass_matrix(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array and jnp.cos
        return jnp.array([[l[0] ** 2 * m[0] + m[1] * (
                    l[0] ** 2 + 2 * l[0] * l[1] * jnp.cos(q[1]) + l[1] ** 2),
                          1.0 * l[1] * m[1] * (l[0] * jnp.cos(q[1]) + l[1])],
                         [1.0 * l[1] * m[1] * (l[0] * jnp.cos(q[1]) + l[1]),
                          1.0 * l[1] ** 2 * m[1]]])

    def gravity(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array and jnp.cos
        return jnp.array(
            [[g * (l[0] * m[0] * jnp.cos(q[0]) + m[1] * (
                        l[0] * jnp.cos(q[0]) + l[1] * jnp.cos(q[0] + q[1])))],
             [g * l[1] * m[1] * jnp.cos(q[0] + q[1])]]).flatten()

    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array and jnp.sin
        return jnp.array([[-2.0 * dq[0] * dq[1] * l[0] * l[1] * m[1] * jnp.sin(
            q[1]) - 1.0 * dq[1] ** 2 * l[0] * l[1] * m[1] * jnp.sin(q[1]) + 1.0 * k[0] * q[0] - 1.0 * k[
                              0] * qr[0]], [
                             1.0 * dq[0] ** 2 * l[0] * l[1] * m[1] * jnp.sin(
                                 q[1]) + 1.0 * k[1] * q[1] - 1.0 * k[
                                 1] * qr[1]]]).flatten()

    def kinetic_energy(self, q, dq):
        # Use jnp.ndim instead of q.ndim
        if jnp.ndim(q) == jnp.ndim(dq) == 1:
            return self._kinetic_energy(q, dq)
        elif jnp.ndim(q) == jnp.ndim(dq) == 2:
            return self._kinetic_energy(q.T, dq.T)
        else:
            raise ValueError

    def potential_energy(self, q, absolute=False):
        # Use jnp.ndim instead of q.ndim
        if jnp.ndim(q) == 1:
            return self._potential(q) - (0 if absolute else self.U0)
        elif jnp.ndim(q) == 2:
            return self._potential(q.T) - (0 if absolute else self.U0)
        else:
            raise ValueError

    def forward_kinematics(self, q):
        # Use jnp.ndim instead of q.ndim
        if jnp.ndim(q) == 1:
            return self._fkin(q).flatten()
        elif jnp.ndim(q) == 2:
            return self._fkin(q.T).T.squeeze()
        else:
            raise ValueError

    def forward_kinematics_for_each_link(self, q):
        # Use jnp.ndim instead of q.ndim
        if jnp.ndim(q) == 1:
            return sutils.hom2xyphi(self._link_positions(q).reshape((-1, 3, 3)))
        elif jnp.ndim(q) == 2:
            out = jnp.empty((q.shape[0], self.dof, 3)) # Use jnp.empty
            for i in range(q.shape[0]):
                out = out.at[i, :, :].set(sutils.hom2xyphi(self._link_positions(q[i]).reshape((-1,
                                                                                    3,
                                                                                    3))))
            return out
        else:
            raise ValueError

    def bad_invkin(self, cart, q0=None, K=1.0, tol=1e-3, max_steps=100):
        if q0 is not None:
            q = q0
        else:
            self._jax_key, subkey = jax.random.split(self._jax_key) # Split key for randomness
            q = jax.random.uniform(subkey, shape=(self.dof,), minval=-jnp.pi, maxval=jnp.pi) # Use jax.random.uniform

        if cart.size == 3:
            f = lambda x: self.forward_kinematics(x)
            if self.dof == 3:
                A = lambda x: jnp.linalg.inv(self.jacobian(x)) # Use jnp.linalg.inv
            else:
                A = lambda x: jnp.linalg.pinv(self.jacobian(x)) # Use jnp.linalg.pinv
        elif cart.size == 2:
            f = lambda x: self.forward_kinematics(x)[0:2]
            A = lambda x: jnp.linalg.pinv(self.jacobian(x)[0:2, :]) # Use jnp.linalg.pinv
        elif cart.size == 1:
            f = lambda x: self.forward_kinematics(x)[0:1]
            A = lambda x: jnp.linalg.pinv(self.jacobian(x)[0:1, :]) # Use jnp.linalg.pinv
        else:
            raise ValueError("Illegal length of desired task-space pose")

        step = 0
        while True:
            if step > max_steps:
                print(f"No invkin solution found for {cart}, trying again")
                self._jax_key, subkey = jax.random.split(self._jax_key) # Split key
                q = jax.random.uniform(subkey, shape=(self.dof,), minval=-jnp.pi, maxval=jnp.pi) # Use jax.random.uniform
                step = 0

            e = cart - f(q)
            if jnp.linalg.norm(e) < tol: # Use jnp.linalg.norm
                break
            try:
                inc = A(q) @ jnp.squeeze(K * e) # Use jnp.squeeze
            except ValueError:
                inc = (A(q) * jnp.squeeze(K * e))[:, 0] # Use jnp.squeeze
            q += inc
            step += 1

        return jnp.arctan2(jnp.sin(q), jnp.cos(q)) # Use jnp.arctan2, jnp.sin, jnp.cos

    def _potential(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.sin
        return g * l[0] * m[0] * jnp.sin(q[0]) + g * m[1] * (
                l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])) + 0.5 * k[0] * (
                    -q[0] + qr[0]) ** 2 + 0.5 * \
            k[1] * (-q[1] + qr[1]) ** 2

    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array, jnp.sin, jnp.cos, jnp.asarray
        ddq_cons = jnp.array([[1.0 * (
                0.5 * dq[0] ** 2 * l[0] ** 2 * l[1] * m[1] * jnp.sin(2 * q[1]) + 1.0 *
                dq[0] ** 2 * l[0] * l[1] ** 2 * m[1] * jnp.sin(q[1]) + 2.0 * dq[0] * dq[1] * l[0] * l[
                    1] ** 2 * m[1] * jnp.sin(
            q[1]) + 1.0 * dq[1] ** 2 * l[0] * l[1] ** 2 * m[1] * jnp.sin(
            q[1]) - 1.0 * g * l[0] * l[1] * m[
                    0] * jnp.cos(q[0]) - 0.5 * g * l[0] * l[1] * m[1] * jnp.cos(
            q[0]) + 0.5 * g * l[0] * l[1] *
                m[1] * jnp.cos(q[0] + 2 * q[1]) - 1.0 * k[0] * l[1] * q[0] + 1.0 * k[0] *
                l[1] * qr[0] + 1.0 * k[
                    1] * l[0] * q[1] * jnp.cos(q[1]) - 1.0 * k[1] * l[0] * qr[
                    1] * jnp.cos(q[1]) + 1.0 * k[1] *
                l[1] * q[1] - 1.0 * k[1] * l[1] * qr[1] - 1.0 * l[0] * tau_in[
                    1] * jnp.cos(q[1]) + 1.0 * l[1] *
                tau_in[0] - 1.0 * l[1] * tau_in[1]) / (l[0] ** 2 * l[1] * (
                    m[0] + m[1] * jnp.sin(q[1]) ** 2))], [
                                 2.0 * (-1.0 * dq[0] ** 2 * l[0] ** 3 * l[1] * m[0] * m[
                                     1] * jnp.sin(q[1]) - 1.0 * dq[
                                            0] ** 2 * l[0] ** 3 * l[1] * m[
                                            1] ** 2 * jnp.sin(q[1]) - 1.0 * dq[0] ** 2 *
                                        l[
                                            0] ** 2 * l[1] ** 2 * m[1] ** 2 * jnp.sin(
                                             2 * q[1]) - 1.0 * dq[0] ** 2 * l[
                                            0] * l[1] ** 3 * m[1] ** 2 * jnp.sin(
                                             q[1]) - 1.0 * dq[0] * dq[1] * l[
                                            0] ** 2 * l[1] ** 2 * m[1] ** 2 * jnp.sin(
                                             2 * q[1]) - 2.0 * dq[0] * dq[1] *
                                        l[0] * l[1] ** 3 * m[1] ** 2 * jnp.sin(
                                             q[1]) - 0.5 * dq[1] ** 2 * l[0] ** 2 *
                                        l[1] ** 2 * m[1] ** 2 * jnp.sin(2 * q[1]) - 1.0 *
                                        dq[1] ** 2 * l[0] * l[
                                            1] ** 3 * m[1] ** 2 * jnp.sin(
                                             q[1]) + 0.5 * g * l[0] ** 2 * l[1] * m[0] *
                                        m[1] * jnp.cos(q[0] - q[1]) - 0.5 * g * l[
                                            0] ** 2 * l[1] * m[0] * m[
                                            1] * jnp.cos(q[0] + q[1]) + 0.5 * g * l[
                                            0] ** 2 * l[1] * m[
                                            1] ** 2 * jnp.cos(q[0] - q[1]) - 0.5 * g * l[
                                            0] ** 2 * l[1] * m[
                                            1] ** 2 * jnp.cos(q[0] + q[1]) + 1.0 * g * l[
                                            0] * l[1] ** 2 * m[0] * m[
                                            1] * jnp.cos(q[0]) + 0.5 * g * l[0] * l[
                                            1] ** 2 * m[1] ** 2 * jnp.cos(
                                             q[0]) - 0.5 * g * l[0] * l[1] ** 2 * m[
                                            1] ** 2 * jnp.cos(
                                             q[0] + 2 * q[1]) + 1.0 * k[0] * l[0] * l[
                                            1] * m[1] * q[0] * jnp.cos(
                                             q[1]) - 1.0 * k[0] * l[0] * l[1] * m[1] *
                                        qr[0] * jnp.cos(q[1]) + 1.0 * k[
                                            0] * l[1] ** 2 * m[1] * q[0] - 1.0 * k[0] *
                                        l[1] ** 2 * m[1] * qr[0] - 1.0 *
                                        k[1] * l[0] ** 2 * m[0] * q[1] + 1.0 * k[1] * l[
                                            0] ** 2 * m[0] * qr[1] - 1.0 * k[
                                            1] * l[0] ** 2 * m[1] * q[1] + 1.0 * k[1] *
                                        l[0] ** 2 * m[1] * qr[1] - 2.0 *
                                        k[1] * l[0] * l[1] * m[1] * q[1] * jnp.cos(
                                             q[1]) + 2.0 * k[1] * l[0] * l[1] *
                                        m[1] * qr[1] * jnp.cos(q[1]) - 1.0 * k[1] * l[
                                            1] ** 2 * m[1] * q[1] + 1.0 * k[
                                            1] * l[1] ** 2 * m[1] * qr[1] + 1.0 * l[
                                            0] ** 2 * m[0] * tau_in[1] + 1.0 * l[
                                            0] ** 2 * m[1] * tau_in[1] - 1.0 * l[0] * l[
                                            1] * m[1] * tau_in[
                                            0] * jnp.cos(q[1]) + 2.0 * l[0] * l[1] * m[
                                            1] * tau_in[1] * jnp.cos(
                                             q[1]) - 1.0 * l[1] ** 2 * m[1] * tau_in[
                                            0] + 1.0 * l[1] ** 2 * m[1] *
                                        tau_in[1]) / (l[0] ** 2 * l[1] ** 2 * m[1] * (
                                         2 * m[0] - m[1] * jnp.cos(2 * q[1]) + m[1]))]])
        ddq = ddq_cons - kf * jnp.asarray([[dq[0]], [dq[1]]]) # Use jnp.asarray
        return ddq

    def _kinetic_energy(self, q, dq):
        l, m, g, k, qr, kf = self.params
        # Use jnp.cos
        return 0.5 * dq[0] ** 2 * l[0] ** 2 * m[0] + 0.5 * m[1] * (
                dq[0] ** 2 * l[0] ** 2 + 2 * dq[0] ** 2 * l[0] * l[1] * jnp.cos(q[1]) +
                dq[0] ** 2 * l[1] ** 2 + 2 * dq[0] * dq[1] * l[0] * l[1] * jnp.cos(q[1]) + 2 * dq[
                    0] * dq[1] * l[1] ** 2 + dq[1] ** 2 * l[1] ** 2)

    def _energy(self, q, dq):
        l, m, g, k, qr, kf = self.params
        # Use jnp.sin and jnp.cos
        return 0.5 * dq[0] ** 2 * l[0] ** 2 * m[0] + g * l[0] * m[0] * jnp.sin(
            q[0]) + g * m[1] * (
                l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])) + 0.5 * k[0] * (
                    -q[0] + qr[0]) ** 2 + 0.5 * \
            k[1] * (-q[1] + qr[1]) ** 2 + 0.5 * m[1] * (
                    dq[0] ** 2 * l[0] ** 2 + 2 * dq[0] ** 2 * l[0] * l[1] * jnp.cos(
                q[1]) + dq[0] ** 2 * l[1] ** 2 + 2 * dq[0] * dq[1] * l[0] * l[1] * jnp.cos(q[1]) + 2 *
                    dq[0] * dq[1] * l[1] ** 2 + dq[1] ** 2 * l[1] ** 2)

    def _link_positions(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array, jnp.cos, jnp.sin
        return jnp.array([[jnp.cos(q[0]), -jnp.sin(q[0]), l[0] * jnp.cos(q[0])],
                         [jnp.sin(q[0]), jnp.cos(q[0]), l[0] * jnp.sin(q[0])], [0, 0, 1],
                         [jnp.cos(q[0] + q[1]), -jnp.sin(q[0] + q[1]),
                          l[0] * jnp.cos(q[0]) + l[1] * jnp.cos(q[0] + q[1])],
                         [jnp.sin(q[0] + q[1]), jnp.cos(q[0] + q[1]),
                          l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])],
                         [0, 0, 1]]).reshape((2, 3, 3))

    def _fkin(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array, jnp.cos, jnp.sin, jnp.arctan2
        return jnp.array([[l[0] * jnp.cos(q[0]) + l[1] * jnp.cos(q[0] + q[1])],
                         [l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])],
                         [jnp.arctan2(jnp.sin(q[0] + q[1]), jnp.cos(q[0] + q[1]))]])

    def endeffector_pose(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array, jnp.sin, jnp.cos
        return jnp.array(
            [[jnp.cos(q[0] + q[1]), -jnp.sin(q[0] + q[1]),
              l[0] * jnp.cos(q[0]) + l[1] * jnp.cos(q[0] + q[1])],
             [jnp.sin(q[0] + q[1]), jnp.cos(q[0] + q[1]),
              l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])],
             [0, 0, 1]])

    def jacobian(self, q):
        l, m, g, k, qr, kf = self.params
        # Use jnp.array, jnp.sin, jnp.cos
        return jnp.array([[-l[0] * jnp.sin(q[0]) - l[1] * jnp.sin(q[0] + q[1]),
                          -l[1] * jnp.sin(q[0] + q[1])],
                         [l[0] * jnp.cos(q[0]) + l[1] * jnp.cos(q[0] + q[1]),
                          l[1] * jnp.cos(q[0] + q[1])],
                         [1, 1]])

    def create_dynamics(self, controllers=None):
        n = self.dof
        if controllers is None:
            controllers = []

        # JIT this inner function if possible, but it captures `self` and `controllers`
        # which might require a more advanced JAX transformation (e.g., functools.partial)
        def ode_precomputed(t, y):
            q, dq = y[0:n], y[n:]
            # Ensure sum works with jax.numpy for accumulation
            tau = sum((c(t, q, dq) for c in controllers), jnp.zeros(self.dof)) # Use jnp.zeros
            ddq = self._ddq(q, dq, tau).flatten()
            return jnp.concatenate([dq, ddq]) # Use jnp.concatenate

        return ode_precomputed


class DoublePendulum(Pendulum):
    def __init__(self, params: dict = None):
        self.l = params.get('l', (0.5, 0.5))
        self.m = params.get('m', (0.05, 0.05))
        self.k_f = params.get('k_f', 0.0)
        self.transition = jnp.asarray([0, 0]) # Use jnp.asarray
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
        # Initialize a JAX PRNG key for random operations
        self._jax_key = jax.random.PRNGKey(0)


    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']
            eqs_backend = self.backend.create_dynamics(
                controllers=[controller])  # get eqs of motion
            return eqs_backend(t, x)

        return eqs_motion

    def select_initial(self, params: dict = None):
        """
        Selects the initial state of the pendulum based on a given energy `E`.
        """
        def inverse_kinetic(key, q, E: float = 0):
            """
            Calculates the inverse kinetic energy to find the initial velocities.
            """
            key, subkey1, subkey2 = jax.random.split(key, 3)
            # Use a random starting point for the velocities
            dq0 = jax.random.uniform(subkey1, shape=(2,), minval=-1, maxval=1)
            
            # Define the functions needed for the specialized solver
            # This function calculates the value we're trying to match (kinetic energy)
            value_func = lambda dq: self.backend.kinetic_energy(q, dq)
            # This function calculates the gradient of the kinetic energy wrt velocities
            grad_func = jax.grad(value_func)
            
            # Call the new, specialized function designed for this task
            dq = sutils.find_by_grad_desc(subkey2, E, dq0, value_func, grad_func)
            return dq, key

        def inverse_potential(key, E: float = 0, q0=None):
            """
            Calculates the inverse potential energy to find the initial positions.
            """
            key, subkey = jax.random.split(key)

            # Use a default starting position if none is provided
            if q0 is None:
                q0 = jnp.zeros(2)

            # Define the functions needed for the specialized solver
            # This function calculates the value we're trying to match (potential energy)
            value_func = self.backend.potential_energy
            # This function calculates the gradient of the potential energy wrt positions
            grad_func = jax.grad(value_func)

            # Call the new, specialized function
            q = sutils.find_by_grad_desc(subkey, E, q0, value_func, grad_func)

            return jnp.arctan2(jnp.sin(q), jnp.cos(q)), key

        # --- Main function logic ---

        # Handle inputs
        params = params if params is not None else {}
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0)

        # Split the key for all random operations in this function
        self._jax_key, key_alpha, key_beta = jax.random.split(self._jax_key, 3)
        
        # Use jax.random for all random numbers for consistency
        alpha = jax.random.uniform(key_alpha)
        beta = jax.random.uniform(key_beta)

        # Choose energies based on mode
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
            E_rand = beta * MAX_ENERGY / 20
            E_k = alpha * E_rand
            E_p = (1 - alpha) * E_rand
        else: # 'equilibrium'
            E_k = 0
            E_p = 0
            
        # Calculate starting positions and velocities
        q_0, self._jax_key = inverse_potential(self._jax_key, E=E_p)
        dq_0, self._jax_key = inverse_kinetic(self._jax_key, q=q_0, E=E_k)

        # Initialize current state
        # Note: jnp.append can be inefficient in JIT-compiled functions.
        # For better performance, consider pre-allocating arrays.
        self.x_0 = jnp.append(q_0, dq_0)
        self.x_cur = self.x_0

        self.q_0 = q_0
        self.q_cur = self.q_0

        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = self.p_0

        # Return the initial position in joint coordinates
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
            # Use jnp.asarray
            return jnp.asarray([int(jnp.pi / 2 < angle[0] < jnp.pi),
                               int(jnp.pi / 2 < angle[1] < jnp.pi)])

        def is_III(angle):
            # Use jnp.asarray
            return jnp.asarray([int(-jnp.pi / 2 > angle[0] > -jnp.pi),
                               int(-jnp.pi / 2 > angle[1] > -jnp.pi)])

        # Use jnp.asarray
        self.transition += is_III(old_angle) * is_II(new_angle) * jnp.asarray([-1, -1]) \
                           + is_II(old_angle) * is_III(new_angle) * jnp.asarray([1, 1])

        return self.transition

    def inverse_kins(self, params: dict = None):
        p = params['pos']
        v = params['speed']
        # coils = params['coils']

        # Check reachability
        r_0 = jnp.linalg.norm(p) # Use jnp.linalg.norm
        if r_0 > jnp.sum(self.l): # Use jnp.sum
            p = p / r_0 * jnp.sum(self.l) # Use jnp.sum

        # Calculate joint position
        q0 = self.x_cur[0:self.num_dof]
        # Pass the JAX key to bad_invkin
        q = self.backend.bad_invkin(p, q0=q0, K=1.0, tol=1e-2, max_steps=100)

        #  Correct angle based on transitions between quadrants
        trans = self.detect_transition(q)
        q = q + trans * 2 * jnp.pi # Use jnp.pi

        # Calculate the joint speed
        # Use jnp.linalg.pinv
        dq = jnp.linalg.pinv(self.backend.jacobian(q))[:, 0:-1] @ v

        return jnp.asarray([q, dq]) # Use jnp.asarray

    # TODO: Expand this in the future to calculate the speeds as wel
    def forward_kins(self,
                     params: dict = None):
        q = params['joints']
        return self.backend.forward_kinematics(q)[0:-1]

    def get_cartesian_state(self):
        # Get joint positions and speeds
        q = jnp.asarray(self.x_cur[:self.num_dof]) 
        dq = jnp.asarray(self.x_cur[self.num_dof:]) 

        # Calculate the cart. positions and speeds
        p = jnp.asarray(self.backend.forward_kinematics(q)[0:-1]) 
        v = jnp.asarray((self.backend.jacobian(q) @ dq)[0:-1]) 

        return jnp.asarray([p, v]) #

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = jnp.asarray(self.x_cur[0:self.num_dof]) # Use jnp.asarray

        # Calculate the link positions
        p = self.backend.forward_kinematics_for_each_link(q)
        p = jnp.asarray(p[:, 0:-1]) # Use jnp.asarray

        return p

    def get_energies(self):
        # Get joint positions and speeds
        q = jnp.asarray(self.x_cur[0:self.num_dof]) # Use jnp.asarray
        dq = jnp.asarray(self.x_cur[self.num_dof:]) # Use jnp.asarray

        # Calculate energies
        E_pot = jnp.asarray([self.backend.potential_energy(q, absolute=False)]) # Use jnp.asarray
        E_kin = jnp.asarray([self.backend.kinetic_energy(q, dq)]) # Use jnp.asarray

        return jnp.asarray([E_pot, E_kin]) # Use jnp.asarray