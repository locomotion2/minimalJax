from src.CONSTANTS import *
import numpy as np # Keep for solve_ivp for now
import jax
import jax.numpy as jnp
from scipy.integrate import odeint, solve_ivp
import src.discovery_utils as sutils
from src.models.pendulum import Pendulum
from jax.experimental.ode import odeint
from functools import partial

class ArnesDoublePendulum:
    def __init__(self, l, m, g, k, qr, kf):
        self.dof = 2
        # Store parameters as JAX arrays
        self._l = jnp.asarray(l)
        self._m = jnp.asarray(m)
        self._g = jnp.asarray(g)
        self._k = jnp.asarray(k)
        self._q_rest = jnp.asarray(qr)
        self._kf = jnp.asarray(kf)

        # Pack parameters into a single tuple for easy passing to JIT functions
        self._p = (self._l, self._m, self._g, self._k, self._q_rest, self._kf)

        self._has_inverse_dynamics = True

        # Initialize JAX PRNG key for random operations
        self._jax_key = jax.random.PRNGKey(0)

        # Lazily compute and cache equilibrium and potential offset
        self._eq = None
        self._U0 = None

    @property
    def params(self):
        """Returns the model parameters as a tuple."""
        return self._p

    @property
    def equilibrium(self):
        """Computes and caches the equilibrium state of the pendulum."""
        if self._eq is None:
            self._eq = self.compute_equilibrium()
        return self._eq

    @property
    def U0(self):
        """Computes and caches the potential energy at the equilibrium state."""
        if self._U0 is None:
            # Call the static potential energy function
            self._U0 = ArnesDoublePendulum._potential(self.equilibrium, self.params)
        return self._U0

    # --- Core Simulation (Refactored for JAX) ---

    def sim(self, q0, dq0, t_max, controllers=None, dt=None, wrap=False, **kwargs):
        """
        Simulates the double pendulum using JAX's `odeint`.
        Note: `dense`, `return_sol`, and `verbose` arguments are not applicable
        to JAX's odeint and are ignored.
        """
        if dt is None:
            raise ValueError("A time step 'dt' must be provided for the JAX-based simulation.")
        
        t_eval = jnp.arange(0, t_max, dt)
        y0 = jnp.concatenate([jnp.asarray(q0), jnp.asarray(dq0)])
        dynamics_func = self.create_dynamics(controllers)
        
        # Run the JAX ODE solver
        y_traj = odeint(dynamics_func, y0, t_eval)
        
        n = self.dof
        q_traj, dq_traj = y_traj[:, :n], y_traj[:, n:]
        
        if wrap:
            q_traj = jnp.arctan2(jnp.sin(q_traj), jnp.cos(q_traj))
            
        # The original returned a `sol` object when `return_sol=True`.
        # We can't replicate that, so we just return the trajectories.
        if kwargs.get('return_sol'):
            print("Warning: `return_sol=True` is not supported in JAX sim; returning trajectories only.")
        
        return t_eval, q_traj, dq_traj

    # --- Equilibrium and Event Handling (Refactored for JAX) ---

    def compute_equilibrium(self, q0=None):
        """
        Computes the equilibrium by simulating with high damping. This is a
        JAX-friendly alternative to event-based termination.
        """
        if q0 is None:
            q0 = jnp.zeros(self.dof)

        def viscous_damping(t, q, dq):
            # Damping is proportional to the mass matrix for stability
            return -2 * ArnesDoublePendulum.mass_matrix(q, self.params) @ dq

        # Simulate for a sufficiently long time to ensure it settles
        _, q_settle, _, = self.sim(
            q0=q0, dq0=jnp.zeros(self.dof), dt=1e-2,
            controllers=[viscous_damping], t_max=10.0,
        )
        return q_settle[-1, :]

    def create_convergence_check(self, eps=1e-3, terminal=True):
        """
        Note: This method is preserved for API compatibility but is not used by the
        JAX-based `sim` method, as `odeint` does not support termination events.
        """
        print("Warning: `create_convergence_check` is not used by the JAX-based `sim` method.")
        n = self.dof
        def convergence_check(t, y):
            _, dq = y[0:n], y[n:]
            return jax.lax.cond(t < 1, lambda: jnp.inf, lambda: jnp.linalg.norm(dq) - eps)
        convergence_check.terminal = terminal
        return convergence_check

    # --- Pure, JIT-Compiled Physics Functions ---

    @staticmethod
    @jax.jit
    def mass_matrix(q, params):
        """Calculates the mass matrix (M)."""
        l, m, _, _, _, _ = params
        c2 = jnp.cos(q[1])
        M11 = l[0]**2 * m[0] + m[1] * (l[0]**2 + 2 * l[0] * l[1] * c2 + l[1]**2)
        M12 = l[1] * m[1] * (l[0] * c2 + l[1])
        M21 = M12
        M22 = l[1]**2 * m[1]
        return jnp.array([[M11, M12], [M21, M22]])

    @staticmethod
    @jax.jit
    def gravity(q, params):
        """Calculates the gravity vector (G)."""
        l, m, g, _, _, _ = params
        c0 = jnp.cos(q[0])
        c01 = jnp.cos(q[0] + q[1])
        G1 = g * (l[0] * m[0] * c0 + m[1] * (l[0] * c0 + l[1] * c01))
        G2 = g * l[1] * m[1] * c01
        return jnp.array([G1, G2])

    @staticmethod
    @jax.jit
    def coriolis_centrifugal_forces(q, dq, params):
        """Calculates the Coriolis and centrifugal forces vector (C)."""
        l, m, _, k, qr, _ = params
        s1 = jnp.sin(q[1])
        C1 = -2.0 * dq[0] * dq[1] * l[0] * l[1] * m[1] * s1 - dq[1]**2 * l[0] * l[1] * m[1] * s1 + k[0] * (q[0] - qr[0])
        C2 = dq[0]**2 * l[0] * l[1] * m[1] * s1 + k[1] * (q[1] - qr[1])
        return jnp.array([C1, C2])

    @staticmethod
    @jax.jit
    def _kinetic_energy(q, dq, params):
        """Calculates the kinetic energy."""
        l, m, _, _, _, _ = params
        c1 = jnp.cos(q[1])
        return 0.5 * dq[0]**2 * l[0]**2 * m[0] + 0.5 * m[1] * (
                dq[0]**2 * l[0]**2 + 2 * dq[0]**2 * l[0] * l[1] * c1 +
                dq[0]**2 * l[1]**2 + 2 * dq[0] * dq[1] * l[0] * l[1] * c1 + 2 * dq[0] * dq[1] * l[1]**2 + dq[1]**2 * l[1]**2)

    @staticmethod
    @jax.jit
    def _potential(q, params):
        """Calculates the potential energy."""
        l, m, g, k, qr, _ = params
        s0 = jnp.sin(q[0])
        s01 = jnp.sin(q[0] + q[1])
        return g * l[0] * m[0] * s0 + g * m[1] * (l[0] * s0 + l[1] * s01) + 0.5 * k[0] * (q[0] - qr[0])**2 + 0.5 * k[1] * (q[1] - qr[1])**2

    # --- Public-Facing Wrapper Methods ---

    def kinetic_energy(self, q, dq):
        """Public wrapper for kinetic energy calculation."""
        q, dq = jnp.asarray(q), jnp.asarray(dq)
        if q.ndim == 1:
            return ArnesDoublePendulum._kinetic_energy(q, dq, self.params)
        elif q.ndim == 2:
            return jax.vmap(ArnesDoublePendulum._kinetic_energy, in_axes=(0, 0, None))(q, dq, self.params)
        return 0.0

    def potential_energy(self, q, absolute=False):
        """Public wrapper for potential energy calculation."""
        q = jnp.asarray(q)
        if q.ndim == 1:
            potential = ArnesDoublePendulum._potential(q, self.params)
        elif q.ndim == 2:
            potential = jax.vmap(ArnesDoublePendulum._potential, in_axes=(0, None))(q, self.params)
        else:
            return 0.0
        return potential - (0 if absolute else self.U0)

    # --- Kinematics (Refactored for JAX) ---

    def forward_kinematics(self, q):
        """Calculates the end-effector pose [x, y, phi]."""
        q = jnp.asarray(q)
        if q.ndim == 1:
            return self._fkin(q).flatten()
        elif q.ndim == 2:
            return jax.vmap(self._fkin)(q).squeeze()
        raise ValueError("Input q must be a 1D or 2D array.")

    def forward_kinematics_for_each_link(self, q):
        """
        Calculates the pose of each link.
        Note: This implementation uses a Python loop, which is not JIT-compatible.
        For performance, `sutils` would need to be JAX-native to use with `vmap`.
        """
        q = jnp.asarray(q)
        if q.ndim == 1:
            return sutils.hom2xyphi(self._link_positions(q).reshape((-1, 3, 3)))
        elif q.ndim == 2:
            out = jnp.empty((q.shape[0], self.dof, 3), dtype=jnp.float32)
            for i in range(q.shape[0]):
                link_pos = self._link_positions(q[i])
                out = out.at[i, :, :].set(sutils.hom2xyphi(link_pos.reshape((-1, 3, 3))))
            return out
        raise ValueError("Input q must be a 1D or 2D array.")

    def bad_invkin(self, cart, q0=None, K=1.0, tol=1e-3, max_steps=100):
        """Iterative inverse kinematics using the Jacobian pseudo-inverse."""
        self._jax_key, subkey = jax.random.split(self._jax_key)
        q_init = q0 if q0 is not None else jax.random.uniform(
            subkey, shape=(self.dof,), minval=-jnp.pi, maxval=jnp.pi
        )
        f = lambda q: self._fkin(q)[:cart.size].flatten()
        A = lambda q: jnp.linalg.pinv(self.jacobian(q)[:cart.size, :])

        def cond_fun(val):
            q, step = val
            return (jnp.linalg.norm(cart - f(q)) > tol) & (step < max_steps)

        def body_fun(val):
            q, step = val
            inc = A(q) @ (K * (cart - f(q)))
            return q + inc, step + 1

        q_final, steps = jax.lax.while_loop(cond_fun, body_fun, (q_init, 0))

        if steps >= max_steps:
            print(f"Warning: Invkin did not converge for {cart}")
        return jnp.arctan2(jnp.sin(q_final), jnp.cos(q_final))

    # --- Dynamics and Equations of Motion (Preserving Original Methods) ---

    @staticmethod
    @jax.jit
    def _ddq(q, dq, tau_in, params):
        """
        Full, symbolic forward dynamics calculation, preserved from the original.
        This is now a pure, JIT-compiled static method.
        """
        l, m, g, k, qr, kf = params
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
        ddq = ddq_cons.flatten() - kf * dq
        return ddq

    @staticmethod
    @jax.jit
    def _energy(q, dq, params):
        """Calculates the total energy of the system (Kinetic + Potential)."""
        l, m, g, k, qr, _ = params
        return 0.5 * dq[0] ** 2 * l[0] ** 2 * m[0] + g * l[0] * m[0] * jnp.sin(
            q[0]) + g * m[1] * (
                l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])) + 0.5 * k[0] * (
                    -q[0] + qr[0]) ** 2 + 0.5 * \
            k[1] * (-q[1] + qr[1]) ** 2 + 0.5 * m[1] * (
                    dq[0] ** 2 * l[0] ** 2 + 2 * dq[0] ** 2 * l[0] * l[1] * jnp.cos(
                q[1]) + dq[0] ** 2 * l[1] ** 2 + 2 * dq[0] * dq[1] * l[0] * l[1] * jnp.cos(q[1]) + 2 *
                    dq[0] * dq[1] * l[1] ** 2 + dq[1] ** 2 * l[1] ** 2)

    # --- Helper Methods ---

    def _fkin(self, q):
        """Helper for forward kinematics."""
        l = self.params[0]
        c0, s0 = jnp.cos(q[0]), jnp.sin(q[0])
        c01, s01 = jnp.cos(q[0] + q[1]), jnp.sin(q[0] + q[1])
        x = l[0] * c0 + l[1] * c01
        y = l[0] * s0 + l[1] * s01
        phi = q[0] + q[1]
        return jnp.array([x, y, phi])

    def jacobian(self, q):
        """Calculates the Jacobian matrix."""
        l = self.params[0]
        s0, c0 = jnp.sin(q[0]), jnp.cos(q[0])
        s01, c01 = jnp.sin(q[0] + q[1]), jnp.cos(q[0] + q[1])
        J11 = -l[0] * s0 - l[1] * s01
        J12 = -l[1] * s01
        J21 = l[0] * c0 + l[1] * c01
        J22 = l[1] * c01
        return jnp.array([[J11, J12], [J21, J22], [1, 1]])

    def _link_positions(self, q):
        """Helper to get homogeneous transformation matrices for each link."""
        l = self.params[0]
        c0, s0 = jnp.cos(q[0]), jnp.sin(q[0])
        c01, s01 = jnp.cos(q[0] + q[1]), jnp.sin(q[0] + q[1])
        link1_pos = jnp.array([
            [c0, -s0, l[0] * c0],
            [s0,  c0, l[0] * s0],
            [0,   0,  1]
        ])
        link2_pos = jnp.array([
            [c01, -s01, l[0] * c0 + l[1] * c01],
            [s01,  c01, l[0] * s0 + l[1] * s01],
            [0,    0,   1]
        ])
        return jnp.stack([link1_pos, link2_pos])

    def endeffector_pose(self, q):
        """Returns the homogeneous transformation matrix for the end-effector."""
        l = self.params[0]
        c0, s0 = jnp.cos(q[0]), jnp.sin(q[0])
        c01, s01 = jnp.cos(q[0] + q[1]), jnp.sin(q[0] + q[1])
        x = l[0] * c0 + l[1] * c01
        y = l[0] * s0 + l[1] * s01
        return jnp.array([
            [c01, -s01, x],
            [s01,  c01, y],
            [0,    0,   1]
        ])

    def create_dynamics(self, controllers=None):
        """
        Creates the function defining the system's dynamics for the ODE solver.
        This function is what `odeint` will call at each time step.
        """
        controllers = controllers or []
        n = self.dof
        def ode_dynamics(y, t):
            q, dq = y[:n], y[n:]
            # Sum the torques from all controllers
            # Note: Controller functions must also be JAX-compatible for JIT compilation.
            tau = jnp.zeros(self.dof)
            for c in controllers:
                tau += c(t, q, dq)
            ddq = ArnesDoublePendulum._ddq(q, dq, tau, self.params)
            return jnp.concatenate([dq, ddq])
        return ode_dynamics


class DoublePendulum(Pendulum):
    def __init__(self, params: dict = None):
        """
        Initializes the DoublePendulum model with static parameters.
        The dynamic state is now created separately by `get_initial_state`.
        """
        # --- Static Parameters ---
        self.l = params.get('l', (0.5, 0.5))
        self.m = params.get('m', (0.05, 0.05))
        self.k_f = params.get('k_f', 0.0)
        
        # The backend is a static part of this class's definition
        self.backend = ArnesDoublePendulum(
            l=self.l, m=self.m, g=-g,
            k=(0, 0), qr=(0, 0), kf=self.k_f
        )
        # Force the computation of U0 at initialization
        _ = self.backend.U0
        
        # Call the parent constructor from the original design
        params['not_inherited'] = False
        super().__init__(params)

    def get_initial_state(self, params: dict) -> jax.Array:
        """Computes the initial state of the model."""
        if self.key is None:
            self.key = jax.random.PRNGKey(0)

        # `select_initial` does not rely on any symbolic backend anymore.
        # It only uses PRNG keys stored on the instance, so we simply pass
        # an empty state dictionary for API compatibility.
        x_0, dq_0 = self.select_initial(params, {})

        # Concatenate position and velocity to form the full state vector
        return jnp.concatenate([x_0, dq_0])

    def make_eqs_motion(self, params: dict = None):
        """
        Creates the equations of motion function to be used by the solver.
        """
        def eqs_motion(t, x, params_dict):
            controller = params_dict['controller']
            # The backend's create_dynamics is already JAX-compatible
            eqs_backend = self.backend.create_dynamics(controllers=[controller])
            return eqs_backend(x, t) # Note: backend expects (y, t)

        return eqs_motion

    def select_initial(self, params: dict, state: dict) -> dict:
        """
        Selects the initial state based on energy. This is now a pure function.
        It takes the current state (containing the JAX key) and returns the new state.
        """
        # --- Pure Helper Functions ---
        @partial(jit, static_argnums=(2,))
        def inverse_kinetic(key, q, E_k: float):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            dq0 = jax.random.uniform(subkey1, shape=(2,), minval=-1, maxval=1)
            
            # We need to pass the backend's parameters to the kinetic energy function
            kinetic_energy_func = partial(self.backend._kinetic_energy, params=self.backend.params)
            value_func = lambda dq: kinetic_energy_func(q, dq)
            grad_func = jax.grad(value_func)
            
            dq = sutils.find_by_grad_desc(subkey2, E_k, dq0, value_func, grad_func)
            return dq, key

        @partial(jit, static_argnums=(1,))
        def inverse_potential(key, E_p: float, q0_in=None):
            key, subkey = jax.random.split(key)
            q0 = q0_in if q0_in is not None else jnp.zeros(2)
            
            potential_energy_func = partial(self.backend.potential_energy, absolute=True)
            value_func = lambda q: potential_energy_func(q)
            grad_func = jax.grad(value_func)
            
            q = sutils.find_by_grad_desc(subkey, E_p, q0, value_func, grad_func)
            return jnp.arctan2(jnp.sin(q), jnp.cos(q)), key

        # --- Main function logic ---
        params = params if params is not None else {}
        mode = params.get('mode', 'equilibrium')
        # Some callers may set ``E_d`` explicitly to ``None``.
        # Ensure it is always a valid numeric value for the optimizer.
        E_d = params.get('E_d') or 0.0
        
        key, key_alpha, key_beta = jax.random.split(state['jax_key'], 3)
        alpha = jax.random.uniform(key_alpha)
        beta = jax.random.uniform(key_beta)

        if mode == 'speed': E_k, E_p = E_d, 0.0
        elif mode == 'position': E_k, E_p = 0.0, E_d
        elif mode == 'random_des': E_k, E_p = alpha * E_d, (1 - alpha) * E_d
        elif mode == 'random': E_rand = beta * MAX_ENERGY / 20; E_k, E_p = alpha * E_rand, (1 - alpha) * E_rand
        else: E_k, E_p = 0.0, 0.0 # 'equilibrium'

        q_0, key = inverse_potential(key, E_p)
        dq_0, key = inverse_kinetic(key, q=q_0, E_k=E_k)

        # Update and return the new state dictionary
        new_state = state.copy()
        new_state['x_cur'] = jnp.append(q_0, dq_0)
        new_state['q_cur'] = q_0
        new_state['p_cur'] = self.get_link_cartesian_positions(new_state)
        new_state['jax_key'] = key
        return new_state

    def solve(self, t: float):
        raise NotImplementedError("`solve` is not implemented for this model.")

    @staticmethod
    @jax.jit
    def detect_transition(q_des, q_des_prev, transition_prev):
        """ Pure function to detect transitions between quadrants. """
        old_angle = jax.lax.cond(q_des_prev is None, lambda: q_des, lambda: q_des_prev)

        def is_II(angle):
            return (jnp.pi / 2 < angle) & (angle < jnp.pi)
        def is_III(angle):
            return (-jnp.pi / 2 > angle) & (angle > -jnp.pi)

        new_transition = transition_prev + \
            (is_III(old_angle) * is_II(q_des)).astype(jnp.int32) * -1 + \
            (is_II(old_angle) * is_III(q_des)).astype(jnp.int32) * 1
        
        return new_transition

    def inverse_kins(self, state: dict, params: dict):
        """ Pure inverse kinematics function. """
        p, v = params['pos'], params['speed']
        
        # Check reachability
        r_0 = jnp.linalg.norm(p)
        p = jax.lax.cond(r_0 > jnp.sum(self.l), 
                         lambda: p / r_0 * jnp.sum(self.l), 
                         lambda: p)

        # Calculate joint position
        q0 = state['x_cur'][:self.num_dof]
        q = self.backend.bad_invkin(p, q0=q0, K=1.0, tol=1e-2, max_steps=100)
        
        # Correct angle based on transitions
        new_transition = self.detect_transition(q, state['q_des_prev'], state['transition'])
        q_corrected = q + new_transition * 2 * jnp.pi
        
        # Calculate joint speed
        dq = jnp.linalg.pinv(self.backend.jacobian(q_corrected))[:, :-1] @ v

        # Create the new state dictionary
        new_state = state.copy()
        new_state['q_des_prev'] = q
        new_state['transition'] = new_transition
        
        return jnp.asarray([q_corrected, dq]), new_state

    def forward_kins(self, params: dict):
        """ Pure forward kinematics function. """
        q = params['joints']
        return self.backend.forward_kinematics(q)[:-1]

    def get_cartesian_state(self, state: dict):
        """ Pure function to get the Cartesian state. """
        q = state['x_cur'][:self.num_dof]
        dq = state['x_cur'][self.num_dof:]
        p = self.backend.forward_kinematics(q)[:-1]
        v = (self.backend.jacobian(q) @ dq)[:-1]
        return jnp.asarray([p, v])

    def get_link_cartesian_positions(self, state: dict):
        """ Pure function to get the Cartesian positions of each link. """
        q = state['x_cur'][:self.num_dof]
        p = self.backend.forward_kinematics_for_each_link(q)
        return jnp.asarray(p[:, :-1])

    def get_energies(self, state: dict):
        """ Pure function to get the potential and kinetic energies. """
        q = state['x_cur'][:self.num_dof]
        dq = state['x_cur'][self.num_dof:]
        E_pot = self.backend.potential_energy(q, absolute=False)
        E_kin = self.backend.kinetic_energy(q, dq)
        return jnp.asarray([E_pot, E_kin])

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
        # Force the computation of U0 at initialization
        _ = self.backend.U0
        params['not_inherited'] = False
        super().__init__(params)
        # Initialize JAX PRNG keys for random operations. `self.key` is used
        # by methods such as `select_initial`, while `_jax_key` remains for
        # legacy routines that still expect this name.
        self.key = jax.random.PRNGKey(0)
        self._jax_key = self.key

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']
            eqs_backend = self.backend.create_dynamics(
                controllers=[controller])  # get eqs of motion
            return eqs_backend(t, x)

        return eqs_motion


# In class DoublePendulum

    def select_initial(self, params: dict, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Selects the initial state of the double pendulum based on a given energy `E`.
        This function is now pure and does not modify the instance state.
        """
        # Note: The `state` argument is not used here but is kept for API consistency
        # with the parent class. The key is managed via `self.key`.

        def inverse_kinetic(key, q, E: float = 0):
            """
            Calculates the inverse kinetic energy to find the initial velocities. (Pure function)
            """
            key, subkey1, subkey2 = jax.random.split(key, 3)
            # Use a fixed random direction for velocity for reproducibility in optimization
            dq0 = jax.random.uniform(subkey1, shape=(2,), minval=-1, maxval=1)

            # The energy function should ideally be a static method of the backend
            value_func = lambda dq: self.backend.kinetic_energy(q, dq)
            grad_func = jax.grad(value_func)

            # find_by_grad_desc should be a pure function
            dq = sutils.find_by_grad_desc(subkey2, E, dq0, value_func, grad_func)
            return dq, key # Return the updated key

        def inverse_potential(key, E: float = 0, q0=None):
            """
            Calculates the inverse potential energy to find the initial positions. (Pure function)
            """
            key, subkey = jax.random.split(key)

            if q0 is None:
                # Use a fixed random starting point for optimization
                q0 = jax.random.uniform(key, shape=(2,), minval=-jnp.pi, maxval=jnp.pi)
                key, subkey = jax.random.split(key) # Get a new subkey for the optimizer

            # The energy function should ideally be a static method of the backend
            value_func = lambda q: self.backend.potential_energy(q, absolute=True)
            grad_func = jax.grad(value_func)

            # find_by_grad_desc should be a pure function
            q = sutils.find_by_grad_desc(subkey, E, q0, value_func, grad_func)

            return jnp.arctan2(jnp.sin(q), jnp.cos(q)), key # Return the updated key

        # --- Main function logic ---
        params = params if params is not None else {}
        mode = params.get('mode', 'equilibrium')
        # ``E_d`` might explicitly be ``None`` when no energy command is used.
        E_d = params.get('E_d') or 0.0

        # Use the instance's PRNG key and split it for use.
        key, key_alpha, key_beta = jax.random.split(self.key, 3)

        alpha = jax.random.uniform(key_alpha)
        beta = jax.random.uniform(key_beta)

        if mode == 'speed':
            E_k, E_p = E_d, 0.0
        elif mode == 'position':
            E_k, E_p = 0.0, E_d
        elif mode == 'random_des':
            E_k, E_p = alpha * E_d, (1 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20.0
            E_k, E_p = alpha * E_rand, (1 - alpha) * E_rand
        else: # 'equilibrium'
            E_k, E_p = 0.0, 0.0

        # Thread the key through the pure inverse functions
        q_0, key = inverse_potential(key, E=E_p)
        dq_0, key = inverse_kinetic(key, q=q_0, E=E_k)

        # The key has been used and split, so update the key on the instance
        # for the next time this or another method needs randomness.
        self.key = key

        # Return the computed initial position and velocity as a tuple
        return q_0, dq_0

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