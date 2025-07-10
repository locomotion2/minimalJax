from src.CONSTANTS import *

import jax
import jax.numpy as jnp

from src.models.base_models import BaseModel
import src.discovery_utils as sutils

class Pendulum(BaseModel):
    """
    A Pendulum model class structured for optimal performance with JAX.

    Instance methods handle state and parameters, while core calculations
    are delegated to pure, JIT-compiled static methods.
    """
    def __init__(self, params: dict = None):
        if params.get('not_inherited', True):
            # Physical parameters of the pendulum
            self.l = params.get('l', 1.0)
            self.m = params.get('m', 0.1)
            self.k_f = params.get('k_f', 0.0)

        # JAX random number generator key
        self.rng = jax.random.PRNGKey(0)
        super().__init__(params)

    # -----------------------------------------------------------------
    # JIT-Compiled Static Methods for High-Performance Computation
    # -----------------------------------------------------------------

    @staticmethod
    @jax.jit
    def eqs_motion(t, x, controller, l, m, k_f):
        """ JIT-compiled equations of motion. """
        q, dq = x[0], x[1]
        tau = controller(t, q, dq)

        # Equations of motion for a simple pendulum
        ddq = (g / l) * jnp.cos(q) - (k_f / (m * l**2)) * dq + tau / (m * l**2)
        
        return jnp.array([dq, ddq])

    @staticmethod
    @jax.jit
    def inverse_potential(E_p: float, m: float, l: float) -> float:
        """ Safely calculates the angle `q` from a given potential energy `E_p`. """
        # Invert E_p = -m*g*l*(1 + sin(q))
        val = -E_p / (m * g * l) - 1.0
        # Clamp value to the valid range for arcsin to avoid NaN errors
        return jnp.arcsin(jnp.clip(val, -1.0, 1.0))

    @staticmethod
    @jax.jit
    def inverse_kinetic(E_k: float, m: float, l: float) -> float:
        """ Calculates the angular velocity `dq` from a given kinetic energy `E_k`. """
        # Invert E_k = 0.5 * m * (l*dq)^2
        return jnp.sqrt(2.0 * E_k / (m * l**2))

    @staticmethod
    @jax.jit
    def forward_kins(q: jnp.ndarray, l: float) -> jnp.ndarray:
        """ Calculates cartesian position `p` from joint angle `q`. """
        return jnp.array([l * jnp.cos(q), l * jnp.sin(q)]).flatten()

    @staticmethod
    @jax.jit
    def get_energies_from_state(x: jnp.ndarray, m: float, l: float) -> jnp.ndarray:
        """ Calculates kinetic and potential energies from state `x`. """
        q, dq = x[0], x[1]
        E_k = 0.5 * m * (l * dq)**2
        E_p = -m * g * l * (1.0 + jnp.sin(q))
        return jnp.array([E_k, E_p])

    # -----------------------------------------------------------------
    # Instance Methods for State Management and interfacing
    # -----------------------------------------------------------------

    def make_eqs_motion(self, params: dict = None):
        """ Returns a callable function for the ODE solver. """
        controller = params['controller']
        
        # This closure captures the controller and model parameters
        def ode_func(t, x):
            return self.eqs_motion(t, x, controller, self.l, self.m, self.k_f)
        
        return ode_func

    def select_initial(self, params: dict = None):
        """
        Selects an initial state [q, dq] based on energy distribution.
        This method manages the class state and uses JIT functions for calculations.
        """
        params = params or {}
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0.0)

        # Split PRNG key for reproducible randomness
        self.rng, alpha_key, beta_key = jax.random.split(self.rng, 3)
        alpha = jax.random.uniform(alpha_key)
        beta = jax.random.uniform(beta_key)

        E_k, E_p = 0.0, 0.0
        if mode == 'speed':
            E_k = E_d
        elif mode == 'position':
            E_p = E_d
        elif mode == 'random_des':
            E_k = alpha * E_d
            E_p = (1.0 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20.0
            E_k = alpha * E_rand
            E_p = (1.0 - alpha) * E_rand

        # Use the static, JIT-compiled methods for calculation
        q_0 = self.inverse_potential(E_p, self.m, self.l)
        dq_0 = self.inverse_kinetic(E_k, self.m, self.l)

        # Update and store the state within the instance
        self.x_0 = jnp.array([q_0, dq_0]).flatten()
        self.x_cur = self.x_0
        self.q_0 = self.x_0[0:1]
        self.q_cur = self.q_0
        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = self.p_0

        return q_0

    def get_link_cartesian_positions(self):
        """ Gets cartesian positions by calling the JIT-compiled forward kinematics. """
        q = self.x_cur[0:self.num_dof]
        return self.forward_kins(q, self.l)

    def get_energies(self):
        """ Gets energies by calling the JIT-compiled energy function. """
        return self.get_energies_from_state(self.x_cur, self.m, self.l)

    def get_joint_state(self):
        """ Returns the current joint state [q, dq]. """
        q = self.x_cur[0:self.num_dof].flatten()
        dq = self.x_cur[self.num_dof:].flatten()
        return [q, dq]