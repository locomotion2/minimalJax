from src.CONSTANTS import *

import jax
import jax.numpy as jnp
from functools import partial

from src.models.base_models import BaseModel
# Assuming sutils and other constants are available
# import src.discovery_utils as sutils

class Pendulum(BaseModel):
    """
    A Pendulum model class structured for optimal performance with JAX.

    This version uses a functional approach where methods operate on an explicit
    state dictionary, making them pure and compatible with JAX's transformations.
    """
    def __init__(self, params: dict = None):
        # --- Static Parameters ---
        # These parameters define the model and do not change during simulation.
        if params.get('not_inherited', True):
            self.l = params.get('l', 1.0)
            self.m = params.get('m', 0.1)
            self.k_f = params.get('k_f', 0.0)

        super().__init__(params)

        # Initialize a JAX PRNG key used for sampling initial conditions
        # This attribute is accessed in `select_initial`, hence it must
        # exist before that method runs.
        self.key = jax.random.PRNGKey(0)

    def get_initial_state(self, params: dict = None) -> dict:
        """
        Creates and returns the initial dynamic state of the model as a dictionary.
        """
        # Get the base initial state
        state = super().get_initial_state(params)
        # Add a JAX PRNG key to the state
        state['rng'] = jax.random.PRNGKey(0)
        # Set the specific initial conditions for the pendulum
        final_state = self.select_initial(params, state)
        return final_state

    # -----------------------------------------------------------------
    # JIT-Compiled Static Methods for High-Performance Computation
    # -----------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _eqs_motion(t, x, controller, l, m, k_f):
        """ JIT-compiled equations of motion. """
        q, dq = x[0], x[1]
        tau = controller(t, q, dq)
        # Equations of motion for a simple pendulum
        ddq = (g / l) * jnp.cos(q) - (k_f / (m * l**2)) * dq + tau / (m * l**2)
        return jnp.array([dq, ddq])

    @staticmethod
    @jax.jit
    def _inverse_potential(E_p: float, m: float, l: float) -> float:
        """ Safely calculates the angle `q` from a given potential energy `E_p`. """
        val = -E_p / (m * g * l) - 1.0
        return jnp.arcsin(jnp.clip(val, -1.0, 1.0))

    @staticmethod
    @jax.jit
    def _inverse_kinetic(E_k: float, m: float, l: float) -> float:
        """ Calculates the angular velocity `dq` from a given kinetic energy `E_k`. """
        return jnp.sqrt(2.0 * E_k / (m * l**2))

    @staticmethod
    @jax.jit
    def _forward_kins(q: jnp.ndarray, l: float) -> jnp.ndarray:
        """ Calculates cartesian position `p` from joint angle `q`. """
        return jnp.array([l * jnp.cos(q), l * jnp.sin(q)]).flatten()

    @staticmethod
    @jax.jit
    def _get_energies_from_state(x: jnp.ndarray, m: float, l: float) -> jnp.ndarray:
        """ Calculates kinetic and potential energies from state `x`. """
        q, dq = x[0], x[1]
        E_k = 0.5 * m * (l * dq)**2
        E_p = -m * g * l * (1.0 + jnp.sin(q))
        return jnp.array([E_p, E_k]) # Return potential first to match convention

    # -----------------------------------------------------------------
    # Instance Methods for State Management and interfacing
    # -----------------------------------------------------------------

    def make_eqs_motion(self, params: dict = None):
        """ Returns a callable function for the ODE solver. """
        # This closure captures the controller and model parameters
        # The controller is passed in the `params` dict during the `step` call
        def ode_func(t, x, params_dict):
            controller = params_dict['controller']
            return self._eqs_motion(t, x, controller, self.l, self.m, self.k_f)
        return ode_func

    def select_initial(self, params: dict, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Selects an initial state [q, dq] based on energy distribution.

        This is a pure function that computes initial conditions based on input parameters.
        """
        params = params or {}
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0.0)
        key = self.key # Use the key from the model's state

        # Split PRNG key for reproducible randomness
        key, alpha_key, beta_key = jax.random.split(key, 3)
        alpha = jax.random.uniform(alpha_key)
        beta = jax.random.uniform(beta_key)

        E_k, E_p = 0.0, 0.0
        if mode == 'speed':
            E_k = E_d
        elif mode == 'position':
            E_p = E_d
        elif mode == 'random_des':
            E_k, E_p = alpha * E_d, (1.0 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20.0
            E_k, E_p = alpha * E_rand, (1.0 - alpha) * E_rand

        # Use the static, JIT-compiled methods for calculation
        # Note: These inverse methods need to be defined, likely in the BaseModel or as static methods.
        # Assuming they are pure functions that don't rely on `self`'s mutable state.
        q_0 = self._inverse_potential(E_p, self.m, self.l)
        dq_0 = self._inverse_kinetic(E_k, self.m, self.l, q_0) # Pass q_0 if kinetic energy depends on position

        # Update the key in the main model state after using it
        self.key = key

        return q_0, dq_0

    def get_link_cartesian_positions(self, state: dict):
        """ Gets cartesian positions by calling the JIT-compiled forward kinematics. """
        q = state['x_cur'][0:self.num_dof]
        return self._forward_kins(q, self.l)

    def get_energies(self, state: dict):
        """ Gets energies by calling the JIT-compiled energy function. """
        return self._get_energies_from_state(state['x_cur'], self.m, self.l)

    def get_joint_state(self, state: dict):
        """ Returns the current joint state [q, dq] from the state dictionary. """
        q = state['x_cur'][0:self.num_dof].flatten()
        dq = state['x_cur'][self.num_dof:].flatten()
        return [q, dq]
