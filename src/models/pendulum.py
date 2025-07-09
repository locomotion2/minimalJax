from src.CONSTANTS import *

import numpy as np # Keep for potential numpy operations if not fully converted
import jax
import jax.numpy as jnp

from src.models.base_models import BaseModel

import src.discovery_utils as sutils

class Pendulum(BaseModel):
    def __init__(self, params: dict = None):
        if params.get('not_inherited', True):
            self.l = params.get('l', 1)
            self.m = params.get('m', 0.1)
            self.k_f = params.get('k_f', 0.0)

        self.rng = jax.random.PRNGKey(0) # Use JAX PRNG key
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']

            x1 = x[0]
            x2 = x[1]
            tau = controller(t, x1, x2)

            dx1 = jnp.asarray([x2]) # Use jnp.asarray
            dx2 = g / jnp.linalg.norm(self.l) * jnp.cos(x1) - \
                  self.k_f * x2 + \
                  tau / (jnp.linalg.norm(self.m) * jnp.linalg.norm(self.l) ** 2) # Use jnp.linalg.norm and jnp.cos

            return jnp.asarray([dx1, dx2]).flatten() # Use jnp.asarray

        return eqs_motion

  ## In class Pendulum:

    def select_initial(self, params: dict = None):
        """
        Corrected method to select the initial state based on energy.
        """
        def inverse_kinetic(E: float = 0):
            # This function is correct.
            return jnp.asarray([(1 / self.l) * jnp.sqrt(2 * E / self.m)])

        def inverse_potential(E: float = 0):
            # Correctly inverts E_p = -m*g*l*(1 + sin(q))
            # E_p / (-m*g*l) = 1 + sin(q)
            # sin(q) = -E_p / (m*g*l) - 1
            val = -E / (self.m * self.l * g) - 1
            # Clamp the value to the valid [-1, 1] range for arcsin to avoid NaN errors.
            return jnp.asarray([jnp.arcsin(jnp.clip(val, -1.0, 1.0))])

        # --- The rest of the function remains the same ---
        params = params or {}
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0)

        # Choose energies based on mode
        key, subkey1, subkey2 = jax.random.split(self.rng, 3)
        alpha = jax.random.uniform(subkey1)
        beta = jax.random.uniform(subkey2)
        self.rng = key # Update the main key

        E_k = 0
        E_p = 0
        if mode == 'speed':
            E_k = E_d
        elif mode == 'position':
            E_p = E_d
        elif mode == 'random_des':
            E_k = alpha * E_d
            E_p = (1 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20
            E_k = alpha * E_rand
            E_p = (1 - alpha) * E_rand

        # Calculate starting positions
        q_0 = inverse_potential(E_p)
        dq_0 = inverse_kinetic(E_k)

        # Initialize tracking arrays
        self.x_0 = jnp.asarray([q_0, dq_0]).flatten()
        self.x_cur = self.x_0
        self.q_0 = q_0.flatten()
        self.q_cur = self.q_0
        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = self.p_0

        return q_0

    def solve(self, t: float):  # Todo: This is not up to date
        [q0, _] = self.x_0
        omega_star = jnp.sqrt(jnp.abs(g) / self.l) # Use jnp.sqrt, jnp.abs
        q = q0 * jnp.cos(omega_star * t) # Use jnp.cos
        dq = - omega_star * q0 * jnp.sin(omega_star * t) # Use jnp.sin

        return jnp.asarray([q, dq]) # Use jnp.asarray

    def inverse_kins(self, key, params: dict = None):
        """
        Corrected method for inverse kinematics. It is now stateless and safer.
        `key` is a jax.random.PRNGKey.
        """
        # Load params
        p = params['pos']
        v = params['speed']
        coils = params['coils']

        # Calculate angular position
        theta = jnp.arctan2(p[1], p[0])
        theta_corrected = theta + jnp.pi / 2 + coils * 2 * jnp.pi

        # Calculate angular speed
        circular_tangential = jnp.asarray([jnp.cos(theta_corrected), jnp.sin(theta_corrected)])

        # Pass the provided JAX key to sutils.project
        v_proj = sutils.project(key, v, circular_tangential)
        omega_abs = jnp.linalg.norm(v_proj) / self.l

        # Robustly determine the sign of the angular velocity using a dot product.
        # A positive sign means v_proj and the tangential vector point in the same direction.
        omega_sign = jnp.sign(jnp.dot(v_proj, circular_tangential))
        omega = omega_sign * omega_abs

        return jnp.asarray([theta_corrected, omega]).flatten()

    def forward_kins(self,
                     params: dict = None):  # TODO: Expand this in the future to calculate the speeds as well
        q = params['joints']
        p = jnp.asarray([self.l * jnp.cos(q), self.l * jnp.sin(q)]) # Use jnp.asarray, jnp.cos, jnp.sin
        return p.flatten()

    def get_joint_state(self):
        q = jnp.asarray(self.x_cur[0:self.num_dof]).flatten() # Use jnp.asarray
        dq = jnp.asarray(self.x_cur[self.num_dof:]).flatten() # Use jnp.asarray
        return [q, dq]

    def get_cartesian_state(self):
        q = self.x_cur[0]
        dq = self.x_cur[1]

        # Todo: This needs to be replaced by forward kins
        p = jnp.asarray([self.l * jnp.cos(q), self.l * jnp.sin(q)]).flatten() # Use jnp.asarray, jnp.cos, jnp.sin
        v = jnp.asarray([- self.l * dq * jnp.sin(q), self.l * dq * jnp.cos(q)]).flatten() # Use jnp.asarray, jnp.sin, jnp.cos
        return [p, v]

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = jnp.asarray(self.x_cur[0:self.num_dof]) # Use jnp.asarray

        # Calculate the link positions
        p = self.forward_kins({'joints': q})

        return p

    def get_energies(self):
        # Use jnp.asarray, jnp.sin
        return jnp.asarray([1 / 2 * self.m * (self.l * self.x_cur[1]) ** 2,
                           - self.m * g * self.l * (1 + jnp.sin(self.x_cur[0]))])