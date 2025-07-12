from src.CONSTANTS import *

# import numpy as np  <- Replaced with jax.numpy
import jax
import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod
from src.discovery_utils import gaus
from functools import partial


class BaseController(ABC):
    """
    Abstract base class for all controllers.
    """
    def __init__(self, params: dict = None):
        # Use .get() for safe dictionary access
        delta_t = params.get('delta_t')
        if delta_t is None:
            self.delta_t = MIN_TIMESTEP
        else:
            self.delta_t = delta_t

        self.n_dof = params.get('num_dof', 2)

    @abstractmethod
    def restart(self):
        """Resets the controller to its initial state."""
        pass

    @abstractmethod
    def input(self, *args, **kwargs):
        """Calculates the control input."""
        pass


class ConstantOutput(BaseController):
    """
    A simple controller that outputs a constant value.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.value = jnp.asarray(params.get('value', 0.0))

    def input(self, inputs: dict = None):
        """Returns the constant control value."""
        return self.value

    def restart(self):
        """No state to reset for this controller."""
        pass


class basePIDController(BaseController):
    """
    A base class for PID controllers, refactored for JAX compatibility.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        gains_outer = params.get('gains_outer')
        gains_eigen = params.get('gains_eigen')
        gains_mode = params.get('mode')

        # Set default gains if not provided, ensuring they are JAX arrays
        if gains_outer is None:
            self.gains_outer = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
            self.gains_eigen = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
        else:
            self.gains_outer = jnp.asarray(gains_outer, dtype=jnp.float32)
            self.gains_eigen = jnp.asarray(gains_eigen, dtype=jnp.float32)

        # Set the gain scheduling mode
        if gains_mode is None or gains_mode == 'maximal':
            self.mode = True
        elif gains_mode == 'minimal':
            self.mode = False
        else:
            raise NotImplementedError
        self.gains_cur = self.gains_eigen

        # Initialize state variables and trajectories with JAX arrays
        self.restart()

    def restart(self):
        """Resets all controller states and trajectories."""
        # Target variables
        self.q_d = jnp.zeros(self.n_dof)
        self.dq_d = jnp.zeros(self.n_dof)

        # D and I variables
        self.q_d_prev = jnp.zeros(self.n_dof)
        self.e_P_accum = jnp.zeros(self.n_dof)

        # Storage and help variables
        self.e_cur = jnp.zeros(self.n_dof)
        self.tau_last = jnp.zeros(self.n_dof)
        
        # Initialize trajectories with a starting point
        self.tau_traj = jnp.zeros((1, self.n_dof))
        self.q_d_traj = jnp.zeros((1, self.n_dof))
        self.e_traj = jnp.zeros((1, 3)) # Assuming error is tracked as 3 values (P,I,D)

    def get_force(self):
        return self.tau_last

    def get_force_traj(self):
        return self.tau_traj

    def get_desired_traj(self):
        return self.q_d_traj

    def get_error_traj(self):
        return self.e_traj

    @abstractmethod
    def input(self, t: float, q: jnp.ndarray, dq: jnp.ndarray):
        pass

    def set_target(self, q_d: jnp.ndarray, dq_d: jnp.ndarray, params: dict = None):
        """Sets the target position and velocity for the PID controller."""
        self.q_d = jnp.asarray(q_d)
        self.dq_d = jnp.asarray(dq_d)

        # Gain scheduling logic
        if params and not params.get('inference', False):
            # Calculate a scalar energy error
            energy_error = jnp.linalg.norm(params.get('E', 0.0) - params.get('E_d', 0.0))
            cost_energy = gaus(energy_error, 0.05)

            # Use jax.lax.cond for conditional logic if it needs to be JIT-compiled
            # For this class-based approach, a simple if/else is fine.
            if self.mode: # Maximal coordination
                self.gains_cur = jnp.maximum(cost_energy * self.gains_eigen, self.gains_outer)
            else: # Minimal coordination
                self.gains_cur = jnp.minimum(1 / jnp.maximum(1e-3, cost_energy) * self.gains_eigen, self.gains_outer)

    def update_trajectories(self):
        """Appends the latest state to the trajectories."""
        # Ensure new rows have the correct shape (1, N) for concatenation
        new_tau_row = jnp.expand_dims(self.tau_last, axis=0)
        new_qd_row = jnp.expand_dims(self.q_d, axis=0)
        # Assuming e_cur holds the [P, I, D] errors as a vector
        new_error_row = jnp.expand_dims(self.e_cur, axis=0)

        self.tau_traj = jnp.concatenate([self.tau_traj, new_tau_row], axis=0)
        self.q_d_traj = jnp.concatenate([self.q_d_traj, new_qd_row], axis=0)
        self.e_traj = jnp.concatenate([self.e_traj, new_error_row], axis=0)


class PID_pos_vel_tracking_modeled(basePIDController):
    """
    A PID controller with JAX-jitted core calculations.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)

    @staticmethod
    @jit
    def _calculate_jit(q_cur, dq_cur, q_d, dq_d, e_P_accum, gains, delta_t):
        """
        A pure, JIT-compiled function for the core PID calculations.
        All inputs are explicit arguments, and it returns all updated values.
        """
        # Proportional error
        e_P = q_d - q_cur

        # Integral error (updated accumulated error)
        new_e_P_accum = e_P_accum + e_P * delta_t
        e_I = new_e_P_accum

        # Derivative error
        e_D = dq_d - dq_cur

        # Store current errors for logging
        current_errors = jnp.array([jnp.linalg.norm(e_P), jnp.linalg.norm(e_I), jnp.linalg.norm(e_D)])

        # Combine errors with gains to calculate torque
        P_gain, I_gain, D_gain = gains
        tau = P_gain * e_P + I_gain * e_I + D_gain * e_D
        
        # Return new state variables and the calculated torque
        return tau, new_e_P_accum, current_errors

    def input(self, t, q, dq):
        """
        Calculates the control input by calling the JIT-compiled function
        and handles the stateful updates.
        """
        # Call the pure, high-performance JIT-compiled function
        tau, new_e_P_accum, current_errors = self._calculate_jit(
            jnp.asarray(q), jnp.asarray(dq), 
            self.q_d, self.dq_d, 
            self.e_P_accum, 
            self.gains_cur,
            self.delta_t
        )

        # Update the state of the controller instance (side-effects)
        self.tau_last = tau
        self.e_P_accum = new_e_P_accum
        self.e_cur = current_errors # Store errors for logging
        self.q_d_prev = self.q_d

        # Update trajectories after calculation
        self.update_trajectories()

        return tau
