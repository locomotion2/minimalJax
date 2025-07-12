from src.CONSTANTS import *

# import numpy as np  <- Replaced with jax.numpy
import jax
import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod
from src.discovery_utils import gaus
from functools import partial


class BaseController(ABC):
    def __init__(self, params: dict = None):
        delta_t = params.get('delta_t') # Use .get for safety
        if delta_t is None:
            self.delta_t = MIN_TIMESTEP
        else:
            self.delta_t = delta_t

        self.n_dof = params.get('num_dof', 2)

    @abstractmethod
    def restart(self):
        pass


class ConstantOutput(BaseController):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.value = params['value']

    def input(self, inputs: dict = None):
        return self.value

    def restart(self):
        pass


class basePIDController(BaseController):
    def __init__(self, params: dict = None):
        super().__init__(params)
        gains_outer = params.get('gains_outer') # Use .get for safety
        gains_eigen = params.get('gains_eigen') # Use .get for safety
        gains_mode = params.get('mode')

        if gains_outer is None:
            self.gains_outer = jnp.asarray([1.0, 0.0, 0.0])
            self.gains_eigen = jnp.asarray([1.0, 0.0, 0.0])
        else:
            # Ensure data types are JAX-compatible (e.g., float32)
            self.gains_outer = jnp.asarray(gains_outer, dtype=jnp.float32)
            self.gains_eigen = jnp.asarray(gains_eigen, dtype=jnp.float32)
        if gains_mode is None:
            self.mode = True
        elif gains_mode == 'maximal':
            self.mode = True
        elif gains_mode == 'minimal':
            self.mode = False
        else:
            raise NotImplementedError
        self.gains_cur = self.gains_eigen

        # Initialize with JAX arrays
        self.q_d = jnp.zeros(self.n_dof)
        self.dq_d = jnp.zeros(self.n_dof)
        self.q_d_prev = jnp.zeros(self.n_dof)
        self.e_P_accum = jnp.zeros(self.n_dof)
        self.e_cur = jnp.zeros(self.n_dof)
        self.tau_last = jnp.zeros(self.n_dof)
        self.tau_traj = jnp.asarray([jnp.zeros(self.n_dof)])
        self.q_d_traj = jnp.asarray([jnp.zeros(self.n_dof)])
        self.e_traj = jnp.asarray([jnp.zeros(3)])

    def restart(self):
        # Target variables
        self.q_d = jnp.zeros(self.n_dof)
        self.dq_d = jnp.zeros(self.n_dof)

        # D and I variables
        self.q_d_prev = jnp.zeros(self.n_dof)
        self.e_P_accum = jnp.zeros(self.n_dof)

        # Storage and help variables
        self.e_cur = jnp.zeros(self.n_dof)
        self.e_traj = jnp.asarray([jnp.asarray([0, 0, 0])])
        self.tau_traj = jnp.asarray([jnp.asarray([0])])
        self.q_d_traj = jnp.asarray([jnp.asarray(self.q_d)])

    def get_force(self):
        return self.tau_last

    def get_force_traj(self):
        return self.tau_traj.flatten()

    def get_desired_traj(self):
        return self.q_d_traj

    def get_error_traj(self):
        return self.e_traj

    @abstractmethod
    def input(self, t: float, q: float, dq: float):
        pass

    def set_target(self, q_d: float, dq_d: float, params: dict = None):
        self.q_d = jnp.asarray(q_d)
        self.dq_d = jnp.asarray(dq_d)

        if not params.get('inference', False):
            # Calculate the norm of the energy difference to get a scalar value
            energy_error = jnp.linalg.norm(params.get('E') - params.get('E_d'))
            cost_energy = gaus(energy_error, 0.05) # Now gaus receives a scalar

            if self.mode:
                self.gains_cur = jnp.maximum(cost_energy * self.gains_eigen, self.gains_outer)
            else:
                self.gains_cur = jnp.minimum(1/jnp.maximum(0.001, cost_energy) * self.gains_eigen, self.gains_outer)



    def update_trajectories(self, q_d):
        """
        Correctly updates all trajectory vectors using jnp.concatenate
        and ensuring shapes are compatible.
        """
        # Calculate the scalar norm of the torque for tracking
        tau = jnp.linalg.norm(self.tau_last)

        # --- CORRECTION IS HERE ---
        # self.e_cur stores the scalar P/I/D error norms. Expand the array
        # to match the (1, 3) shape expected by e_traj.
        new_error_row = jnp.expand_dims(self.e_cur, axis=0)

        # Prepare other rows for concatenation
        new_tau_row = jnp.expand_dims(jnp.asarray([tau]), axis=0)
        new_qd_row = jnp.expand_dims(q_d, axis=0)

        # Concatenate the new rows to their respective trajectories
        self.e_traj = jnp.concatenate([self.e_traj, new_error_row], axis=0)
        self.tau_traj = jnp.concatenate([self.tau_traj, new_tau_row], axis=0)
        self.q_d_traj = jnp.concatenate([self.q_d_traj, new_qd_row], axis=0)


class PID_pos_vel_tracking_modeled(basePIDController):
    """
    A PID controller optimized for JAX.

    The core logic is JIT-compiled in a static method for maximum performance,
    while the `input` method handles state updates.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)

    @staticmethod
    @jit
    def _calculate_jit(q_cur, dq_cur, q_d, dq_d, e_P_accum, gains):
        """
        A pure, JIT-compiled function for the core PID calculations.
        """
        # Proportional error
        e_P = q_d - q_cur

        # Integral error (updated accumulated error)
        new_e_P_accum = e_P_accum + e_P
        e_I = new_e_P_accum

        # Derivative error
        e_D = dq_d - dq_cur

        # Combine errors with gains to calculate torque
        P_gain, I_gain, D_gain = gains
        tau = P_gain * e_P + I_gain * e_I + D_gain * e_D
        
        # We return the new accumulated error to be stored in the class state
        return tau, new_e_P_accum

    def input(self, t, q, dq):
        """
        Calculates the control input by calling the JIT-compiled function.
        This method handles the stateful parts of the controller.
        """
        # Ensure inputs are JAX arrays for the JIT function
        q_cur = jnp.asarray(q)
        dq_cur = jnp.asarray(dq)

        # Call the pure, high-performance JIT-compiled function
        tau, new_e_P_accum = self._calculate_jit(
            q_cur, dq_cur, self.q_d, self.dq_d, self.e_P_accum, self.gains_cur
        )

        # Compute individual error components for tracking
        e_P = self.q_d - q_cur
        e_I = new_e_P_accum
        e_D = self.dq_d - dq_cur

        P_gain, I_gain, D_gain = self.gains_cur
        self.e_cur = jnp.asarray([
            jnp.linalg.norm(P_gain * e_P),
            jnp.linalg.norm(I_gain * e_I),
            jnp.linalg.norm(D_gain * e_D),
        ])

        # Update the state of the controller instance (side-effects)
        self.tau_last = tau
        self.e_P_accum = new_e_P_accum
        self.q_d_prev = self.q_d

        return tau