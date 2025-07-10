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
        # The error vector self.e_cur is shape (N, 1). Transpose it to (1, N)
        # to match the shape of e_traj for concatenation.
        new_error_row = self.e_cur.T

        # Prepare other rows for concatenation
        new_tau_row = jnp.expand_dims(jnp.asarray([tau]), axis=0)
        new_qd_row = jnp.expand_dims(q_d, axis=0)

        # Concatenate the new rows to their respective trajectories
        self.e_traj = jnp.concatenate([self.e_traj, new_error_row], axis=0)
        self.tau_traj = jnp.concatenate([self.tau_traj, new_tau_row], axis=0)
        self.q_d_traj = jnp.concatenate([self.q_d_traj, new_qd_row], axis=0)


class PID_pos_vel_tracking_modeled(basePIDController):
    def __init__(self, params: dict = None):
        super().__init__(params)

    # The static, JIT-compiled core of the controller
    @staticmethod
    @jit
    def _calculate_jit(q_cur, dq_cur, q_d, dq_d, e_P_accum, gains_cur):
        # Calc. P error
        e_P = q_d - q_cur

        # Calc. I error
        new_e_P_accum = e_P_accum + e_P
        e_I = new_e_P_accum

        # Calc. D error
        e_D = (dq_d - dq_cur).flatten()

        # Calc. Force
        P, I, D = gains_cur
        new_e_cur = jnp.asarray([P * e_P, I * e_I, D * e_D])
        tau = new_e_cur.sum(axis=0).flatten()
        
        return tau, new_e_cur, new_e_P_accum

    def input(self, t, q, dq):
        # Ensure inputs are JAX arrays
        q_cur = jnp.asarray(q)
        dq_cur = jnp.asarray(dq)

        # Call the pure, JIT-compiled function with the current state
        tau, e_cur, e_P_accum = self._calculate_jit(
            q_cur, dq_cur, self.q_d, self.dq_d, self.e_P_accum, self.gains_cur
        )

        # Update the controller's state based on the results from the JIT'd function
        self.tau_last = tau
        self.e_cur = e_cur
        self.e_P_accum = e_P_accum
        self.q_d_prev = self.q_d # q_d is not modified in the JIT part, so update prev state here

        # print(jax.devices())
        return tau

