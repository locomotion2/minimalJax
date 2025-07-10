from src.CONSTANTS import *

import jax
import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod

# Replace SciPy's solver with JAX's native ODE integrator
from jax.experimental.ode import odeint


class BaseModel(ABC):
    def __init__(self, params: dict = None):
        # Load params
        self.delta_t = params.get('delta_t', MIN_TIMESTEP)
        self.state_size = params['state_size']
        self.num_dof = params['num_dof']

        # Get the equations of motion (defined in child class)
        self.eqs_motion = self.make_eqs_motion()

        # Starting values
        self.t_0 = jnp.asarray([params.get('t_0', 0.0)]).flatten()
        self.x_0 = jnp.zeros(self.state_size)
        self.q_0 = jnp.zeros(self.num_dof)

        # Current variables
        self.t_cur = self.t_0
        self.x_cur = self.x_0
        self.q_cur = self.q_0

        # Derived variables, initialized by calling abstract methods
        self.p_0 = self.get_link_cartesian_positions()
        self.E_0 = jnp.asarray([sum(self.get_energies())]).flatten()
        self.p_cur = self.p_0
        self.E_cur = self.E_0

        # Tracking variables for trajectories
        self.t_traj = jnp.asarray(self.t_cur)
        self.x_traj = jnp.expand_dims(self.x_cur, axis=0)
        self.q_traj = jnp.expand_dims(self.q_cur, axis=0)
        self.p_traj = jnp.expand_dims(self.p_cur, axis=0)
        self.E_traj = jnp.asarray(self.E_cur)

    @abstractmethod
    def make_eqs_motion(self, params: dict = None):
        raise NotImplementedError

    @abstractmethod
    def select_initial(self, params: dict = None):
        raise NotImplementedError

    def state_to_joints(self):
        return jnp.asarray(self.x_cur[0:self.num_dof])

    def restart(self, params: dict = None):
        # Handle inputs
        self.t_0 = params.get('t_0', self.t_0)

        # Set up initial state conditions by calling child class implementation
        p_0 = self.select_initial(params)

        # Restart current values
        self.t_cur = self.t_0
        self.E_0 = jnp.asarray([sum(self.get_energies())]).flatten()
        self.E_cur = self.E_0

        # Restart tracking
        self.t_traj = jnp.asarray(self.t_cur)
        self.x_traj = jnp.expand_dims(self.x_cur, axis=0)
        self.q_traj = jnp.expand_dims(self.q_cur, axis=0)
        self.p_traj = jnp.expand_dims(self.p_cur, axis=0)
        self.E_traj = jnp.asarray(self.E_cur)

        return p_0

    @staticmethod
    @jit
    def simulate(params: dict):
        """
        A static, JIT-compiled simulation function using JAX's native ODE solver.
        """
        # Unpack parameters
        user_ode_func = params['eqs']
        eqs_params = params['eqs_params']
        ts = params['ts']
        x_0 = params['x_0']

        # JAX's odeint expects func(y, t, *args), while SciPy's solve_ivp
        # expects func(t, y, *args). This wrapper handles the difference.
        def odeint_wrapper(y, t, args_tuple):
            return user_ode_func(t, y, args_tuple)

        # The user's parameters are passed as a single tuple argument
        args = (eqs_params,)
        
        # Solve the ODE
        x_traj = odeint(odeint_wrapper, x_0, ts, args)
        
        # Return the final state from the trajectory
        return x_traj[-1]

    def step(self, params: dict = None):
        # Define the integration interval
        t_final = jnp.asarray([params.get('t_final', self.t_cur.item() + self.delta_t)]).flatten()
        ts = jnp.asarray([self.t_cur.item(), t_final.item()]).flatten()

        # Simulate the system by calling the static, JIT-compiled method
        self.x_cur = BaseModel.simulate({'eqs': self.eqs_motion,
                                         'eqs_params': params,
                                         'ts': ts,
                                         'x_0': self.x_cur})

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        self.E_cur = jnp.asarray([sum(self.get_energies())]).flatten()
        self.t_cur = t_final

    def update_trajectories(self, params: dict = None):
        """
        Appends the current state to trajectories.
        Note: For very long simulations, repeatedly calling concatenate can be
        inefficient due to recompilation. The standard JAX pattern is to
        append results to a Python list and use a single jnp.stack() at the end.
        """
        self.x_traj = jnp.concatenate([self.x_traj, jnp.expand_dims(self.x_cur, axis=0)], axis=0)
        self.q_traj = jnp.concatenate([self.q_traj, jnp.expand_dims(self.q_cur, axis=0)], axis=0)
        self.p_traj = jnp.concatenate([self.p_traj, jnp.expand_dims(self.p_cur, axis=0)], axis=0)
        self.E_traj = jnp.concatenate([self.E_traj, self.E_cur], axis=0)
        self.t_traj = jnp.concatenate([self.t_traj, self.t_cur], axis=0)

    # --- Abstract Methods (to be implemented by child classes) ---
    @abstractmethod
    def inverse_kins(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def forward_kins(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def solve(self, t): raise NotImplementedError
    @abstractmethod
    def get_cartesian_state(self): raise NotImplementedError
    @abstractmethod
    def get_link_cartesian_positions(self): raise NotImplementedError
    @abstractmethod
    def get_joint_state(self): raise NotImplementedError
    @abstractmethod
    def get_energies(self): raise NotImplementedError

    # --- Getter Methods ---
    def get_time(self): return self.t_cur
    def get_state_traj(self): return self.x_traj
    def get_joint_traj(self): return self.q_traj
    def get_cartesian_traj(self): return self.p_traj
    def get_energy_traj(self): return self.E_traj
    def get_temporal_traj(self): return self.t_traj


# --- Child Classes (No changes needed, they inherit the optimized behavior) ---

class JointsGenerator(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params_traj = jnp.expand_dims(jnp.zeros(self.num_dof + 1), axis=0)
    def restart(self, params: dict = None):
        super().restart(params)
        self.params_traj = jnp.expand_dims(jnp.zeros(self.num_dof + 1), axis=0)
    def update_trajectories(self, params: dict = None):
        super().update_trajectories(params=params)
        new_param_row = jnp.expand_dims(jnp.asarray(params.get('input')), axis=0)
        self.params_traj = jnp.concatenate([self.params_traj, new_param_row], axis=0)
    def select_initial(self, params: dict = None):
        self.x_0 = params.get('x_0', self.x_0)
        self.x_cur = self.x_0
        self.q_0 = self.x_0
        self.q_cur = self.q_0
        self.p_0 = jnp.zeros(self.num_dof)
        self.p_cur = self.p_0
    def get_cartesian_state(self): raise NotImplementedError
    def get_link_cartesian_positions(self): return jnp.zeros(self.num_dof)
    @abstractmethod
    def get_params(self): raise NotImplementedError
    def get_parametric_traj(self): return self.params_traj
    def solve(self, t): raise NotImplementedError
    def inverse_kins(self, params: dict = None): raise NotImplementedError
    def forward_kins(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def get_joint_state(self): return NotImplementedError
    def get_energies(self): return jnp.zeros(2)


class PolarGenerator(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params_traj = jnp.expand_dims(jnp.zeros(self.num_dof + 1), axis=0)
    def restart(self, params: dict = None):
        super().restart(params)
        self.params_traj = jnp.expand_dims(jnp.zeros(self.num_dof + 1), axis=0)
    def update_trajectories(self, params: dict = None):
        super().update_trajectories(params=params)
        new_param_row = jnp.expand_dims(jnp.asarray(params.get('input')), axis=0)
        self.params_traj = jnp.concatenate([self.params_traj, new_param_row], axis=0)
    def select_initial(self, params: dict = None):
        self.x_0 = self.joints_to_polar(params.get('x_0', self.x_0))
        self.x_cur = self.x_0
        self.q_0 = self.polar_to_joints(self.x_0)[0]
        self.q_cur = self.polar_to_joints(self.x_cur)[0]
        self.p_0 = jnp.zeros(self.num_dof)
        self.p_cur = self.p_0
    def state_to_joints(self):
        return self.polar_to_joints(self.x_cur[0:self.num_dof])[0]
    @abstractmethod
    def polar_to_joints(self, state: jnp.ndarray = None): raise NotImplementedError
    @abstractmethod
    def joints_to_polar(self, joints: jnp.ndarray = None): raise NotImplementedError
    def get_cartesian_state(self): raise NotImplementedError
    def get_link_cartesian_positions(self): return jnp.zeros(self.num_dof)
    @abstractmethod
    def get_params(self): raise NotImplementedError
    def get_parametric_traj(self): return self.params_traj
    def solve(self, t): raise NotImplementedError
    def inverse_kins(self, params: dict = None): raise NotImplementedError
    def forward_kins(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def get_joint_state(self): return NotImplementedError
    def get_energies(self): return jnp.zeros(2)


class DummyOutput(JointsGenerator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.dq_cur = jnp.zeros(self.num_dof)
    def step(self, params: dict = None):
        # This class has a custom step that doesn't use the ODE solver
        t_final = params.get('t_final', self.t_cur + self.delta_t)
        self.x_cur = params.get('q_d')
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = jnp.asarray(self.x_cur[0:self.num_dof])
        self.dq_cur = params.get('dq_d')
        self.E_cur = jnp.asarray([sum(self.get_energies())])
        self.t_cur = t_final
    def make_eqs_motion(self, params: dict = None):
        return lambda: None # Return a dummy function
    def update_trajectories(self, params: dict = None):
        params['input'] = jnp.zeros(self.num_dof + 1)
        super().update_trajectories(params=params)
    def get_joint_state(self):
        return jnp.asarray([self.q_cur, self.dq_cur])
    def get_params(self):
        return jnp.zeros(2)