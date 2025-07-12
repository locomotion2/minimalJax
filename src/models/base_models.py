from src.CONSTANTS import *

import jax
import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod
from jax.experimental.ode import odeint

class BaseModel(ABC):
    def __init__(self, params: dict = None):
        # --- Static Parameters ---
        # These parameters define the model and do not change during simulation.
        self.delta_t = params.get('delta_t', MIN_TIMESTEP)
        self.state_size = params['state_size']
        self.num_dof = params['num_dof']
        self.eqs_motion = self.make_eqs_motion()

    def get_initial_state(self, params: dict = None) -> dict:
        """
        Creates and returns the initial state of the model as a dictionary.
        This function replaces the direct initialization in the original __init__.
        """
        x_0 = jnp.zeros(self.state_size)
        q_0 = jnp.zeros(self.num_dof)
        
        # The 'select_initial' method (implemented in child classes) sets the
        # specific starting conditions (e.g., random or fixed).
        # We pass a temporary state dict to it.
        temp_state = {'x_cur': x_0, 'q_cur': q_0}
        initial_conditions = self.select_initial(params, temp_state)
        x_0 = initial_conditions['x_cur']
        q_0 = initial_conditions['q_cur']

        # Initial derived values
        p_0 = self.get_link_cartesian_positions(initial_conditions)
        E_0 = jnp.asarray([sum(self.get_energies(initial_conditions))]).flatten()

        return {
            "t_cur": jnp.asarray([params.get('t_0', 0.0)]).flatten(),
            "x_cur": x_0,
            "q_cur": q_0,
            "p_cur": p_0,
            "E_cur": E_0,
            "t_traj": [jnp.asarray([params.get('t_0', 0.0)]).flatten()],
            "x_traj": [x_0],
            "q_traj": [q_0],
            "p_traj": [p_0],
            "E_traj": [E_0],
        }

    def restart(self, params: dict = None):
        """
        Restarts the simulation by generating a new initial state.
        This is now a simple wrapper around get_initial_state.
        """
        return self.get_initial_state(params)

    @staticmethod
    @jit
    def simulate(eqs_motion, x_0, ts, eqs_params):
        """
        A static, JIT-compiled simulation function using JAX's native ODE solver.
        This is now a pure function with explicit arguments.
        """
        def odeint_wrapper(y, t, args_tuple):
            return eqs_motion(t, y, args_tuple)

        args = (eqs_params,)
        x_traj = odeint(odeint_wrapper, x_0, ts, args)
        return x_traj[-1]

    def step(self, state: dict, params: dict = None) -> dict:
        """
        Performs one simulation step.
        This is now a pure function that takes the current state and returns the new state.
        """
        # Define the integration interval
        t_final = jnp.asarray([state['t_cur'].item() + self.delta_t]).flatten()
        ts = jnp.asarray([state['t_cur'].item(), t_final.item()]).flatten()

        # Simulate the system by calling the static, JIT-compiled method
        new_x_cur = BaseModel.simulate(self.eqs_motion, state['x_cur'], ts, params)

        # Create a new state dictionary for the updated values
        new_state = state.copy()
        new_state['x_cur'] = new_x_cur
        new_state['t_cur'] = t_final
        
        # Update derived variables
        new_state['q_cur'] = self.state_to_joints(new_state)
        new_state['p_cur'] = self.get_link_cartesian_positions(new_state)
        new_state['E_cur'] = jnp.asarray([sum(self.get_energies(new_state))]).flatten()

        return new_state

    def update_trajectories(self, state: dict) -> dict:
        """
        Appends the current state to trajectory lists.
        This is now more efficient, using Python lists for collection.
        """
        state['x_traj'].append(state['x_cur'])
        state['q_traj'].append(state['q_cur'])
        state['p_traj'].append(state['p_cur'])
        state['E_traj'].append(state['E_cur'])
        state['t_traj'].append(state['t_cur'])
        return state

    # --- Abstract Methods (to be implemented by child classes) ---
    @abstractmethod
    def make_eqs_motion(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def select_initial(self, params: dict, state: dict) -> dict: raise NotImplementedError
    @abstractmethod
    def inverse_kins(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def forward_kins(self, params: dict = None): raise NotImplementedError
    @abstractmethod
    def get_cartesian_state(self, state: dict): raise NotImplementedError
    @abstractmethod
    def get_link_cartesian_positions(self, state: dict): raise NotImplementedError
    @abstractmethod
    def get_joint_state(self, state: dict): raise NotImplementedError
    @abstractmethod
    def get_energies(self, state: dict): raise NotImplementedError

    # --- Helper & Getter Methods (now take state as input) ---
    def state_to_joints(self, state: dict):
        return jnp.asarray(state['x_cur'][0:self.num_dof])

    def get_time(self, state: dict): return state['t_cur']
    def get_state_traj(self, state: dict): return jnp.stack(state['x_traj'])
    def get_joint_traj(self, state: dict): return jnp.stack(state['q_traj'])
    def get_cartesian_traj(self, state: dict): return jnp.stack(state['p_traj'])
    def get_energy_traj(self, state: dict): return jnp.stack(state['E_traj'])
    def get_temporal_traj(self, state: dict): return jnp.stack(state['t_traj'])

# --- Child Classes (Refactored for the new state management) ---

class JointsGenerator(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def get_initial_state(self, params: dict = None) -> dict:
        state = super().get_initial_state(params)
        state['params_traj'] = [jnp.zeros(self.num_dof + 1)]
        return state

    def update_trajectories(self, state: dict, params: dict = None) -> dict:
        state = super().update_trajectories(state)
        new_param_row = jnp.asarray(params.get('input'))
        state['params_traj'].append(new_param_row)
        return state

    def select_initial(self, params: dict, state: dict) -> dict:
        x_0 = params.get('x_0', state['x_cur'])
        return {"x_cur": x_0, "q_cur": x_0}

    def get_link_cartesian_positions(self, state: dict): return jnp.zeros(self.num_dof)
    def get_energies(self, state: dict): return jnp.zeros(2)
    def get_parametric_traj(self, state: dict): return jnp.stack(state['params_traj'])

    # --- Unchanged abstract methods ---
    @abstractmethod
    def get_params(self): raise NotImplementedError
    @abstractmethod
    def get_joint_state(self, state: dict): return NotImplementedError
    def get_cartesian_state(self, state: dict): raise NotImplementedError
    def inverse_kins(self, params: dict = None): raise NotImplementedError
    def forward_kins(self, params: dict = None): raise NotImplementedError

class PolarGenerator(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)

    def get_initial_state(self, params: dict = None) -> dict:
        state = super().get_initial_state(params)
        state['params_traj'] = [jnp.zeros(self.num_dof + 1)]
        return state

    def update_trajectories(self, state: dict, params: dict = None) -> dict:
        state = super().update_trajectories(state)
        new_param_row = jnp.asarray(params.get('input'))
        state['params_traj'].append(new_param_row)
        return state

    def select_initial(self, params: dict, state: dict) -> dict:
        x_0_joints = params.get('x_0', state['x_cur'])
        x_0_polar = self.joints_to_polar(x_0_joints)
        q_0 = self.polar_to_joints(x_0_polar)[0]
        return {"x_cur": x_0_polar, "q_cur": q_0}

    def state_to_joints(self, state: dict):
        return self.polar_to_joints(state['x_cur'][0:self.num_dof])[0]
        
    def get_link_cartesian_positions(self, state: dict): return jnp.zeros(self.num_dof)
    def get_energies(self, state: dict): return jnp.zeros(2)
    def get_parametric_traj(self, state: dict): return jnp.stack(state['params_traj'])

    # --- Unchanged abstract methods ---
    @abstractmethod
    def polar_to_joints(self, state_polar: jnp.ndarray = None): raise NotImplementedError
    @abstractmethod
    def joints_to_polar(self, joints: jnp.ndarray = None): raise NotImplementedError
    @abstractmethod
    def get_params(self): raise NotImplementedError
    @abstractmethod
    def get_joint_state(self, state: dict): return NotImplementedError
    def get_cartesian_state(self, state: dict): raise NotImplementedError
    def inverse_kins(self, params: dict = None): raise NotImplementedError
    def forward_kins(self, params: dict = None): raise NotImplementedError


class DummyOutput(JointsGenerator):
    def step(self, state: dict, params: dict = None) -> dict:
        # This class has a custom step that bypasses the ODE solver
        new_state = state.copy()
        new_state['t_cur'] = params.get('t_final', state['t_cur'] + self.delta_t)
        new_state['x_cur'] = params.get('q_d')
        new_state['q_cur'] = jnp.asarray(new_state['x_cur'][0:self.num_dof])
        new_state['dq_cur'] = params.get('dq_d') # Custom field for this class
        new_state['p_cur'] = self.get_link_cartesian_positions(new_state)
        new_state['E_cur'] = jnp.asarray([sum(self.get_energies(new_state))])
        return new_state

    def make_eqs_motion(self, params: dict = None):
        return lambda t, y, args: None # Dummy function

    def update_trajectories(self, state: dict, params: dict = None) -> dict:
        # Ensure 'input' key exists before calling super method
        params_with_input = params.copy() if params is not None else {}
        params_with_input['input'] = jnp.zeros(self.num_dof + 1)
        return super().update_trajectories(state, params=params_with_input)

    def get_joint_state(self, state: dict):
        return jnp.asarray([state['q_cur'], state.get('dq_cur', jnp.zeros_like(state['q_cur']))])

    def get_params(self):
        return jnp.zeros(2)