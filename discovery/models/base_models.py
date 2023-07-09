from discovery.CONSTANTS import *

# import sys
import numpy as np
from abc import ABC, abstractmethod

from scipy.integrate import odeint, solve_ivp


class BaseModel(ABC):
    def __init__(self, params: dict = None):
        # Load params
        self.delta_t = params.get('delta_t', MIN_TIMESTEP)
        self.state_size = params['state_size']
        self.num_dof = params['num_dof']

        # Get the equations of motion
        self.eqs_motion = self.make_eqs_motion()

        # Starting values
        self.t_0 = np.asarray([params.get('t_0', 0)]).flatten()
        self.x_0 = np.asarray([0] * self.state_size)
        self.q_0 = np.asarray([0] * self.num_dof)

        # Current variables
        self.t_cur = self.t_0
        self.x_cur = self.x_0
        self.q_cur = self.q_0

        # Derived variables
        self.p_0 = self.get_link_cartesian_positions()
        self.E_0 = np.asarray([sum(self.get_energies())]).flatten()
        self.p_cur = self.p_0
        self.E_cur = self.E_0

        # Tracking variables # Todo: add conditional if eval
        self.t_traj = self.t_cur
        self.x_traj = np.asarray([self.x_cur])
        self.q_traj = np.asarray([self.q_cur])
        self.p_traj = np.asarray([self.p_cur])
        self.E_traj = self.E_cur

    @abstractmethod
    def make_eqs_motion(self, params: dict = None):
        raise NotImplementedError

    @abstractmethod
    def select_initial(self, params: dict = None):
        raise NotImplementedError

    def state_to_joints(self):
        return np.asarray(self.x_cur[0:self.num_dof])

    def restart(self, params: dict = None):
        # Handle inputs
        self.t_0 = params.get('t_0', self.t_0)

        # Set up initial state conditions
        p_0 = self.select_initial(params)

        # Restart current values (positional values restarted in select_initial)
        self.t_cur = self.t_0
        self.E_0 = np.asarray([sum(self.get_energies())]).flatten()
        self.E_cur = self.E_0

        # Restart tracking
        self.t_traj = self.t_cur
        self.x_traj = np.asarray([self.x_cur])
        self.q_traj = np.asarray([self.q_cur])
        self.p_traj = np.asarray([self.p_cur])
        self.E_traj = self.E_cur

        return p_0

    @abstractmethod
    def inverse_kins(self, params: dict = None):
        raise NotImplementedError

    @abstractmethod
    def forward_kins(self, params: dict = None):
        raise NotImplementedError

    # @profile
    def step(self, params: dict = None):
        # Define the integration interval # Todo: clean up, define the ability to have more points
        t_final = np.asarray([params.get('t_final', self.t_cur + self.delta_t)])
        # num_points = int(np.rint((t_final - self.t_cur) / self.delta_t)) + 1
        # ts = np.linspace(self.t_cur, t_final, num_points)
        ts = np.asarray([self.t_cur, t_final]).flatten()
        # debug_print('ts', ts)

        # Simulate the system until t_final
        self.x_cur = self.simulate({'eqs': self.eqs_motion,
                                    'eqs_params': params,
                                    'ts': ts,
                                    'x_0': self.x_cur})

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        self.E_cur = np.asarray([sum(self.get_energies())]).flatten()
        self.t_cur = t_final

    # @profile
    def simulate(self, params: dict):
        # Handle inputs
        eqs = params.get('eqs')
        eqs_params = params.get('eqs_params')
        ts = params.get('ts')
        x_0 = params.get('x_0')

        # Default working well (default tolerances: rtol=1e-3, atol=1e-6)
        output = solve_ivp(eqs, t_span=ts, y0=x_0, method='RK23', args=(eqs_params,), rtol=5e-2, atol=1e-5)
        x_final = np.asarray(output.y[:, -1])

        # Todo: compare with the current system
        # old_reference
        # xs = odeint(self.eqs_motion, y0=self.x_cur, t=ts, args=(params,), rtol=1.49012e-8, atol=1.49012e-8)  # default tolerances: 1.49012e-8.
        # x_cur = np.asarray(xs[-1])

        # Todo: look into getting this to work
        # New tests
        # solver = nbkode.ForwardEuler(eqs_motion, t0=self.t_cur, y0=self.x_cur, params=params)
        # _, xs = solver.run(ts)
        # x_cur = np.asarray(xs[:, -1])

        return x_final

    def update_trajectories(self, params: dict = None):
        self.x_traj = np.append(self.x_traj, [self.x_cur], axis=0)
        self.q_traj = np.append(self.q_traj, [self.q_cur], axis=0)
        self.p_traj = np.append(self.p_traj, [self.p_cur], axis=0)
        self.E_traj = np.append(self.E_traj, self.E_cur)
        self.t_traj = np.append(self.t_traj, self.t_cur)

    @abstractmethod
    def solve(self, t):
        raise NotImplementedError

    @abstractmethod
    def get_cartesian_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_link_cartesian_positions(self):
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self):
        raise NotImplementedError

    def get_time(self):
        return self.t_cur

    @abstractmethod
    def get_energies(self):
        raise NotImplementedError

    def get_state_traj(self):
        return self.x_traj

    def get_joint_traj(self):
        return self.q_traj

    def get_cartesian_traj(self):
        return self.p_traj

    def get_energy_traj(self):
        return self.E_traj

    def get_temporal_traj(self):
        return self.t_traj


class JointsGenerator(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the tracking Todo: add conditional when training
        self.params_traj = np.asarray([np.asarray([0] * (self.num_dof + 1))])

    def restart(self, params: dict = None):
        super().restart(params)

        # Restart tracking
        self.params_traj = np.asarray([np.asarray([0] * (self.num_dof + 1))])

    def update_trajectories(self, params: dict = None):
        super().update_trajectories(params=params)

        # Update trajectories
        self.params_traj = np.append(self.params_traj, [params.get('input')], axis=0)

    def select_initial(self, params: dict = None):
        self.x_0 = params.get('x_0', self.x_0)
        self.x_cur = self.x_0

        self.q_0 = self.x_0
        self.q_cur = self.q_0

        self.p_0 = np.asarray([0] * self.num_dof)
        self.p_cur = self.p_0

    def get_cartesian_state(self):
        raise NotImplementedError

    def get_link_cartesian_positions(self):
        return np.asarray([0] * self.num_dof)

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    def get_parametric_traj(self):
        return self.params_traj

    def solve(self, t):
        raise NotImplementedError

    def inverse_kins(self, params: dict = None):
        raise NotImplementedError

    def forward_kins(self, params: dict = None):
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self):
        return NotImplementedError

    def get_energies(self):
        return np.asarray([0, 0])


class PolarGenerator(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the tracking Todo: add conditional when training
        self.params_traj = np.asarray([np.asarray([0] * (self.num_dof + 1))])

    def restart(self, params: dict = None):
        super().restart(params)

        # Restart tracking
        self.params_traj = np.asarray([np.asarray([0] * (self.num_dof + 1))])

    def update_trajectories(self, params: dict = None):
        super().update_trajectories(params=params)

        # Update trajectories
        self.params_traj = np.append(self.params_traj, [params.get('input')], axis=0)

    def select_initial(self, params: dict = None):
        self.x_0 = self.joints_to_polar(params.get('x_0', self.x_0))
        self.x_cur = self.x_0

        self.q_0 = self.polar_to_joints(self.x_0)[0]
        self.q_cur = self.polar_to_joints(self.x_cur)[0]

        self.p_0 = np.asarray([0] * self.num_dof)
        self.p_cur = self.p_0

    def state_to_joints(self):
        return self.polar_to_joints(self.x_cur[0:self.num_dof])[0]

    @abstractmethod
    def polar_to_joints(self, state: np.ndarray = None):
        raise NotImplementedError

    @abstractmethod
    def joints_to_polar(self, joints: np.ndarray = None):
        raise NotImplementedError

    def get_cartesian_state(self):
        raise NotImplementedError

    def get_link_cartesian_positions(self):
        return np.asarray([0] * self.num_dof)

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    def get_parametric_traj(self):
        return self.params_traj

    def solve(self, t):
        raise NotImplementedError

    def inverse_kins(self, params: dict = None):
        raise NotImplementedError

    def forward_kins(self, params: dict = None):
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self):
        return NotImplementedError

    def get_energies(self):
        return np.asarray([0, 0])


class DummyOutput(JointsGenerator):
    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the vel vars
        self.dq_cur = np.asarray([0] * self.num_dof)

    def step(self, params: dict = None):
        # Define the integration interval
        t_final = params.get('t_final', self.t_cur + self.delta_t)

        # Simulate the system until t_final
        self.x_cur = params.get('q_d')

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = np.asarray(self.x_cur[0:self.num_dof])
        self.dq_cur = params.get('dq_d')
        self.E_cur = np.asarray([sum(self.get_energies())])
        self.t_cur = t_final

    def make_eqs_motion(self, params: dict = None):
        def dummy_func():
            return None

        return dummy_func

    def update_trajectories(self, params: dict = None):
        params['input'] = np.asarray([0] * (self.num_dof + 1))
        super().update_trajectories(params=params)

    def get_joint_state(self):
        return np.asarray([self.q_cur, self.dq_cur])

    def get_params(self):
        return np.asarray([0, 0])