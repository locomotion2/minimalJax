from sim.CONSTANTS import *

import sys
import numpy as np
from abc import ABC, abstractmethod

# Todo: Fix this, add this to your project
sys.path.insert(1, '/home/gonz_jm/Documents/thesis_workspace/EigenHunt/pendulum-orbits/')
from links_and_joints.planar_dynamical_system.generated.rr import RR as DoublePendulumBackend

from scipy.integrate import odeint, solve_ivp


# import nbkode
# from jitcode import jitcode, y

def project(u: np.ndarray, v: np.ndarray):
    v_norm = np.linalg.norm(v)
    return (u @ v / v_norm ** 2) * v


def inverse_grad_desc(var_des, error_func, jacobian_func, name: str = 'variable',
                      num_dims: int = 2, q0=None, K=1.0, tol=1e-3, max_steps=100, max_tries=5):
    # Handle starting point
    if q0 is not None:
        q = q0
    else:
        q = np.random.uniform(-np.pi, np.pi, num_dims)

    # Shorten the functions for practicality
    f = lambda x: error_func(x)
    A = lambda x: np.linalg.pinv(jacobian_func(x))

    # Iterate until solution is found
    step = 0
    counter = 1
    while True:
        # Did not find a solution this try? Start from another seed and try
        if step > max_steps:
            step = 0
            counter += 1
            if counter > max_tries:
                print(f"No {name} solution found for {var_des} in final try, problem is hard.")
                return q

            q = np.random.uniform(-np.pi, np.pi, num_dims)
            print(f"No {name} solution found for {var_des}, try # {counter}")

        # Check if solution is good enough
        e = var_des - f(q)
        if np.linalg.norm(e) < tol:
            break

        # If not, improve solution
        try:
            inc = A(q) @ np.squeeze(K * e)
        except ValueError:
            inc = (A(q) * np.squeeze(K * e))[:, 0]
        q += inc
        step += 1

    return q


class BaseModel(ABC):
    def __init__(self, params: dict = None):
        # Load params
        self.delta_t = params.get('delta_t', MIN_TIMESTEP)
        self.state_size = params['state_size']
        self.num_dof = params['num_dof']

        # Get the equations of motion
        self.eqs_motion = self.make_eqs_motion()

        # Starting values
        self.t_0 = params.get('t_0', 0)
        self.x_0 = [0] * self.state_size
        self.q_0 = np.asarray([0] * self.num_dof)

        # Current variables
        self.t_cur = self.t_0
        self.x_cur = np.asarray(self.x_0)
        self.q_cur = np.asarray(self.q_0)

        # Derived variables
        self.p_0 = self.get_link_cartesian_positions()
        self.E_0 = sum(self.get_energies())
        self.p_cur = np.asarray(self.p_0)
        self.E_cur = np.asarray(self.E_0)

        # Tracking variables # Todo: add conditional if eval
        self.t_traj = self.t_cur
        self.x_traj = np.asarray([self.x_cur])
        self.q_traj = np.asarray([self.q_cur])
        self.p_traj = np.asarray([self.p_cur])
        self.E_traj = np.asarray([self.E_cur])

    @abstractmethod
    def make_eqs_motion(self, params: dict = None):
        raise NotImplementedError

    @abstractmethod
    def select_initial(self, params: dict = None):
        raise NotImplementedError

    def restart(self, params: dict = None):
        # Handle inputs
        self.t_0 = params.get('t_0', self.t_0)

        # Set up initial state conditions
        p_0 = self.select_initial(params)

        # Restart current values (positional values restarted in select_initial)
        self.t_cur = self.t_0
        self.E_0 = sum(self.get_energies())
        self.E_cur = np.asarray(self.E_0)

        # Restart tracking
        self.t_traj = self.t_cur
        self.x_traj = np.asarray([self.x_cur])
        self.q_traj = np.asarray([self.q_cur])
        self.p_traj = np.asarray([self.p_cur])
        self.E_traj = np.asarray([self.E_cur])

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
        t_final = params.get('t_final', self.t_cur + self.delta_t)
        # num_points = int(np.rint((t_final - self.t_cur) / self.delta_t)) + 1
        # ts = np.linspace(self.t_cur, t_final, num_points)
        ts = [self.t_cur, t_final]

        # Simulate the system until t_final
        self.x_cur = self.simulate({'eqs': self.eqs_motion, 'eqs_params': params, 'ts': ts, 'x_0': self.x_cur})

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.x_cur[0:self.num_dof]
        self.E_cur = sum(self.get_energies())
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
        # old
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
        self.E_traj = np.append(self.E_traj, [self.E_cur], axis=0)
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


class Generator(BaseModel):
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
        self.x_cur = np.asarray(self.x_0)

        self.q_0 = self.x_0
        self.q_cur = np.asarray(self.q_0)

        self.p_0 = [0] * self.num_dof
        self.p_cur = np.asarray(self.p_0)

    def get_cartesian_state(self):
        raise NotImplementedError

    def get_link_cartesian_positions(self):
        return [0] * self.num_dof

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
        return [0, 0]


class DummyOutput(Generator):
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
        self.q_cur = self.x_cur[0:self.num_dof]
        self.dq_cur = params.get('dq_d')
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final

    def make_eqs_motion(self, params: dict = None):
        def dummy_func():
            return None

        return dummy_func

    def update_trajectories(self, params: dict = None):
        params['input'] = np.asarray([0] * (self.num_dof + 1))
        super().update_trajectories(params=params)

    def get_joint_state(self):
        return [self.q_cur, self.dq_cur]

    def get_params(self):
        return [0, 0]


# TODO: decide what to do with this class, the other can implement this just as well
class CPG(BaseModel):
    def __init__(self, params: dict = None):
        # Todo: set the sizes of this variables correctly
        self.omega_cur = np.asarray([0] * self.num_dof)
        self.mu_cur = 0
        self.coils = 0

        params_cpg = np.asarray([0, 1])
        self.params_traj = np.asarray([params_cpg])
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            mu = params['mu']
            omega = params['omega']

            rho = x[0] ** 2 + x[1] ** 2
            circleDist = mu ** 2 - rho

            dx1 = -x[1] * omega + x[0] * circleDist
            dx2 = x[0] * omega + x[1] * circleDist

            return [dx1, dx2]

        return eqs_motion

    def restart(self, params: dict = None):
        p_0 = super().restart(params)
        params_cpg = np.asarray([0, 1])  # Todo: should this be changed?
        self.params_traj = np.asarray([params_cpg])
        self.coils = 0

        return p_0

    def step(self, params: dict = None):
        params['eqs_motion'] = self.eqs_motion
        super().step(params)

    def update_trajectories(self, params: dict = None):
        super().update_trajectories()
        self.mu_cur = params['mu']
        self.omega_cur = params['omega']
        params = np.asarray([self.omega_cur, self.mu_cur])
        self.params_traj = np.append(self.params_traj, [params], axis=0)

    def detect_coiling(self):
        x_new = self.x_cur
        x_old = self.x_traj[-2]

        new_angle = np.arctan2(x_new[1], x_new[0])
        old_angle = np.arctan2(x_old[1], x_old[0])
        if (-np.pi / 2 > new_angle) and (old_angle > np.pi / 2):
            self.coils += 1
        elif (-np.pi / 2 > old_angle) and (new_angle > np.pi / 2):
            self.coils -= 1

        return self.coils

    def select_initial(self, params: dict = None):
        self.x_0 = params.get('x_0', self.x_0)
        self.x_cur = np.asarray(self.x_0)

        self.q_0 = np.asarray([0] * self.num_dof)
        self.q_cur = np.asarray(self.q_0)

        self.p_0 = self.get_link_cartesian_positions().tolist()  # TODO: Change the function, maybe it solves the problem with tolist()
        self.p_cur = np.asarray(self.p_0)

    def get_cartesian_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}

        def eqs_motion(t, x, params):
            mu = params['mu']
            omega = params['omega']

            rho = x[0] ** 2 + x[1] ** 2
            circleDist = mu ** 2 - rho

            dx1 = -x[1] * omega + x[0] * circleDist
            dx2 = x[0] * omega + x[1] * circleDist

            return [dx1, dx2]

        p = self.x_cur
        v = eqs_motion(0, self.x_cur, params)

        return [p, v]

    def get_link_cartesian_positions(self):
        return self.get_cartesian_state()[0]

    def get_state_traj(self):
        return self.x_traj

    def get_temporal_traj(self):
        return self.t_traj

    def get_parametric_traj(self):
        return self.params_traj

    def solve(self, t):
        pass

    def inverse_kins(self, params: dict = None):
        pass

    def get_joint_state(self):
        pass

    def get_energies(self):
        return [0, 0]


class GPG(Generator):

    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the RL vars
        self.omega_cur = np.asarray([0] * (self.num_dof - 1))
        self.omega_past = np.asarray([0] * (self.num_dof - 1))
        self.mu_cur = np.asarray([0])

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            # Handle the inputs
            # num_dof = params[0]
            # mu = params[1]
            # omega = params[1:1+num_dof]
            # dq_escape = params[1+num_dof:1+2*num_dof]

            mu = params['mu']
            omega = params['omega']
            num_dof = params['num_dof']

            # Calculate help variables
            x = np.asarray(x, dtype=float)
            psi = mu ** 2 - np.linalg.norm(x) ** 2

            # Define the state matrix
            root = np.ones((num_dof, num_dof))
            upper = np.triu(root, 1) * omega
            lower = np.tril(root, -1) * -omega
            diag = np.diag(np.diag(np.full((num_dof, num_dof), psi)))
            A = upper + diag + lower

            # Calculate the state derivatives
            dq = A @ x

            return dq

        return eqs_motion

    def step(self, params: dict = None):
        # Run the integration
        params['num_dof'] = self.num_dof
        q_off = np.asarray([np.pi] * self.num_dof)
        q_off = 0

        # Define the integration interval # Todo: clean up, define the ability to have more points
        t_final = params.get('t_final', self.t_cur + self.delta_t)
        # num_points = int(np.rint((t_final - self.t_cur) / self.delta_t)) + 1
        # ts = np.linspace(self.t_cur, t_final, num_points)
        ts = [self.t_cur, t_final]

        # Simulate the system until t_final
        self.x_cur = self.simulate({'eqs': self.eqs_motion, 'eqs_params': params, 'ts': ts, 'x_0': self.x_cur + q_off})
        self.x_cur -= q_off

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.E_cur = sum(self.get_energies())
        self.q_cur = self.x_cur[0:self.num_dof]
        self.t_cur = t_final

        # Update current variables
        self.omega_cur = np.asarray(params['omega'])
        self.mu_cur = np.asarray(params['mu'])

    def update_trajectories(self, params: dict = None):
        # Update trajectories
        change = 0
        if np.sign(self.omega_cur) != np.sign(self.omega_past):
            change = 1
        self.omega_past = self.omega_cur
        params['input'] = np.concatenate([self.omega_cur, self.mu_cur, np.asarray([change])], axis=0)

        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        q = self.x_cur
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        dq = self.eqs_motion(0, q, params)

        return [q[0:self.num_dof], dq[0:self.num_dof]]


class Pendulum(BaseModel):
    def __init__(self, params: dict = None):
        if params.get('not_inherited', True):
            self.l = params.get('l', 1)
            self.m = params.get('m', 0.1)

        self.rng = np.random.default_rng()
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']

            x1 = x[0]
            x2 = x[1]
            tau = controller(t, x1, x2)

            dx1 = x2
            dx2 = g / np.linalg.norm(self.l) * np.sin(x1) + 1 / (
                    np.linalg.norm(self.m) * np.linalg.norm(self.l) ** 2) * tau

            return [dx1, dx2]

        return eqs_motion

    def select_initial(self, params: dict = None):

        def inverse_kinetic(E: float = 0):
            return (1 / self.l) * np.sqrt(2 * E / self.m)

        def inverse_potential(E: float = 0):
            return np.arccos(1 + E / (self.m * self.l * g))

        # Handle inputs
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0)

        # Choose energies based on mode
        alpha = self.rng.uniform(0, 1)
        beta = self.rng.uniform(0, 1)
        E_k = 0
        E_p = 0
        if mode == 'speed':
            E_k = E_d
            E_p = 0
        elif mode == 'position':
            E_k = 0
            E_p = E_d
        elif mode == 'random_des':
            E_k = alpha * E_d
            E_p = (1 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20  # TODO: Change into it's own constant
            E_k = alpha * E_rand
            E_p = (1 - alpha) * E_rand

        # Calculate starting positions
        q_0 = inverse_potential(E_p)
        dq_0 = inverse_kinetic(E_k)

        # Initialize tracking arrays
        self.x_0 = [q_0, dq_0]
        self.x_cur = np.asarray(self.x_0)

        self.q_0 = q_0
        self.q_cur = np.asarray(self.q_0)

        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = np.asarray(self.p_0)

        return [q_0, 0]

    def solve(self, t: float):
        [q0, dq0] = self.x_0
        omega_star = np.sqrt(np.abs(g) / self.l)
        q = q0 * np.cos(omega_star * t)
        dq = - omega_star * q0 * np.sin(omega_star * t)

        return [q, dq]

    def inverse_kins(self, params: dict = None):
        # Load params
        p = params['pos']
        v = params['speed']
        coils = params['coils']

        # Calculate angular position
        theta = np.arctan2(p[1], p[0])  # Finds angle in range [-pi, pi]
        theta_corrected = theta + np.pi / 2 + coils * 2 * np.pi  # This offsets the origin and accounts for the coiling of the CPG

        # Calculate angular speed
        circular_tangential = np.asarray([np.cos(theta_corrected), np.sin(theta_corrected)])
        v_proj = project(v, circular_tangential)  # This finds the speed in the pendulum circle
        omega_abs = np.sqrt(sum(v_proj ** 2)) / self.l
        omega_sign = np.sign(v_proj[1] / np.sin(theta_corrected))
        omega = omega_sign * omega_abs

        return [theta_corrected, omega]

    def forward_kins(self, params: dict = None):  # TODO: Expand this in the future to calculate the speeds as wel
        q = params['joints']
        p = [self.l * np.sin(q), - self.l * np.cos(q)]
        return np.asarray(p)

    def get_joint_state(self):
        q = self.x_cur[0:self.num_dof]
        dq = self.x_cur[self.num_dof:]
        return [q, dq]

    def get_cartesian_state(self):
        q = self.x_cur[0]
        dq = self.x_cur[1]

        p = [self.l * np.sin(q), - self.l * np.cos(q)]
        v = [self.l * dq * np.cos(q), self.l * dq * np.sin(q)]

        return [p, v]

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = self.x_cur[0:self.num_dof]

        # Calculate the link positions
        p = self.forward_kins({'joints': q})

        return p

    def get_energies(self):
        return [1 / 2 * self.m * (self.l * self.x_cur[1]) ** 2,
                self.m * -g * self.l * (1 - np.cos(self.x_cur[0]))]


class DoublePendulum(Pendulum):
    def __init__(self, params: dict = None):
        self.l = params.get('l', (0.5, 0.5))
        self.m = params.get('m', (0.05, 0.05))
        self.transition = np.asarray([0, 0])
        self.q_des_prev = None
        self.backend = DoublePendulumBackend(
            l=self.l,
            m=self.m,
            g=-g,
            k=(0, 0),
            qr=(0, 0)
        )
        params['not_inherited'] = False
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            controller = params['controller']
            eqs_backend = self.backend.create_dynamics(controllers=[controller])  # get eqs of motion
            return eqs_backend(t, x)

        return eqs_motion

    def select_initial(self, params: dict = None):

        def inverse_kinetic(q, E: float = 0):
            # Randomize vector direction
            dq0 = self.rng.uniform(-1, 1, 2)

            # Scale up so that it produces the desired energy
            E_norm = self.backend.kinetic_energy(q, dq0)
            dq0 = dq0 / E_norm * E

            def dEk(q, dq):
                l = self.l
                m = self.m
                dE = [[m[0] * dq[0] * l[0] ** 2 + m[1] * dq[0] * l[0] ** 2
                       + m[1] * 2 * dq[0] * l[0] * l[1] * np.cos(q[1]) + m[1] * dq[0] * l[1] ** 2
                       + m[1] * dq[1] * l[0] * l[1] * np.cos(q[1]) + m[1] * dq[1] * l[1] ** 2,
                       m[1] * dq[0] * l[0] * l[1] * np.cos(q[1])
                       + m[1] * dq[0] * l[1] ** 2 + m[1] * dq[1] * l[1] ** 2]]

                return dE

            error_func = lambda x: self.backend.kinetic_energy(q, x)
            jacobian_func = lambda x: dEk(q, x)

            return inverse_grad_desc(E, error_func, jacobian_func,
                                     name='inv. in. energy',
                                     q0=dq0, max_steps=1000, max_tries=10)

        def inverse_potential(E: float = 0, q0=None):
            def dEp(q):
                l = self.l
                m = self.m
                dE = [[-g * m[0] * l[0] * np.cos(q[0]) - g * m[1] * l[0] * np.cos(q[0]) - g * m[1] * l[1] * np.cos(
                    q[0] + q[1]),
                       - g * m[1] * l[1] * np.cos(q[0] + q[1])]]
                return dE

            q = inverse_grad_desc(E, self.backend.potential_energy, dEp,
                                  name='inv. pot. energy',
                                  q0=q0, max_steps=1000, max_tries=10)

            return np.arctan2(np.sin(q), np.cos(q))

        # Handle inputs
        mode = params.get('mode', 'equilibrium')
        E_d = params.get('E_d', 0)

        # Choose energies based on mode
        alpha = self.rng.uniform(0, 1)
        beta = self.rng.uniform(0, 1)
        E_k = 0
        E_p = 0
        if mode == 'speed':
            E_k = E_d
            E_p = 0
        elif mode == 'position':
            E_k = 0
            E_p = E_d
        elif mode == 'random_des':
            E_k = alpha * E_d
            E_p = (1 - alpha) * E_d
        elif mode == 'random':
            E_rand = beta * MAX_ENERGY / 20  # TODO: Change into it's own constant
            E_k = alpha * E_rand
            E_p = (1 - alpha) * E_rand

        # Calculate starting positions # TODO: implement the inverse energy functions (ask Arne)
        q_0 = inverse_potential(E=E_p)
        dq_0 = inverse_kinetic(E=E_k, q=q_0)

        # Initialize current positional values
        self.x_0 = np.append(q_0, dq_0)
        self.x_cur = np.asarray(self.x_0)

        self.q_0 = q_0
        self.q_cur = np.asarray(self.q_0)

        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = np.asarray(self.p_0)

        # Returns the initial position in cartesian coordinates
        return q_0

    def solve(self, t: float):
        raise NotImplementedError

    def detect_transition(self, q_des):
        new_angle = q_des
        old_angle = new_angle
        if self.q_des_prev is not None:
            old_angle = self.q_des_prev
        self.q_des_prev = q_des

        def is_II(angle):
            return np.asarray([int(np.pi / 2 < angle[0] < np.pi),
                               int(np.pi / 2 < angle[1] < np.pi)])

        def is_III(angle):
            return np.asarray([int(-np.pi / 2 > angle[0] > -np.pi),
                               int(-np.pi / 2 > angle[1] > -np.pi)])

        self.transition += is_III(old_angle) * is_II(new_angle) * np.asarray([-1, -1]) \
                           + is_II(old_angle) * is_III(new_angle) * np.asarray([1, 1])

        return self.transition

    # @profile
    def inverse_kins(self, params: dict = None):
        p = params['pos']
        v = params['speed']
        # coils = params['coils']

        # Check reachability
        r_0 = np.linalg.norm(p)
        if r_0 > np.sum(self.l):
            p = p / r_0 * np.sum(self.l)

        # Calculate joint position
        q0 = self.x_cur[0:self.num_dof]
        q = self.backend.bad_invkin(p, q0=q0, K=1.0, tol=1e-2, max_steps=100)

        #  Correct angle based on transitions between quadrants
        trans = self.detect_transition(q)
        q = q + trans * 2 * np.pi  # This accounts for the non-uniqueness of the inv-kinematics

        # Calculate the joint speed
        dq = np.linalg.pinv(self.backend.jacobian(q))[:, 0:-1] @ v

        return [q, dq]

    def forward_kins(self, params: dict = None):  # TODO: Expand this in the future to calculate the speeds as wel
        q = params['joints']
        return self.backend.forward_kinematics(q)[0:-1]

    def get_cartesian_state(self):
        # Get joint positions and speeds
        q = self.x_cur[0:self.num_dof]
        dq = self.x_cur[self.num_dof:]

        # Calculate the cart. positions and speeds
        p = self.backend.forward_kinematics(q)[0:-1]
        v = (self.backend.jacobian(q) @ dq)[0:-1]

        return [p, v]

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = self.x_cur[0:self.num_dof]

        # Calculate the link positions
        p = self.backend.forward_kinematics_for_each_link(q)
        p = p[:, 0:-1]

        return p

    def get_energies(self):
        # Get joint positions and speeds
        q = self.x_cur[0:self.num_dof]
        dq = self.x_cur[self.num_dof:]

        # Calculate energies
        E_pot = self.backend.potential_energy(q, absolute=False)
        E_kin = self.backend.kinetic_energy(q, dq)

        return [E_pot, E_kin]
