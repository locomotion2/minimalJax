from CONSTANTS import *

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from base_models import BaseModel

from identification.systems import snake_utils
from identification.src.training import trainer
from identification.src.dynamix import wrappings

import stable_baselines3.common.save_util as loader

class SnakeBaseBackend:

    def __init__(self, settings):
        # kin amounts
        self.l = settings['segment_length']

        settings['sys_utils'] = snake_utils
        params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
        train_state = trainer.create_train_state(settings, 0,
                                                 params=params)

        # set up help variables
        self.num_dof = settings['num_dof']
        self.buffer_length = settings['buffer_length']

        # set up energy function
        self.energies = wrappings.build_energy_call(settings,
                                                    params,
                                                    train_state)

        # set up state var
        self.state = None

    def kinetic_energy(self):
        T, _ = self.energies(jnp.concat(self.state))
        return T

    def potential_energy(self):
        _, V = self.energies(self.state)
        return V

    def forward_kinematics(self, q):
        q1 = q[0]
        q2 = q[1]

        p1 = -self.l / 2 * jnp.array([jnp.sin(q1),
                                      1 + jnp.cos(q1)])

        p2 = self.l / 2 * jnp.array([jnp.sin(q2),
                                     1 + jnp.cos(q2)])

        return [p1, p2]

    def forward_kinematics_for_each_link(self, q):
        return self.forward_kinematics(q)

    def jacobian(self, q):
        q1 = q[0]
        q2 = q[1]

        J_p1 = -self.l / 2 * jnp.array([jnp.cos(q1), 0],
                                       [-jnp.sin(q2), 0])

        J_p2 = self.l / 2 * jnp.array([0, jnp.cos(q2)],
                                      [0, -jnp.sin(q2)])

        return [J_p1, J_p2]

# class SnakeRealBackend(SnakeBaseBackend):


class Snake(BaseModel):
    def __init__(self, params: dict = None):
        self.backend = None
        self.rng = np.random.default_rng()
        super().__init__(params)

    def make_eqs_motion(self, params: dict = None):
        controller = params['controller']

        def eqs_motion(t, x):
            return None

        return eqs_motion

    def select_initial(self, params: dict = None):

        # Handle inputs
        mode = params.get('mode', 'equilibrium')
        # ``E_d`` can be explicitly ``None`` when no energy schedule is used.
        E_d = params.get('E_d') or 0

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

        # Calculate starting positions # TODO: implement inverse energy functions
        q_0 = np.array([0] * self.num_dof)
        dq_0 = np.array([0] * self.num_dof)

        # Initialize current positional values
        self.x_0 = np.append(q_0, dq_0)
        self.x_cur = self.x_0

        self.q_0 = q_0
        self.q_cur = self.q_0

        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = self.p_0

        # Returns the initial position in joint coordinates
        return q_0

    def solve(self, t: float):
        raise NotImplementedError

    def forward_kins(self,
                     params: dict = None):
        raise NotImplementedError

    def inverse_kins(self, params: dict = None):
        raise NotImplementedError

    def get_joint_state(self):
        q, dq = np.split(self.x_cur, 2)
        return [q, dq]

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

    def get_cartesian_state(self):
        # Get joint positions and speeds
        q = np.asarray(self.x_cur[:self.num_dof])
        dq = np.asarray(self.x_cur[self.num_dof:])

        # TODO: implement this functions in backend
        # Calculate the cart. positions and speeds
        p = np.asarray(self.backend.forward_kinematics(q)[0:-1])
        v = np.asarray((self.backend.jacobian(q) @ dq)[0:-1])

        return np.asarray([p, v])

    def get_link_cartesian_positions(self):
        # Get joint positions
        q = np.asarray(self.x_cur[0:self.num_dof])

        # TODO: implement this functions in backend
        # Calculate the link positions
        p = self.backend.forward_kinematics_for_each_link(q)
        p = np.asarray(p[:, 0:-1])

        return p

    def get_energies(self):
        # Get joint positions and speeds
        q = np.asarray(self.x_cur[:self.num_dof])
        dq = np.asarray(self.x_cur[self.num_dof:])

        # Calculate energies
        E_pot = np.asarray([self.backend.potential_energy(q, absolute=False)])
        E_kin = np.asarray([self.backend.kinetic_energy(q, dq)])

        return np.asarray([E_pot, E_kin])
