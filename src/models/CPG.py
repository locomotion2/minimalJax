from abc import ABC

import numpy as np
from src.models.base_models import BaseModel
from src.models.base_models import JointsGenerator
from src.models.base_models import PolarGenerator


class CPG(BaseModel, ABC):
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

        self.p_0 = self.get_link_cartesian_positions().tolist()  # TODO: Change the
        # function, maybe it solves the problem with tolist()
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


class GPG(JointsGenerator):

    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the RL vars
        if self.num_dof != 1:
            self.omega_cur = np.asarray([0] * (self.num_dof - 1))
            self.omega_past = np.asarray([0] * (self.num_dof - 1))
            self.mu_cur = np.asarray([0])
        else:
            self.omega_cur = np.asarray([0])
            self.omega_past = np.asarray([0])
            self.mu_cur = None

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

            # Define the state matrix
            if num_dof != 1:
                # Calculate help variables
                q = np.asarray([x], dtype=float).flatten()
                psi = mu ** 2 - np.linalg.norm(q) ** 2

                # Calculate state matrix
                root = np.ones((num_dof, num_dof))
                upper = np.multiply(np.triu(root, 1), omega)
                lower = np.multiply(np.tril(root, -1), -omega)
                diag = np.diag(np.diag(np.full((num_dof, num_dof), psi)))
                A = upper + diag + lower

                dq = A @ q
            else:
                dq = omega

            return dq

        return eqs_motion

    def step(self,
             params: dict = None):  # Todo: check if it's still necesary to have this appart from the other class
        # Run the integration
        params['num_dof'] = self.num_dof
        super().step(params=params)

        # Update current variables
        self.omega_cur = np.asarray(params['omega'])
        self.mu_cur = np.asarray(params['mu'])

    def update_trajectories(self, params: dict = None):
        # Update trajectories
        change = 0
        if np.sign(self.omega_cur) != np.sign(self.omega_past):
            change = 1
        self.omega_past = self.omega_cur
        # params['input'] = np.asarray([self.omega_cur, self.mu_cur, change])
        params['input'] = np.concatenate(
            [self.omega_cur, self.mu_cur, np.asarray([change])], axis=0)

        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        q = np.asarray([self.x_cur]).flatten()
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        dq = np.asarray([self.eqs_motion(0, q, params)]).flatten()

        return [q, dq]


class SPG(PolarGenerator):

    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the RL vars
        if self.num_dof != 1:
            self.omega_cur = np.asarray([0] * (self.num_dof - 1))
            self.omega_past = np.asarray([0] * (self.num_dof - 1))
            self.mu_cur = np.asarray([0])
        else:
            self.omega_cur = np.asarray([0])
            self.omega_past = np.asarray([0])
            self.mu_cur = None

    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            # Handle the inputs
            mu = params['mu']
            omega = params['omega']
            num_dof = params['num_dof']

            if num_dof != 1:
                r = x[-1]

                # Calculate state derivs
                dr = (mu ** 2 - r ** 2) * r
                dphi = omega

                return np.asarray([dphi, dr], dtype=float).flatten()
            else:
                return omega

        return eqs_motion

    def step(self,
             params: dict = None):  # Todo: check if it's still necesary to have this appart from the other class
        # Run the integration
        params['num_dof'] = self.num_dof
        super().step(params=params)

        # Update current variables
        self.omega_cur = np.asarray(params['omega'])
        self.mu_cur = np.asarray(params['mu'])

    def polar_to_joints(self, state: np.ndarray = None):
        # Handle inputs
        mu = self.mu_cur
        omega = self.omega_cur
        num_dof = self.num_dof

        # Define the state matrix
        if num_dof != 1:
            phi = self.x_cur[0:self.num_dof - 1]
            r = self.x_cur[-1]

            # Calculate help variables
            q = np.asarray([r * np.cos(phi),
                            r * np.sin(phi)])

            # Calculate state matrix
            psi = mu ** 2 - r ** 2
            root = np.ones((num_dof, num_dof))
            upper = np.multiply(np.triu(root, 1), -omega)
            lower = np.multiply(np.tril(root, -1), omega)
            diag = np.diag(np.diag(np.full((num_dof, num_dof), psi)))
            A = upper + diag + lower

            dq = A @ q
        else:
            q = self.x_cur[0]
            dq = omega

        q = np.asarray([q], dtype=float).flatten()
        dq = np.asarray([dq], dtype=float).flatten()

        return [q, dq]

    def joints_to_polar(self, joints: np.ndarray = None):
        if self.num_dof != 1:
            r = np.linalg.norm(joints)
            phi = np.arctan2(joints[1], joints[0])

            return np.asarray([phi, r])
        else:
            return joints

    def update_trajectories(self, params: dict = None):
        # Update trajectories
        change = 0
        if np.sign(self.omega_cur) != np.sign(self.omega_past):
            change = 1
        self.omega_past = self.omega_cur
        # params['input'] = np.asarray([self.omega_cur, self.mu_cur, change])
        params['input'] = np.concatenate(
            [self.omega_cur, self.mu_cur, np.asarray([change])], axis=0)

        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        out = self.polar_to_joints(self.x_cur)
        return out
