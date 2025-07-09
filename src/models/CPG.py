from abc import ABC

import numpy as np # Keep for solve_ivp
import jax
import jax.numpy as jnp
from src.models.base_models import BaseModel
from src.models.base_models import JointsGenerator
from src.models.base_models import PolarGenerator


class CPG(BaseModel, ABC):
    def __init__(self, params: dict = None):
        # Todo: set the sizes of this variables correctly
        self.omega_cur = jnp.asarray([0] * self.num_dof) # Use jnp.asarray
        self.mu_cur = 0
        self.coils = 0

        params_cpg = jnp.asarray([0, 1]) # Use jnp.asarray
        self.params_traj = jnp.asarray([params_cpg]) # Use jnp.asarray
        super().__init__(params)

    @jax.jit
    def make_eqs_motion(self, params: dict = None):
        def eqs_motion(t, x, params):
            mu = params['mu']
            omega = params['omega']

            rho = x[0] ** 2 + x[1] ** 2
            circleDist = mu ** 2 - rho

            dx1 = -x[1] * omega + x[0] * circleDist
            dx2 = x[0] * omega + x[1] * circleDist

            return jnp.asarray([dx1, dx2]) # Use jnp.asarray

        return eqs_motion

    def restart(self, params: dict = None):
        p_0 = super().restart(params)
        params_cpg = jnp.asarray([0, 1])  # Todo: should this be changed? # Use jnp.asarray
        self.params_traj = jnp.asarray([params_cpg]) # Use jnp.asarray
        self.coils = 0

        return p_0

    def step(self, params: dict = None):
        params['eqs_motion'] = self.eqs_motion
        super().step(params)

    def update_trajectories(self, params: dict = None):
        super().update_trajectories()
        self.mu_cur = params['mu']
        self.omega_cur = params['omega']
        params_val = jnp.asarray([self.omega_cur, self.mu_cur]) # Use jnp.asarray
        self.params_traj = jnp.append(self.params_traj, [params_val], axis=0) # Use jnp.append

    def detect_coiling(self):
        x_new = self.x_cur
        x_old = self.x_traj[-2]

        new_angle = jnp.arctan2(x_new[1], x_new[0]) # Use jnp.arctan2
        old_angle = jnp.arctan2(x_old[1], x_old[0]) # Use jnp.arctan2
        if (-jnp.pi / 2 > new_angle) and (old_angle > jnp.pi / 2): # Use jnp.pi
            self.coils += 1
        elif (-jnp.pi / 2 > old_angle) and (new_angle > jnp.pi / 2): # Use jnp.pi
            self.coils -= 1

        return self.coils

    def select_initial(self, params: dict = None):
        self.x_0 = params.get('x_0', self.x_0)
        self.x_cur = jnp.asarray(self.x_0) # Use jnp.asarray

        self.q_0 = jnp.asarray([0] * self.num_dof) # Use jnp.asarray
        self.q_cur = jnp.asarray(self.q_0) # Use jnp.asarray

        self.p_0 = self.get_link_cartesian_positions().tolist()  # TODO: Change the
        # function, maybe it solves the problem with tolist()
        self.p_cur = jnp.asarray(self.p_0) # Use jnp.asarray

    def get_cartesian_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}

        @jax.jit
        def eqs_motion_jit(t, x, params): # JIT this inner function
            mu = params['mu']
            omega = params['omega']

            rho = x[0] ** 2 + x[1] ** 2
            circleDist = mu ** 2 - rho

            dx1 = -x[1] * omega + x[0] * circleDist
            dx2 = x[0] * omega + x[1] * circleDist

            return jnp.asarray([dx1, dx2]) # Use jnp.asarray

        p = self.x_cur
        v = eqs_motion_jit(0, self.x_cur, params)

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
            self.omega_cur = jnp.asarray([0] * (self.num_dof - 1)) # Use jnp.asarray
            self.omega_past = jnp.asarray([0] * (self.num_dof - 1)) # Use jnp.asarray
            self.mu_cur = jnp.asarray([0]) # Use jnp.asarray
        else:
            self.omega_cur = jnp.asarray([0]) # Use jnp.asarray
            self.omega_past = jnp.asarray([0]) # Use jnp.asarray
            self.mu_cur = None

    @jax.jit
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
                q = jnp.asarray([x], dtype=float).flatten() # Use jnp.asarray
                psi = mu ** 2 - jnp.linalg.norm(q) ** 2 # Use jnp.linalg.norm

                # Calculate state matrix
                root = jnp.ones((num_dof, num_dof)) # Use jnp.ones
                upper = jnp.multiply(jnp.triu(root, 1), omega) # Use jnp.multiply, jnp.triu
                lower = jnp.multiply(jnp.tril(root, -1), -omega) # Use jnp.multiply, jnp.tril
                diag = jnp.diag(jnp.diag(jnp.full((num_dof, num_dof), psi))) # Use jnp.diag, jnp.full
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
        self.omega_cur = jnp.asarray(params['omega']) # Use jnp.asarray
        self.mu_cur = jnp.asarray(params['mu']) # Use jnp.asarray

    def update_trajectories(self, params: dict = None):
        # Update trajectories
        change = 0
        if jnp.sign(self.omega_cur) != jnp.sign(self.omega_past): # Use jnp.sign
            change = 1
        self.omega_past = self.omega_cur
        # params['input'] = jnp.asarray([self.omega_cur, self.mu_cur, change])
        params['input'] = jnp.concatenate( # Use jnp.concatenate
            [self.omega_cur, self.mu_cur, jnp.asarray([change])], axis=0) # Use jnp.asarray

        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        q = jnp.asarray([self.x_cur]).flatten() # Use jnp.asarray
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        dq = jnp.asarray([self.eqs_motion(0, q, params)]).flatten() # Use jnp.asarray

        return [q, dq]


class SPG(PolarGenerator):

    def __init__(self, params: dict = None):
        super().__init__(params)

        # Define the RL vars
        if self.num_dof != 1:
            self.omega_cur = jnp.asarray([0] * (self.num_dof - 1)) # Use jnp.asarray
            self.omega_past = jnp.asarray([0] * (self.num_dof - 1)) # Use jnp.asarray
            self.mu_cur = jnp.asarray([0]) # Use jnp.asarray
        else:
            self.omega_cur = jnp.asarray([0]) # Use jnp.asarray
            self.omega_past = jnp.asarray([0]) # Use jnp.asarray
            self.mu_cur = None

    @jax.jit
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

                return jnp.asarray([dphi, dr], dtype=float).flatten() # Use jnp.asarray
            else:
                return omega

        return eqs_motion

    def step(self,
             params: dict = None):  # Todo: check if it's still necesary to have this appart from the other class
        # Run the integration
        params['num_dof'] = self.num_dof
        super().step(params=params)

        # Update current variables
        self.omega_cur = jnp.asarray(params['omega']) # Use jnp.asarray
        self.mu_cur = jnp.asarray(params['mu']) # Use jnp.asarray

    @jax.jit
    def polar_to_joints(self, state: jnp.ndarray = None): # Parameter type hint changed to jnp.ndarray
        # Handle inputs
        mu = self.mu_cur
        omega = self.omega_cur
        num_dof = self.num_dof

        # Define the state matrix
        if num_dof != 1:
            phi = self.x_cur[0:self.num_dof - 1]
            r = self.x_cur[-1]

            # Calculate help variables
            q = jnp.asarray([r * jnp.cos(phi), # Use jnp.asarray, jnp.cos
                            r * jnp.sin(phi)]) # Use jnp.sin

            # Calculate state matrix
            psi = mu ** 2 - r ** 2
            root = jnp.ones((num_dof, num_dof)) # Use jnp.ones
            upper = jnp.multiply(jnp.triu(root, 1), -omega) # Use jnp.multiply, jnp.triu
            lower = jnp.multiply(jnp.tril(root, -1), omega) # Use jnp.multiply, jnp.tril
            diag = jnp.diag(jnp.diag(jnp.full((num_dof, num_dof), psi))) # Use jnp.diag, jnp.full
            A = upper + diag + lower

            dq = A @ q
        else:
            q = self.x_cur[0]
            dq = omega

        q = jnp.asarray([q], dtype=float).flatten() # Use jnp.asarray
        dq = jnp.asarray([dq], dtype=float).flatten() # Use jnp.asarray

        return [q, dq]

    @jax.jit
    def joints_to_polar(self, joints: jnp.ndarray = None): # Parameter type hint changed to jnp.ndarray
        if self.num_dof != 1:
            r = jnp.linalg.norm(joints) # Use jnp.linalg.norm
            phi = jnp.arctan2(joints[1], joints[0]) # Use jnp.arctan2

            return jnp.asarray([phi, r]) # Use jnp.asarray
        else:
            return joints

    def update_trajectories(self, params: dict = None):
        # Update trajectories
        change = 0
        if jnp.sign(self.omega_cur) != jnp.sign(self.omega_past): # Use jnp.sign
            change = 1
        self.omega_past = self.omega_cur
        # params['input'] = jnp.asarray([self.omega_cur, self.mu_cur, change])
        params['input'] = jnp.concatenate( # Use jnp.concatenate
            [self.omega_cur, self.mu_cur, jnp.asarray([change])], axis=0) # Use jnp.asarray

        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        out = self.polar_to_joints(self.x_cur)
        return out