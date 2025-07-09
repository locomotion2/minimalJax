from abc import ABC
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from src.models.base_models import BaseModel, JointsGenerator, PolarGenerator

# Set the default device to CPU to avoid potential device mismatches with solve_ivp
# You can change this to 'gpu' or 'tpu' if your entire pipeline is on that device.
jax.config.update('jax_platform_name', 'cpu')

class CPG(BaseModel, ABC):
    """
    This class is not fully implemented for JAX as its logic
    is complex and contains coiling detection which is stateful.
    It's left here as a template but is not used in the main training loop.
    The primary generators used are GPG and SPG.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.omega_cur = jnp.zeros(self.num_dof)
        self.mu_cur = 0
        self.coils = 0
        params_cpg = jnp.array([0, 1])
        self.params_traj = jnp.array([params_cpg])

    @staticmethod
    @jax.jit
    def eqs_motion(t, x, params):
        """JAX-compatible equations of motion for the CPG."""
        mu = params['mu']
        omega = params['omega']
        rho = x[0] ** 2 + x[1] ** 2
        circleDist = mu ** 2 - rho
        dx1 = -x[1] * omega + x[0] * circleDist
        dx2 = x[0] * omega + x[1] * circleDist
        return jnp.array([dx1, dx2])

    def make_eqs_motion(self, params: dict = None):
        # This function now returns the static, JIT-compiled eqs_motion method
        return self.eqs_motion

    def restart(self, params: dict = None):
        p_0 = super().restart(params)
        params_cpg = jnp.array([0, 1])
        self.params_traj = jnp.array([params_cpg])
        self.coils = 0
        return p_0

    def step(self, params: dict = None):
        # Pass the static JAX-compatible function to the simulator
        params['eqs'] = CPG.eqs_motion
        super().step(params)

    def update_trajectories(self, params: dict = None):
        """
        Note: This method is stateful and not JIT-compatible without refactoring.
        """
        super().update_trajectories()
        self.mu_cur = params['mu']
        self.omega_cur = params['omega']
        params_val = jnp.array([self.omega_cur, self.mu_cur])
        self.params_traj = jnp.append(self.params_traj, [params_val], axis=0)

    def detect_coiling(self):
        """
        Note: This method uses standard Python conditionals and updates object state,
        making it incompatible with @jax.jit.
        """
        if len(self.x_traj) < 2:
            return 0
        x_new = self.x_cur
        x_old = self.x_traj[-2]
        new_angle = jnp.arctan2(x_new[1], x_new[0])
        old_angle = jnp.arctan2(x_old[1], x_old[0])
        if (-jnp.pi / 2 > new_angle) and (old_angle > jnp.pi / 2):
            self.coils += 1
        elif (-jnp.pi / 2 > old_angle) and (new_angle > jnp.pi / 2):
            self.coils -= 1
        return self.coils

    def select_initial(self, params: dict = None):
        self.x_0 = params.get('x_0', self.x_0)
        self.x_cur = jnp.asarray(self.x_0)
        self.q_0 = jnp.zeros(self.num_dof)
        self.q_cur = jnp.asarray(self.q_0)
        # tolist() is okay here as it's part of the non-JIT setup phase
        self.p_0 = self.get_link_cartesian_positions().tolist()
        self.p_cur = jnp.asarray(self.p_0)

    def get_cartesian_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}
        p = self.x_cur
        v = CPG.eqs_motion(0, self.x_cur, params)
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

# --- Helper functions for GPG ---

@jax.jit
def _gpg_eqs_motion_multi_dof(x, mu, omega, num_dof):
    """Helper for GPG multi-DOF dynamics."""
    q = jnp.asarray(x, dtype=float).flatten()
    psi = mu ** 2 - jnp.linalg.norm(q) ** 2
    root = jnp.ones((num_dof, num_dof))
    upper = jnp.multiply(jnp.triu(root, 1), omega)
    lower = jnp.multiply(jnp.tril(root, -1), -omega)
    diag_values = jnp.full(num_dof, psi)
    diag = jnp.diag(diag_values)
    A = upper + diag + lower
    return A @ q

@jax.jit
def _gpg_eqs_motion_single_dof(x, mu, omega, num_dof):
    """Helper for GPG single-DOF dynamics."""
    return omega

@jax.jit
def gpg_eqs_motion_wrapper(t, x, params):
    """JAX-compatible wrapper for GPG equations of motion."""
    mu = params['mu']
    omega = params['omega']
    num_dof = params['num_dof']
    return jax.lax.cond(
        num_dof != 1,
        _gpg_eqs_motion_multi_dof,
        _gpg_eqs_motion_single_dof,
        x, mu, omega, num_dof
    )

class GPG(JointsGenerator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        if self.num_dof != 1:
            self.omega_cur = jnp.zeros(self.num_dof - 1)
            self.omega_past = jnp.zeros(self.num_dof - 1)
            self.mu_cur = jnp.zeros(1)
        else:
            self.omega_cur = jnp.zeros(1)
            self.omega_past = jnp.zeros(1)
            self.mu_cur = None

    def make_eqs_motion(self, params: dict = None):
        return gpg_eqs_motion_wrapper

    def step(self, params: dict = None):
        params['num_dof'] = self.num_dof
        super().step(params=params)
        self.omega_cur = jnp.asarray(params['omega'])
        self.mu_cur = jnp.asarray(params['mu']) if self.mu_cur is not None else None

    def update_trajectories(self, params: dict = None):
        change = jnp.where(jnp.sign(self.omega_cur) != jnp.sign(self.omega_past), 1.0, 0.0)
        self.omega_past = self.omega_cur
        mu_val = self.mu_cur if self.mu_cur is not None else jnp.array([])
        params['input'] = jnp.concatenate(
            [jnp.atleast_1d(self.omega_cur), jnp.atleast_1d(mu_val), jnp.atleast_1d(change)]
        )
        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        q = jnp.asarray(self.x_cur).flatten()
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        dq = jnp.asarray(gpg_eqs_motion_wrapper(0, q, params)).flatten()
        return [q, dq]

# --- Helper functions for SPG ---

@jax.jit
def _spg_eqs_motion_multi_dof(x, mu, omega):
    """Helper for SPG multi-DOF dynamics."""
    r = x[-1]
    dr = (mu ** 2 - r ** 2) * r
    dphi = omega
    return jnp.asarray([dphi, dr], dtype=float).flatten()

@jax.jit
def _spg_eqs_motion_single_dof(x, mu, omega):
    """Helper for SPG single-DOF dynamics."""
    return omega

@jax.jit
def spg_eqs_motion_wrapper(t, x, params):
    """JAX-compatible wrapper for SPG equations of motion."""
    mu = params['mu']
    omega = params['omega']
    num_dof = params['num_dof']
    return jax.lax.cond(
        num_dof != 1,
        _spg_eqs_motion_multi_dof,
        _spg_eqs_motion_single_dof,
        x, mu, omega
    )

@jax.jit
def _polar_to_joints_multi_dof(x_cur, mu, omega, num_dof):
    """Helper for polar to joint conversion (multi-DOF)."""
    phi = x_cur[0:num_dof - 1]
    r = x_cur[-1]
    q = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])
    psi = mu ** 2 - r ** 2
    root = jnp.ones((num_dof, num_dof))
    upper = jnp.multiply(jnp.triu(root, 1), -omega)
    lower = jnp.multiply(jnp.tril(root, -1), omega)
    diag_values = jnp.full(num_dof, psi)
    diag = jnp.diag(diag_values)
    A = upper + diag + lower
    dq = A @ q
    return jnp.asarray(q, dtype=float).flatten(), jnp.asarray(dq, dtype=float).flatten()

@jax.jit
def _polar_to_joints_single_dof(x_cur, mu, omega, num_dof):
    """Helper for polar to joint conversion (single-DOF)."""
    q = x_cur[0]
    dq = omega
    return jnp.asarray([q], dtype=float).flatten(), jnp.asarray([dq], dtype=float).flatten()

class SPG(PolarGenerator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        if self.num_dof != 1:
            self.omega_cur = jnp.zeros(self.num_dof - 1)
            self.omega_past = jnp.zeros(self.num_dof - 1)
            self.mu_cur = jnp.zeros(1)
        else:
            self.omega_cur = jnp.zeros(1)
            self.omega_past = jnp.zeros(1)
            self.mu_cur = None

    def make_eqs_motion(self, params: dict = None):
        return spg_eqs_motion_wrapper

    def step(self, params: dict = None):
        params['num_dof'] = self.num_dof
        super().step(params=params)
        self.omega_cur = jnp.asarray(params['omega'])
        self.mu_cur = jnp.asarray(params['mu']) if self.mu_cur is not None else None

    def polar_to_joints(self, state: jnp.ndarray = None):
        """Converts polar coordinates to joint space using JAX-compatible logic."""
        mu = self.mu_cur if self.mu_cur is not None else 0.0
        omega = self.omega_cur
        num_dof = self.num_dof
        q, dq = jax.lax.cond(
            num_dof != 1,
            _polar_to_joints_multi_dof,
            _polar_to_joints_single_dof,
            self.x_cur, mu, omega, num_dof
        )
        return [q, dq]

    def joints_to_polar(self, joints: jnp.ndarray = None):
        """Converts joint space to polar coordinates using JAX-compatible logic."""
        def multi_dof_conversion(j):
            r = jnp.linalg.norm(j)
            phi = jnp.arctan2(j[1], j[0])
            return jnp.array([phi, r])
        def single_dof_conversion(j):
            return j
        return jax.lax.cond(
            self.num_dof != 1,
            multi_dof_conversion,
            single_dof_conversion,
            joints
        )

    def update_trajectories(self, params: dict = None):
        change = jnp.where(jnp.sign(self.omega_cur) != jnp.sign(self.omega_past), 1.0, 0.0)
        self.omega_past = self.omega_cur
        mu_val = self.mu_cur if self.mu_cur is not None else jnp.array([])
        params['input'] = jnp.concatenate(
            [jnp.atleast_1d(self.omega_cur), jnp.atleast_1d(mu_val), jnp.atleast_1d(change)]
        )
        super().update_trajectories(params)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        return self.polar_to_joints()