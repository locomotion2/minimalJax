from abc import ABC
import jax
import jax.numpy as jnp

# Assuming the base models are in this path.
# These base classes are expected to provide the .simulate() method
# and define the basic structure.
from src.models.base_models import BaseModel, JointsGenerator, PolarGenerator


class CPG(BaseModel, ABC):
    """
    Cartesian Central Pattern Generator (CPG).

    This model generates rhythmic patterns in Cartesian space using a limit cycle oscillator.
    The core dynamics are implemented in a JAX-jittable function for performance.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        # Initialize state variables as JAX arrays
        self.omega_cur = jnp.zeros(self.num_dof)
        self.mu_cur = 0.0
        self.coils = 0

        # Pre-compile the JITted function for the equations of motion
        self.eqs_motion = self._make_eqs_motion()

    @staticmethod
    @jax.jit
    def _make_eqs_motion():
        """Creates a JIT-compiled function for the oscillator dynamics."""
        def eqs_motion(t, x, params):
            """CPG equations of motion."""
            mu = params['mu']
            omega = params['omega']
            rho = x[0] ** 2 + x[1] ** 2
            circle_dist = mu ** 2 - rho
            dx1 = -x[1] * omega + x[0] * circle_dist
            dx2 = x[0] * omega + x[1] * circle_dist
            return jnp.asarray([dx1, dx2])
        return eqs_motion

    def restart(self, params: dict = None):
        """
        Resets the model to its initial state. This is a stateful operation
        and is not JIT-compiled.
        """
        # --- Logic from BaseModel.restart ---
        self.t_0 = params.get('t_0', self.t_0)
        p_0 = self.select_initial(params)
        self.t_cur = self.t_0
        self.E_0 = sum(self.get_energies())
        self.E_cur = self.E_0
        self.t_traj = jnp.asarray([self.t_cur])
        self.x_traj = jnp.asarray([self.x_cur])
        self.q_traj = jnp.asarray([self.q_cur])
        self.p_traj = jnp.asarray([self.p_cur])
        self.E_traj = jnp.asarray([self.E_cur])

        # --- Logic from CPG.restart ---
        params_cpg = jnp.array([0.0, 1.0])
        self.params_traj = jnp.asarray([params_cpg])
        self.coils = 0
        return p_0

    def step(self, params: dict = None):
        """
        Advances the simulation by one time step. This is a stateful wrapper
        that calls the pure, JIT-compiled dynamics function.
        """
        # --- Logic from BaseModel.step ---
        t_final = params.get('t_final', self.t_cur + self.delta_t)
        ts = jnp.asarray([self.t_cur, t_final])

        # Simulate using the JIT-compiled dynamics
        self.x_cur = self.simulate({
            'eqs': self.eqs_motion,
            'eqs_params': params,
            'ts': ts,
            'x_0': self.x_cur
        })

        # Update state variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final

    def update_trajectories(self, params: dict = None):
        """
        Appends the current state to the trajectory history.
        Note: jnp.append is used for simplicity, but for very long simulations
        in a JIT context, pre-allocation would be more performant.
        """
        # --- Logic from BaseModel.update_trajectories (Corrected) ---
        self.t_traj = jnp.append(self.t_traj, self.t_cur)
        self.x_traj = jnp.append(self.x_traj, jnp.expand_dims(self.x_cur, axis=0), axis=0)
        self.q_traj = jnp.append(self.q_traj, jnp.expand_dims(self.q_cur, axis=0), axis=0)
        self.p_traj = jnp.append(self.p_traj, jnp.expand_dims(self.p_cur, axis=0), axis=0)
        self.E_traj = jnp.append(self.E_traj, self.E_cur)

        # --- Logic from CPG.update_trajectories ---
        self.mu_cur = params['mu']
        self.omega_cur = params['omega']
        params_val = jnp.asarray([self.omega_cur, self.mu_cur])
        self.params_traj = jnp.append(self.params_traj, jnp.expand_dims(params_val, axis=0), axis=0)

    def detect_coiling(self):
        """Detects full rotations of the oscillator."""
        if len(self.x_traj) < 2:
            return self.coils
        x_new = self.x_cur
        x_old = self.x_traj[-2]
        new_angle = jnp.arctan2(x_new[1], x_new[0])
        old_angle = jnp.arctan2(x_old[1], x_old[0])
        if (new_angle < -jnp.pi / 2) and (old_angle > jnp.pi / 2):
            self.coils += 1
        elif (old_angle < -jnp.pi / 2) and (new_angle > jnp.pi / 2):
            self.coils -= 1
        return self.coils

    def select_initial(self, params: dict = None):
        """Sets the initial conditions for the model."""
        self.x_0 = params.get('x_0', self.x_0)
        self.x_cur = jnp.asarray(self.x_0)
        self.q_0 = jnp.zeros(self.num_dof)
        self.q_cur = jnp.asarray(self.q_0)
        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = jnp.asarray(self.p_0)
        return self.p_0

    # --- Getter and other methods ---
    def get_cartesian_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}
        v = self.eqs_motion(0, self.x_cur, params)
        return [self.x_cur, v]

    def get_link_cartesian_positions(self):
        return self.get_cartesian_state()[0]

    def get_state_traj(self):
        return self.x_traj

    def get_temporal_traj(self):
        return self.t_traj

    def get_parametric_traj(self):
        return self.params_traj

    def get_energies(self):
        return [0, 0]

    # --- Abstract methods that should be implemented if needed ---
    def solve(self, t): pass
    def inverse_kins(self, params: dict = None): pass
    def get_joint_state(self): pass


class GPG(JointsGenerator):
    """
    Joint-space Pattern Generator (GPG).

    Generates rhythmic patterns directly in the joint space. The dynamics
    are defined in a JAX-jittable function, chosen based on the number of DoF.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        # Initialization logic is static and runs once, so Python `if` is fine.
        if self.num_dof != 1:
            self.omega_cur = jnp.zeros(self.num_dof - 1)
            self.omega_past = jnp.zeros(self.num_dof - 1)
        else:
            self.omega_cur = jnp.zeros(1)
            self.omega_past = jnp.zeros(1)
        self.mu_cur = jnp.zeros(1)
        # Create the appropriate JITted function based on static parameter `num_dof`.
        self.eqs_motion = self._make_eqs_motion()

    def _make_eqs_motion(self):
        """
        Factory for the JIT-compiled dynamics function.
        A standard Python `if` is used here because `self.num_dof` is a static
        parameter that doesn't change during the simulation.
        """
        if self.num_dof != 1:
            num_dof = self.num_dof  # Capture as a static value for the jit
            @jax.jit
            def eqs_motion(t, x, params):
                """GPG equations of motion for num_dof > 1."""
                mu, omega = params['mu'], params['omega']
                q = jnp.asarray(x, dtype=float).flatten()
                psi = mu ** 2 - jnp.linalg.norm(q) ** 2
                root = jnp.ones((num_dof, num_dof))
                upper = jnp.multiply(jnp.triu(root, 1), omega)
                lower = jnp.multiply(jnp.tril(root, -1), -omega)
                diag = jnp.diag(jnp.full((num_dof,), psi.squeeze()))
                A = upper + diag + lower
                return A @ q
        else:
            @jax.jit
            def eqs_motion(t, x, params):
                """GPG equations of motion for num_dof == 1."""
                return params['omega']
        return eqs_motion

    def restart(self, params: dict = None):
        """Resets the model to its initial state."""
        # --- Logic from BaseModel.restart ---
        self.t_0 = params.get('t_0', self.t_0)
        p_0 = self.select_initial(params)
        self.t_cur = self.t_0
        self.E_0 = sum(self.get_energies())
        self.E_cur = self.E_0
        self.t_traj = jnp.asarray([self.t_cur])
        self.x_traj = jnp.asarray([self.x_cur])
        self.q_traj = jnp.asarray([self.q_cur])
        self.p_traj = jnp.asarray([self.p_cur])
        self.E_traj = jnp.asarray([self.E_cur])
        # --- Logic from JointsGenerator.restart ---
        self.params_traj = jnp.zeros((1, self.num_dof + 1))
        return p_0

    def step(self, params: dict = None):
        """Advances the simulation by one time step."""
        # --- Logic from BaseModel.step ---
        t_final = params.get('t_final', self.t_cur + self.delta_t)
        ts = jnp.asarray([self.t_cur, t_final])
        self.x_cur = self.simulate({
            'eqs': self.eqs_motion,
            'eqs_params': params,
            'ts': ts,
            'x_0': self.x_cur
        })
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final
        # --- Logic from GPG.step ---
        self.omega_cur = jnp.asarray(params['omega'])
        self.mu_cur = jnp.asarray(params['mu'])

    def update_trajectories(self, params: dict = None):
        """Appends the current state to the trajectory history."""
        # --- Logic from GPG.update_trajectories ---
        change = jnp.any(jnp.sign(self.omega_cur) != jnp.sign(self.omega_past)).astype(float)
        self.omega_past = self.omega_cur
        params_input = jnp.concatenate([
            jnp.atleast_1d(self.omega_cur),
            jnp.atleast_1d(self.mu_cur),
            jnp.atleast_1d(change)
        ])

        # --- Logic from BaseModel.update_trajectories (Corrected) ---
        self.t_traj = jnp.append(self.t_traj, self.t_cur)
        self.x_traj = jnp.append(self.x_traj, jnp.expand_dims(self.x_cur, axis=0), axis=0)
        self.q_traj = jnp.append(self.q_traj, jnp.expand_dims(self.q_cur, axis=0), axis=0)
        self.p_traj = jnp.append(self.p_traj, jnp.expand_dims(self.p_cur, axis=0), axis=0)
        self.E_traj = jnp.append(self.E_traj, self.E_cur)

        # --- Logic from JointsGenerator.update_trajectories ---
        self.params_traj = jnp.append(self.params_traj, jnp.expand_dims(params_input, axis=0), axis=0)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        """Returns the current joint state [q, dq]."""
        q = jnp.asarray(self.x_cur).flatten()
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}
        dq = self.eqs_motion(0, q, params).flatten()
        return [q, dq]


class SPG(PolarGenerator):
    """
    Polar-space Pattern Generator (SPG).

    Generates rhythmic patterns using polar coordinates (radius and angle).
    The dynamics are defined in a JAX-jittable function.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        if self.num_dof != 1:
            self.omega_cur = jnp.zeros(self.num_dof - 1)
            self.omega_past = jnp.zeros(self.num_dof - 1)
        else:
            self.omega_cur = jnp.zeros(1)
            self.omega_past = jnp.zeros(1)
        self.mu_cur = jnp.zeros(1)
        self.eqs_motion = self._make_eqs_motion()

    def _make_eqs_motion(self):
        """Factory for the JIT-compiled dynamics function."""
        if self.num_dof != 1:
            @jax.jit
            def eqs_motion(t, x, params):
                """SPG equations of motion for num_dof > 1."""
                mu, omega = params['mu'], params['omega']
                r = x[-1]
                dr = (mu ** 2 - r ** 2) * r
                dphi = omega
                # Note: This assumes dphi is a scalar or matches the shape of phi.
                # If omega has num_dof-1 elements, dphi will too.
                return jnp.append(dphi, dr)
        else:
            @jax.jit
            def eqs_motion(t, x, params):
                """SPG equations of motion for num_dof == 1."""
                return params['omega']
        return eqs_motion

    def restart(self, params: dict = None):
        """Resets the model to its initial state."""
        # --- Logic from BaseModel.restart & PolarGenerator.restart ---
        self.t_0 = params.get('t_0', self.t_0)
        p_0 = self.select_initial(params)
        self.t_cur = self.t_0
        self.E_0 = sum(self.get_energies())
        self.E_cur = self.E_0
        self.t_traj = jnp.asarray([self.t_cur])
        self.x_traj = jnp.asarray([self.x_cur])
        self.q_traj = jnp.asarray([self.q_cur])
        self.p_traj = jnp.asarray([self.p_cur])
        self.E_traj = jnp.asarray([self.E_cur])
        self.params_traj = jnp.zeros((1, self.num_dof + 1))
        return p_0

    def step(self, params: dict = None):
        """Advances the simulation by one time step."""
        # --- Logic from BaseModel.step ---
        t_final = params.get('t_final', self.t_cur + self.delta_t)
        ts = jnp.asarray([self.t_cur, t_final])
        self.x_cur = self.simulate({
            'eqs': self.eqs_motion,
            'eqs_params': params,
            'ts': ts,
            'x_0': self.x_cur
        })
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final
        # --- Logic from SPG.step ---
        self.omega_cur = jnp.asarray(params['omega'])
        self.mu_cur = jnp.asarray(params['mu'])

    def polar_to_joints(self, state: jnp.ndarray = None):
        """Converts polar state to joint state [q, dq]."""
        if self.num_dof != 1:
            x_cur = self.x_cur if state is None else state
            phi = x_cur[0:self.num_dof - 1]
            r = x_cur[-1]
            q = jnp.asarray([r * jnp.cos(phi), r * jnp.sin(phi)]).flatten()
            
            # Calculate dq
            psi = self.mu_cur ** 2 - r ** 2
            root = jnp.ones((self.num_dof, self.num_dof))
            upper = jnp.multiply(jnp.triu(root, 1), -self.omega_cur)
            lower = jnp.multiply(jnp.tril(root, -1), self.omega_cur)
            diag = jnp.diag(jnp.full((self.num_dof,), psi.squeeze()))
            A = upper + diag + lower
            dq = (A @ q).flatten()
            return [q, dq]
        else:
            q = self.x_cur[0] if state is None else state[0]
            dq = self.omega_cur
            return [jnp.atleast_1d(q), jnp.atleast_1d(dq)]

    def joints_to_polar(self, joints: jnp.ndarray = None):
        """Converts joint state to polar state."""
        if self.num_dof != 1:
            r = jnp.linalg.norm(joints)
            # Ensure phi is always calculated for safety, even if r is 0
            phi = jnp.arctan2(joints[1], joints[0])
            return jnp.asarray([phi, r])
        else:
            return joints

    def update_trajectories(self, params: dict = None):
        """Appends the current state to the trajectory history."""
        # --- Logic from SPG.update_trajectories ---
        change = jnp.any(jnp.sign(self.omega_cur) != jnp.sign(self.omega_past)).astype(float)
        self.omega_past = self.omega_cur
        params_input = jnp.concatenate([
            jnp.atleast_1d(self.omega_cur),
            jnp.atleast_1d(self.mu_cur),
            jnp.atleast_1d(change)
        ])

        # --- Logic from BaseModel & PolarGenerator (Corrected) ---
        self.t_traj = jnp.append(self.t_traj, self.t_cur)
        self.x_traj = jnp.append(self.x_traj, jnp.expand_dims(self.x_cur, axis=0), axis=0)
        self.q_traj = jnp.append(self.q_traj, jnp.expand_dims(self.q_cur, axis=0), axis=0)
        self.p_traj = jnp.append(self.p_traj, jnp.expand_dims(self.p_cur, axis=0), axis=0)
        self.E_traj = jnp.append(self.E_traj, self.E_cur)
        self.params_traj = jnp.append(self.params_traj, jnp.expand_dims(params_input, axis=0), axis=0)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        """Returns the current joint state [q, dq] from the polar state."""
        return self.polar_to_joints()

