from abc import ABC
import numpy as np  # Keep for solve_ivp which may be used in base_models
import jax
import jax.numpy as jnp

# Assuming your base models are structured like this
from src.models.base_models import BaseModel, JointsGenerator, PolarGenerator

class CPG(BaseModel, ABC):
    def __init__(self, params: dict = None):
        # The __init__ method is not JIT-compiled, so regular Python is fine.
        # Assuming num_dof is set in the super().__init__(params) call
        super().__init__(params)
        self.omega_cur = jnp.asarray([0] * self.num_dof)
        self.mu_cur = 0
        self.coils = 0

        params_cpg = jnp.asarray([0, 1])
        self.params_traj = jnp.asarray([params_cpg])
        # It's good practice to create the jitted function once
        self.eqs_motion = self.make_eqs_motion()


    def make_eqs_motion(self, params: dict = None):
        @jax.jit
        def eqs_motion(t, x, params):
            mu = params['mu']
            omega = params['omega']

            rho = x[0] ** 2 + x[1] ** 2
            circleDist = mu ** 2 - rho

            dx1 = -x[1] * omega + x[0] * circleDist
            dx2 = x[0] * omega + x[1] * circleDist

            return jnp.asarray([dx1, dx2])

        return eqs_motion

    def restart(self, params: dict = None):
        # Re-implement BaseModel.restart to fix E_cur initialization
        self.t_0 = params.get('t_0', self.t_0)
        p_0 = self.select_initial(params)
        self.t_cur = self.t_0
        # Corrected E_cur/E_0 initialization to produce a scalar
        self.E_0 = sum(self.get_energies())
        self.E_cur = self.E_0
        # Reset tracking vars from BaseModel.restart
        self.t_traj = self.t_cur
        self.x_traj = jnp.asarray([self.x_cur])
        self.q_traj = jnp.asarray([self.q_cur])
        self.p_traj = jnp.asarray([self.p_cur])
        self.E_traj = self.E_cur

        # Logic from CPG.restart itself
        params_cpg = jnp.asarray([0, 1])
        self.params_traj = jnp.asarray([params_cpg])
        self.coils = 0
        return p_0

    def step(self, params: dict = None):
        # Re-implement BaseModel.step to fix E_cur calculation, avoiding the call to super().step().
        t_final = jnp.asarray([params.get('t_final', self.t_cur + self.delta_t)]).flatten()
        ts = jnp.asarray([self.t_cur, t_final]).flatten()

        # Simulate the system until t_final
        self.x_cur = self.simulate({'eqs': self.eqs_motion,
                                    'eqs_params': params,
                                    'ts': ts,
                                    'x_0': self.x_cur})

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        # Corrected E_cur calculation to produce a scalar
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final

    def update_trajectories(self, params: dict = None):
        # Re-implement the BaseModel.update_trajectories logic correctly here to avoid the error in the superclass.
        self.t_traj = jnp.append(self.t_traj, self.t_cur)
        # Correctly append arrays without creating lists
        self.x_traj = jnp.append(self.x_traj, jnp.expand_dims(self.x_cur, axis=0), axis=0)
        self.q_traj = jnp.append(self.q_traj, jnp.expand_dims(self.q_cur, axis=0), axis=0)
        self.p_traj = jnp.append(self.p_traj, jnp.expand_dims(self.p_cur, axis=0), axis=0)
        self.E_traj = jnp.append(self.E_traj, self.E_cur)

        # Now do the CPG-specific updates that were originally after the super() call.
        self.mu_cur = params['mu']
        self.omega_cur = params['omega']
        params_val = jnp.asarray([self.omega_cur, self.mu_cur])
        self.params_traj = jnp.append(self.params_traj, jnp.expand_dims(params_val, axis=0), axis=0)

    def detect_coiling(self):
        # This method is not JIT-compiled, so Python conditionals are fine.
        if len(self.x_traj) < 2:
            return self.coils
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
        self.q_0 = jnp.asarray([0] * self.num_dof)
        self.q_cur = jnp.asarray(self.q_0)
        # Assuming get_link_cartesian_positions returns a JAX array
        # The .tolist() call from the original might cause issues if this method is ever JITted.
        # Sticking to JAX arrays is safer.
        self.p_0 = self.get_link_cartesian_positions()
        self.p_cur = jnp.asarray(self.p_0)

    def get_cartesian_state(self):
        params = {'mu': self.mu_cur, 'omega': self.omega_cur}
        # Use the pre-compiled function
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
        # This conditional logic is fine in __init__
        if self.num_dof != 1:
            self.omega_cur = jnp.asarray([0] * (self.num_dof - 1))
            self.omega_past = jnp.asarray([0] * (self.num_dof - 1))
            self.mu_cur = jnp.asarray([0])
        else:
            self.omega_cur = jnp.asarray([0])
            self.omega_past = jnp.asarray([0])
            self.mu_cur = jnp.asarray([0]) # Ensure mu_cur is always an array for consistency
        self.eqs_motion = self.make_eqs_motion()

    def make_eqs_motion(self, params: dict = None):
        # Use a static Python if to create the correct JIT'd function
        if self.num_dof != 1:
            num_dof = self.num_dof # Capture as static value
            @jax.jit
            def eqs_motion(t, x, params):
                mu = params['mu']
                omega = params['omega']
                q = jnp.asarray(x, dtype=float).flatten()
                psi = mu ** 2 - jnp.linalg.norm(q) ** 2
                root = jnp.ones((num_dof, num_dof))
                upper = jnp.multiply(jnp.triu(root, 1), omega)
                lower = jnp.multiply(jnp.tril(root, -1), -omega)
                diag_values = jnp.full((num_dof,), psi.squeeze())
                diag = jnp.diag(diag_values)
                A = upper + diag + lower
                return A @ q
        else:
            @jax.jit
            def eqs_motion(t, x, params):
                return params['omega']
        return eqs_motion

    def restart(self, params: dict = None):
        # Re-implement BaseModel.restart and JointsGenerator.restart
        # 1. Logic from BaseModel.restart
        self.t_0 = params.get('t_0', self.t_0)
        p_0 = self.select_initial(params)
        self.t_cur = self.t_0
        # CORRECTED LOGIC:
        self.E_0 = sum(self.get_energies())
        self.E_cur = self.E_0
        # Reset tracking vars from BaseModel.restart
        self.t_traj = self.t_cur
        self.x_traj = jnp.asarray([self.x_cur])
        self.q_traj = jnp.asarray([self.q_cur])
        self.p_traj = jnp.asarray([self.p_cur])
        self.E_traj = self.E_cur

        # 2. Logic from JointsGenerator.restart
        self.params_traj = jnp.asarray([jnp.asarray([0] * (self.num_dof + 1))])
        
        return p_0

    def step(self, params: dict = None):
        params['num_dof'] = self.num_dof
        
        # Re-implement BaseModel.step to fix E_cur calculation
        t_final = jnp.asarray([params.get('t_final', self.t_cur + self.delta_t)]).flatten()
        ts = jnp.asarray([self.t_cur, t_final]).flatten()

        # Simulate the system until t_final
        self.x_cur = self.simulate({'eqs': self.eqs_motion,
                                    'eqs_params': params,
                                    'ts': ts,
                                    'x_0': self.x_cur})

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        # Corrected E_cur calculation to produce a scalar
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final

        # GPG-specific updates
        self.omega_cur = jnp.asarray(params['omega'])
        self.mu_cur = jnp.asarray(params['mu'])

    def update_trajectories(self, params: dict = None):
        # Re-implement the full inheritance chain of update_trajectories to avoid the error in BaseModel.
        # 1. Logic from GPG itself
        change = 0
        if jnp.any(jnp.sign(self.omega_cur) != jnp.sign(self.omega_past)):
            change = 1
        self.omega_past = self.omega_cur
        params_input = jnp.concatenate([
            jnp.atleast_1d(self.omega_cur),
            jnp.atleast_1d(self.mu_cur),
            jnp.asarray([change])
        ], axis=0)
        
        # 2. Logic from BaseModel (Corrected)
        self.t_traj = jnp.append(self.t_traj, self.t_cur)
        self.x_traj = jnp.append(self.x_traj, jnp.expand_dims(self.x_cur, axis=0), axis=0)
        self.q_traj = jnp.append(self.q_traj, jnp.expand_dims(self.q_cur, axis=0), axis=0)
        self.p_traj = jnp.append(self.p_traj, jnp.expand_dims(self.p_cur, axis=0), axis=0)
        self.E_traj = jnp.append(self.E_traj, self.E_cur)

        # 3. Logic from JointsGenerator (Corrected)
        self.params_traj = jnp.append(self.params_traj, jnp.expand_dims(params_input, axis=0), axis=0)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        q = jnp.asarray(self.x_cur).flatten()
        # Pass num_dof in params dict for the JITted function
        params = {'mu': self.mu_cur, 'omega': self.omega_cur, 'num_dof': self.num_dof}
        dq = jnp.asarray(self.eqs_motion(0, q, params)).flatten()
        return [q, dq]

class SPG(PolarGenerator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        if self.num_dof != 1:
            self.omega_cur = jnp.asarray([0] * (self.num_dof - 1))
            self.omega_past = jnp.asarray([0] * (self.num_dof - 1))
            self.mu_cur = jnp.asarray([0])
        else:
            self.omega_cur = jnp.asarray([0])
            self.omega_past = jnp.asarray([0])
            self.mu_cur = jnp.asarray([0]) # Ensure mu_cur is always an array
        self.eqs_motion = self.make_eqs_motion()

    def make_eqs_motion(self, params: dict = None):
        if self.num_dof != 1:
            @jax.jit
            def eqs_motion(t, x, params):
                mu = params['mu']
                omega = params['omega']
                r = x[-1]
                dr = (mu ** 2 - r ** 2) * r
                dphi = omega
                return jnp.asarray([dphi, dr], dtype=float).flatten()
        else:
            @jax.jit
            def eqs_motion(t, x, params):
                return params['omega']
        return eqs_motion

    def restart(self, params: dict = None):
        # Re-implement BaseModel.restart and PolarGenerator.restart
        # 1. Logic from BaseModel.restart
        self.t_0 = params.get('t_0', self.t_0)
        p_0 = self.select_initial(params)
        self.t_cur = self.t_0
        # CORRECTED LOGIC:
        self.E_0 = sum(self.get_energies())
        self.E_cur = self.E_0
        # Reset tracking vars from BaseModel.restart
        self.t_traj = self.t_cur
        self.x_traj = jnp.asarray([self.x_cur])
        self.q_traj = jnp.asarray([self.q_cur])
        self.p_traj = jnp.asarray([self.p_cur])
        self.E_traj = self.E_cur

        # 2. Logic from PolarGenerator.restart
        self.params_traj = jnp.asarray([jnp.asarray([0] * (self.num_dof + 1))])
        
        return p_0

    def step(self, params: dict = None):
        params['num_dof'] = self.num_dof
        
        # Re-implement BaseModel.step to fix E_cur calculation
        t_final = jnp.asarray([params.get('t_final', self.t_cur + self.delta_t)]).flatten()
        ts = jnp.asarray([self.t_cur, t_final]).flatten()

        # Simulate the system until t_final
        self.x_cur = self.simulate({'eqs': self.eqs_motion,
                                    'eqs_params': params,
                                    'ts': ts,
                                    'x_0': self.x_cur})

        # Update the current variables
        self.p_cur = self.get_link_cartesian_positions()
        self.q_cur = self.state_to_joints()
        # Corrected E_cur calculation to produce a scalar
        self.E_cur = sum(self.get_energies())
        self.t_cur = t_final

        # SPG-specific updates
        self.omega_cur = jnp.asarray(params['omega'])
        self.mu_cur = jnp.asarray(params['mu'])

    def polar_to_joints(self, state: jnp.ndarray = None):
        # Since num_dof is static for the instance, a regular Python if is appropriate
        # and avoids the strict shape-matching rules of jax.lax.cond.
        if self.num_dof != 1:
            x_cur, mu_cur, omega_cur = self.x_cur, self.mu_cur, self.omega_cur
            num_dof = self.num_dof
            
            phi = x_cur[0:num_dof - 1]
            r = x_cur[-1]
            q = jnp.asarray([r * jnp.cos(phi), r * jnp.sin(phi)])
            psi = mu_cur ** 2 - r ** 2
            root = jnp.ones((num_dof, num_dof))
            upper = jnp.multiply(jnp.triu(root, 1), -omega_cur)
            lower = jnp.multiply(jnp.tril(root, -1), omega_cur)
            diag = jnp.diag(jnp.full((num_dof, num_dof), psi.squeeze()))
            A = upper + diag + lower
            dq = A @ q.flatten()
            q_out, dq_out = q.flatten(), dq
        else:
            x_cur, mu_cur, omega_cur = self.x_cur, self.mu_cur, self.omega_cur
            q_out = x_cur[0]
            dq_out = omega_cur

        return [jnp.asarray(q_out, dtype=float).flatten(), jnp.asarray(dq_out, dtype=float).flatten()]

    def joints_to_polar(self, joints: jnp.ndarray = None):
        # Using a static Python `if` is correct here as well.
        if self.num_dof != 1:
            r = jnp.linalg.norm(joints)
            phi = jnp.arctan2(joints[1], joints[0])
            return jnp.asarray([phi, r])
        else:
            return joints

    def update_trajectories(self, params: dict = None):
        # Re-implement the full inheritance chain of update_trajectories to avoid the error in BaseModel.
        # 1. Logic from SPG itself
        change = 0
        if jnp.any(jnp.sign(self.omega_cur) != jnp.sign(self.omega_past)):
            change = 1
        self.omega_past = self.omega_cur
        params_input = jnp.concatenate([
            jnp.atleast_1d(self.omega_cur),
            jnp.atleast_1d(self.mu_cur),
            jnp.asarray([change])
        ], axis=0)

        # 2. Logic from BaseModel (Corrected)
        self.t_traj = jnp.append(self.t_traj, self.t_cur)
        self.x_traj = jnp.append(self.x_traj, jnp.expand_dims(self.x_cur, axis=0), axis=0)
        self.q_traj = jnp.append(self.q_traj, jnp.expand_dims(self.q_cur, axis=0), axis=0)
        self.p_traj = jnp.append(self.p_traj, jnp.expand_dims(self.p_cur, axis=0), axis=0)
        self.E_traj = jnp.append(self.E_traj, self.E_cur)
        
        # 3. Logic from PolarGenerator (Corrected)
        self.params_traj = jnp.append(self.params_traj, jnp.expand_dims(params_input, axis=0), axis=0)

    def get_params(self):
        return [self.omega_cur, self.mu_cur]

    def get_joint_state(self):
        out = self.polar_to_joints(self.x_cur)
        return out
