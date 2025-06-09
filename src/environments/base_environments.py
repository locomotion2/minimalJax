from src.CONSTANTS import *

from src.controllers.controllers import PID_pos_vel_tracking_modeled
from src.models.CPG import SPG
from src.models.base_models import DummyOutput

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class BaseEnvironment(ABC):
    def __init__(self, params: dict = None):
        self.delta_t_system = params.get('delta_t_system', MIN_TIMESTEP)
        self.delta_t_learning = params.get('delta_t_learning', MIN_TIMESTEP)
        self.t_final = params.get('t_final', FINAL_TIME)
        self.t_elapsed = 0
        self.solve = params.get('solve', False)
        self.mode = params.get('mode', 'equilibrium')
        self.state_size = params.get('state_size', 4)
        self.num_dof = params.get('num_dof', 2)
        self.step_done = False

        # Build components
        self.model = None
        self.controller = None
        self.generator = None

        # Define parameters
        model_params = {'delta_t': self.delta_t_system,
                        'state_size': self.state_size,
                        'num_dof': self.num_dof,
                        'k_f': 0.0
                        }
        controller_params = {'delta_t': self.delta_t_system,
                             'gains_eigen': [0.15, 0.0, 0.15],
                             'gains_outer': [0.15, 0.0, 0.15],
                             'mode': 'maximal',
                             'num_dof': self.num_dof
                             }
        generator_params = {'delta_t': self.delta_t_learning,
                            'state_size': self.num_dof,
                            'num_dof': self.num_dof
                            }
        self.config = {'model_params': model_params,
                       'controller_params': controller_params,
                       'generator_params': generator_params
                       }

    @abstractmethod
    def step(self, params: dict = None):
        raise NotImplementedError

    def get_cartesian_state_model(self):
        return self.model.get_cartesian_state()

    def get_joint_state_model(self):
        return self.model.get_joint_state()

    def get_joint_state_generator(self):
        [q, dq] = self.generator.get_joint_state()
        return q, dq

    def get_cartesian_state_generator(self):
        [q, _] = self.generator.get_joint_state()
        p = self.model.forward_kins({'joints': q})
        return p

    def get_params_generator(self):
        return self.generator.get_params()

    def get_energies(self):
        return self.model.get_energies()

    def is_done(self):
        done = self.t_elapsed + self.delta_t_learning / 100 >= self.t_final  # TODO: This is a hack to get it to work
        return done

    def is_time_energy_step(self):
        step_bool = self.t_elapsed >= self.t_final / 2 and not self.step_done
        if step_bool:
            self.step_done = True
        return step_bool

    def restart(self, params: dict = None):
        p_0 = self.model.restart({'mode': self.mode, 'E_d': params.get('E_d', 0)})
        self.controller.restart()
        self.generator.restart({'x_0': p_0})
        self.t_elapsed = 0
        self.step_done = False

    def animate(self, step: int = 0, lines: [plt.Line2D] = None):
        # Update positions in cartesian coords
        p_traj_model = self.model.get_cartesian_traj()
        px_model = p_traj_model[step, :, 0] if self.num_dof > 1 else np.asarray([p_traj_model[step, 0]])
        py_model = p_traj_model[step, :, 1] if self.num_dof > 1 else np.asarray([p_traj_model[step, 1]])
        px_model = np.concatenate(([0], px_model), axis=0)
        py_model = np.concatenate(([0], py_model), axis=0)
        lines[0].set_xdata(px_model)
        lines[0].set_ydata(py_model)

        q_traj_CPG = self.generator.get_joint_traj()
        q_traj_CPG = q_traj_CPG[:, 0] if self.num_dof == 1 else q_traj_CPG
        p_CPG = self.model.forward_kins({'joints': q_traj_CPG[step]})
        px_CPG, py_CPG = p_CPG
        lines[1].set_xdata(np.atleast_1d(px_CPG))
        lines[1].set_ydata(np.atleast_1d(py_CPG))

        # Update positions in joint coords
        q_traj_model_rad = np.rad2deg(self.model.get_joint_traj())
        if self.num_dof > 1:
            qx_model, qy_model = q_traj_model_rad[step]
        else:
            qx_model = q_traj_model_rad[step]
            qy_model = 0
        lines[2].set_xdata(np.atleast_1d(qx_model))
        lines[2].set_ydata(np.atleast_1d(qy_model))

        q_traj_CPG = np.rad2deg(q_traj_CPG)
        if self.num_dof > 1:
            qx_CPG, qy_CPG = q_traj_CPG[step]
        else:
            qx_CPG = q_traj_CPG[step]
            qy_CPG = 0
        lines[3].set_xdata(np.atleast_1d(qx_CPG))
        lines[3].set_ydata(np.atleast_1d(qy_CPG))

        # Update param position
        param_traj = self.generator.get_parametric_traj()
        param_x_traj = param_traj[:, 0]
        param_y_traj = np.asarray([0] * np.size(param_traj[:, 0]))
        if self.num_dof != 1:
            param_y_traj = param_traj[:, -2] ** 2
        param_x = param_x_traj[step]
        param_y = param_y_traj[step]
        lines[4].set_xdata(np.atleast_1d(param_x))
        lines[4].set_ydata(np.atleast_1d(param_y))

    def prepare_plot(self):  # TODO: Implement easy closing, transfer methods to underlying classes
        try:
            # Time dataframe
            t_traj = self.model.get_temporal_traj()
            time_data = pd.DataFrame({'time': t_traj})

            # Config. pos against time
            q_traj_model = self.model.get_joint_traj() * 180 / np.pi
            q_d_traj = self.controller.get_desired_traj() * 180 / np.pi
            system_data = pd.DataFrame({})
            for i in range(self.num_dof):
                des_name = f'des_traj_{i}'
                cur_name = f'cur_traj_{i}'
                des_temp_data = pd.DataFrame({des_name: q_d_traj[:, i]})
                cur_temp_data = pd.DataFrame({cur_name: q_traj_model[:, i]})
                system_data = pd.concat([system_data, des_temp_data, cur_temp_data], axis=1)
            system_data = pd.concat([time_data, system_data], axis=1)

            # Pendulum simulation and CPG path in cartesian coords
            p_traj_model = self.model.get_cartesian_traj()
            if self.num_dof == 1:
                px_model = np.asarray([p_traj_model[0, 0]])
                py_model = np.asarray([p_traj_model[0, 1]])
            else:
                px_model = p_traj_model[0, :, 0]
                py_model = p_traj_model[0, :, 1]

            px_model = np.concatenate(([0], px_model), axis=0)
            py_model = np.concatenate(([0], py_model), axis=0)

            q_traj_CPG = self.generator.get_joint_traj()
            if self.num_dof == 1:
                q_traj_CPG = q_traj_CPG[:, 0]
            p_CPG = self.model.forward_kins({'joints': q_traj_CPG[0]})
            [px_CPG, py_CPG] = p_CPG
            x_traj_CPG = np.asarray([self.model.forward_kins({'joints': q}).tolist() for q in q_traj_CPG])
            px_traj = x_traj_CPG[:, 0]
            py_traj = x_traj_CPG[:, 1]

            sim_model_data = pd.DataFrame({'x_model': px_model, 'y_model': py_model})
            sim_CPG_data = pd.DataFrame({'x_CPG': px_CPG, 'y_CPG': py_CPG}, index=[0])
            sim_CPG_traj_data = pd.DataFrame({'x_traj_CPG': px_traj, 'y_traj_CPG': py_traj})

            # Pendulum simulation and CPG path in joint coords
            q_traj_model = self.model.get_joint_traj() * 180 / np.pi
            qx_model = q_traj_model[0, 0]
            qx_model_traj = q_traj_model[:, 0]
            if self.num_dof > 1:
                qy_model = q_traj_model[0, 1]
                qy_model_traj = q_traj_model[:, 1]
            else:
                qy_model = 0
                qy_model_traj = np.asarray([0] * np.size(qx_model_traj))

            if self.num_dof > 1:
                [qx_CPG, qy_CPG] = q_traj_CPG[0] * 180 / np.pi
                qx_CPG_traj = q_traj_CPG[:, 0] * 180 / np.pi
                qy_CPG_traj = q_traj_CPG[:, 1] * 180 / np.pi
            else:
                qx_CPG = q_traj_CPG[0] * 180 / np.pi
                qy_CPG = 0
                qx_CPG_traj = q_traj_CPG * 180 / np.pi
                qy_CPG_traj = np.asarray([0] * np.size(q_traj_CPG))

            sim_model_data_joints = pd.DataFrame({'x_model': qx_model, 'y_model': qy_model}, index=[0])
            sim_model_traj_data_joints = pd.DataFrame({'x_traj_model': qx_model_traj, 'y_traj_model': qy_model_traj})
            sim_CPG_data_joints = pd.DataFrame({'x_CPG': qx_CPG, 'y_CPG': qy_CPG}, index=[0])
            sim_CPG_traj_data_joints = pd.DataFrame({'x_traj_CPG': qx_CPG_traj, 'y_traj_CPG': qy_CPG_traj})

            # RL-params
            param_traj = self.generator.get_parametric_traj()
            param_x_traj = param_traj[:, 0]
            param_y_traj = np.asarray([0] * np.size(param_traj[:, 0]))
            if self.num_dof != 1:
                param_y_traj = param_traj[:, -2] ** 2
            param_gen_traj = param_traj[:, -1]
            param_x = param_x_traj[0]
            param_y = param_y_traj[0]
            rl_param_data = pd.DataFrame({'omega': param_x, 'mu': param_y}, index=[0])
            rl_traj_data = pd.DataFrame({'omega_traj': param_x_traj, 'mu_traj': param_y_traj, 'gen': param_gen_traj})

            # PID controller
            tau_traj = self.controller.get_force_traj().flatten()
            e_traj = self.controller.get_error_traj()
            e_P = e_traj[:, 0].flatten()
            e_I = e_traj[:, 1].flatten()
            e_D = e_traj[:, 2].flatten()
            controller_data = pd.DataFrame({'time': t_traj, 'torque': tau_traj, 'e_P': e_P, 'e_I': e_I, 'e_D': e_D})

            # Energies Plot
            E_traj = self.model.get_energy_traj()
            energy_data = pd.DataFrame({'time': t_traj, 'energy': E_traj})

            return [system_data, [sim_model_data, sim_CPG_data, sim_CPG_traj_data],
                    [sim_model_data_joints, sim_model_traj_data_joints, sim_CPG_data_joints, sim_CPG_traj_data_joints],
                    [rl_param_data, rl_traj_data], controller_data, energy_data]

        except KeyboardInterrupt:
            plt.close(fig=plt.figure('System'))
            plt.close(fig=plt.figure('Reward'))
            raise KeyboardInterrupt


class CPGEnv(BaseEnvironment):
    def __init__(self, params: dict = None):
        super().__init__(params=params)

        # Build components
        self.controller = PID_pos_vel_tracking_modeled(params=self.config.get('controller_params'))
        self.generator = SPG(params=self.config.get('generator_params'))

    def step(self, params: dict = None):
        action = params.get('action')

        # Get RL params
        if self.num_dof != 1:
            omega = np.asarray(action[0:self.num_dof - 1])
            mu = np.asarray(action[self.num_dof - 1:self.num_dof])
        else:
            omega = np.asarray([action[0]])
            mu = np.asarray([])

        if not self.solve:
            # Generate next point in path
            generator_input = {'omega': omega, 'mu': mu}
            self.generator.step(generator_input)
            self.generator.update_trajectories(generator_input)
            [q_d, dq_d] = self.generator.get_joint_state()
        else:
            # Obtain solution from model to compare results
            [q_d, dq_d] = self.model.solve(self.t_elapsed)

        # New step for model
        params['E'] = np.asarray([sum(self.model.get_energies())]).flatten()
        self.controller.set_target(q_d, dq_d, params=params)
        self.model.step({'controller': self.controller.input, 't_final': self.t_elapsed + self.delta_t_learning})

        # Save latest trajectory for plotting
        self.model.update_trajectories()
        self.controller.update_trajectories(q_d)  # Todo: delete the tracking of the desired position in the cont
        self.t_elapsed += self.delta_t_learning


class DirectEnv(BaseEnvironment):
    def __init__(self, params: dict = None):
        super().__init__(params=params)

        # Build components
        self.controller = PID_pos_vel_tracking_modeled(params=self.config.get('controller_params'))
        self.generator = DummyOutput(params=self.config.get('generator_params'))

    def step(self, params: dict = None):
        action = params.get('action')

        # Get desired joint state
        q_d_com = np.asarray(action[0:self.num_dof])
        dq_d_com = np.asarray(action[self.num_dof:2 * self.num_dof])

        # Run through DummyGenerator
        generator_input = {'q_d': q_d_com, 'dq_d': dq_d_com}
        self.generator.step(generator_input)
        self.generator.update_trajectories(generator_input)
        [q_d, dq_d] = self.generator.get_joint_state()

        # New step for model
        params['E'] = np.asarray([sum(self.model.get_energies())]).flatten()
        self.controller.set_target(q_d, dq_d, params=params)
        self.model.step(
            {'controller': self.controller.input, 't_final': self.t_elapsed + self.delta_t_learning})

        # Save latest trajectory for plotting
        self.model.update_trajectories()
        self.controller.update_trajectories(q_d)
        self.t_elapsed += self.delta_t_learning
