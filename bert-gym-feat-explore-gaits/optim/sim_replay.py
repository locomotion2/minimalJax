import argparse

import pybullet
import zmq
from bert_utils import RateLimiter, RobotSim
from cpg_gamepad import COUPLING_MATRICES, GAITS, CPGController

from custom_envs.constants import RESET_MOTOR_PARAM
from optim.study_utils import PLOT_PUBLISHER_PORT, gait_params_pattern, get_coupling_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--listen", help="Listen to publisher", action="store_true", default=False)
    parser.add_argument("-f", "--follow", help="Follow robot with camera", action="store_true", default=False)
    args = parser.parse_args()

    if args.listen:
        context = zmq.Context()
        subscriber = context.socket(zmq.PULL)
        subscriber.bind(f"tcp://*:{PLOT_PUBLISHER_PORT}")

    robot = RobotSim(
        fps=500,
        sim_dt=1 / 1000,
        reset_motor_params=RESET_MOTOR_PARAM,
        max_vel=4,
        max_force=3,
        fixed_base=False,
        kp=1000,
        kd=2,
        max_torque=10,
        f_cutoff=100,
        # f_cutoff=-1,
        # additional_leg_friction=1.5,
        with_spring=True,
        spring_stiffness=3,
        # motor_inertia=0.000005,
    )

    cpg_controller = CPGController(GAITS.PRONK, backward=False, dt=1 / robot.fps)
    # Change gait
    # cpg_controller.switch_gait(GAITS.BOUND, coupling_matrix=COUPLING_MATRICES[GAITS.BOUND], backward=True)

    # Print every 2 seconds
    rate_limiter = RateLimiter(wanted_dt=1 / robot.fps, print_freq=2 * robot.fps, verbose=1)

    print("Visualization ready")
    try:
        robot.move_robot(RESET_MOTOR_PARAM)
        while True:

            desired_motor_pos = cpg_controller.update()
            # robot.move_robot(cpg_controller.direction * desired_motor_pos)

            for _ in range(robot.action_repeat):
                # robot.move_robot(cpg_controller.direction * desired_motor_pos)

                # robot.move_robot(RESET_MOTOR_PARAM * 1.2)
                robot.move_robot(RESET_MOTOR_PARAM + 0.5)
                robot.pybullet_client.stepSimulation()

            rate_limiter.update_control_frequency(init=False)

            events = robot.pybullet_client.getKeyboardEvents()
            # r is pressed: reset
            if 114 in events:
                robot.reset_robot_pos()

            # Follow robot
            if args.follow:
                robot_position = robot.pybullet_client.getBasePositionAndOrientation(robot.robot_id)[0]
                cam_info = robot.pybullet_client.getDebugVisualizerCamera()
                distance = cam_info[10]
                pitch = cam_info[9]
                yaw = cam_info[8]
                robot.pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, robot_position)

            if args.listen:
                try:
                    trial = subscriber.recv_pyobj(flags=zmq.NOBLOCK)
                    parameters = {key: trial.params[key] for key in trial.params if gait_params_pattern.match(key)}
                    cpg_controller.switch_gait(
                        GAITS.TROT,
                        coupling_matrix=get_coupling_matrix(trial),
                        backward=False,
                        parameters=parameters,
                    )
                except zmq.Again:
                    pass

    except KeyboardInterrupt:
        robot.pybullet_client.disconnect()
    except pybullet.error:
        pass
