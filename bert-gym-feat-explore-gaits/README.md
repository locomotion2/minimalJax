# Bert Gym Environments

TODO(cpg):
- check mean speed computation/that orientation is recorded


TODO:
- normalize imu data (+ others?)

TODO(walking):
- try to train using polar coordinates directly
- try using symmetric control for trot (diagonal)
- better to have desired walking speed

TODO(modes):
- properly fix sim/real robot mismatch:

real robot:
- [0.5, -0.5] bended forward vs backward in sim
- tau [-0.3, 0.3] vs [-0.3, -0.3] in the sim
- freezes when losing tracking (because topic is blocking, also limit the rate to 60Hz)

### Bert Simulation

Gazebo plugin: https://rmc-github.robotic.dlr.de/quadruped/gazebo-plugins
(`cissy create` and `cissy create conanfile.msgdef.py` to build)

Gazebo models: https://rmc-github.robotic.dlr.de/quadruped/bert_model
(`cissy create` to build)

Gazebo sim: https://rmc-github.robotic.dlr.de/quadruped/gazebo-sim

1. Start LN manager (gazebo sim, `cissy run`) and start the simulation process

2. Set the right `LN_MANAGER` env variable and add ln python binding to current env

3. Train the agent using the zoo (check `bert/util/constants.py` before for setting the right socket server address)

```
python train.py --algo tqc --env BertEnv-v1 --eval-freq -1 --save-replay-buffer --save-freq 10000 --env-kwargs is_real_robot:True
```

1h logs/tqc/TurningBertEnv-v1_11
```
python train.py --algo tqc --env TurningBertEnv-v1 -f logs --save-replay-buffer --eval-freq -1 --save-freq 10000 --env-kwargs is_real_robot:True render:False
```

Fixed turning:
```
python train.py --algo tqc --env FixedTurningBertEnv-v2 -f logs --save-replay-buffer --eval-freq -1 --save-freq 10000 --env-kwargs is_real_robot:True randomize_target:False turning_angle:30 render:False
```

Note: it saves regular checkpoints and the replay buffer at the end of training

# Real bert

controller repo: https://rmc-github.robotic.dlr.de/quadruped/controller_quadruped
and updated instructions: https://rmc-github.robotic.dlr.de/quadruped/controller_bert_simulink/blob/master/README.md

`cissy deploy -fu`

for rebuilding matlab controllers: `make -j all` (be careful with the RAM)

### Bert Modes

```
export LN_MANAGER=localhost:54133
```

NUMBA_DISABLE_JIT=1 for debug

- minimal control frequency -> works with lower control freq (5Hz) but slower and with higher control freq, need to be tuned (thresholds)


with optuna:
```
python optim/optuna_optimize.py --env ModesBertReal-v1 -name greybert-one-param-leg --storage sqlite:///logs/studies.db
```

```
TREADMILL_ENABLED=True python optim/optuna_optimize.py --env CPGBertReal-v1 -name cpg-trot-optimize-energy --storage sqlite:///logs/studies.db --optimize-energy
```

and visualize results with optuna dashboard
```
optuna-dashboard sqlite:///logs/studies.db
```

Print and save best params:
```
python optim/parse_study.py -name greybert-global-tpe --storage sqlite:///logs/studies.db --print-n-best-trials 10 --save-n-best-hyperparameters 10 -f logs/greybert-global-tpe-params/
```

Test a set of params (using a json file)
```
python optim/eval_params.py -i logs/greybert-global-tpe-params/hyperparameters_1.json --n-eval-episodes 5
```

## Central Pattern Generator

```
python custom_envs/cpg/cpg.py -g trot -plot
```

Optimize:
```
TREADMILL_ENABLED=True python optim/optuna_optimize.py --env CPGBertReal-v1 -name cpg-optimize-speed --storage sqlite:///logs/studies.db --optimize-speed
```

Evaluate:
```
TREADMILL_ENABLED=True python optim/eval_params.py --storage sqlite:///logs/studies.db --env CPGBertReal-v1 -name cpg-trot-optimize-energy -trial 3 --n-eval-episodes 3 --max-episode-steps 300
```

Pareto plot (need to check keys first)
```
python optim/plot_pareto.py --storage sqlite:///logs/studies.db -name cpg-trot-optimize-energy
```

Optimize pronk:
```
python optim/optuna_optimize.py --env CPGBertPronk-v1 -name cpg-optimize-pronk --storage sqlite:///logs/studies.db --optimize-reward
```
Evaluate:
```
python optim/eval_params.py --storage sqlite:///logs/studies.db --env CPGBertPronk-v1 -name cpg-optimize-pronk -trial 94 --n-eval-episodes 3 --max-episode-steps 300
```

RL:
```
TREADMILL_ENABLED=True CUDA_VISIBLE_DEVICES= python train.py --algo tqc --env BertTrotSpeed-v1 --env-kwargs action_repeat:2 symmetry:"'trot'" max_offset:0.004 --eval-freq -1 -tb logs/tb/bert-trot --save-freq 10000 --save-replay-buffer

CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertTrotSpeed-v1 -f logs/ --exp-id 0 -n 1500 --env-kwargs action_repeat:1

CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertPronk-v1 -f logs/ --exp-id 0 -n 1500 --env-kwargs action_repeat:1
```

## Explore gaits


Plots

```
python optim/plot_reached_points.py -name cpg-explore-optimize-speed --storage sqlite:///logs/studies.db
```

```
python custom_envs/cpg/plot.py -g trot -shift front_left:0 front_right:0.5 rear_left:0.5 rear_right:0.0
```

## System Overview

### Tracking system

1. start `tracking_system/dtrack_controller` (start the gui for the tracking system) and follow the steps (can close
   the window)
2. start `tracking_system/dtrack_ln_publisher_0`

topic: `tracking.bert` at 60Hz

### Treadmill

- Make sure that the treadmill can safely be operated and that the emergency stop is within reach!
- Enable the treadmill by disengaging the emergency stop.
- Start ln topics

In `treadmill_tracking`:

- `treadmill_provider`
- `treadmill_gui` (optional for manual operation)

### Simulink Model Compilation

1. start matlab
2. open `controller.slx` (will open automatically normally)
3. call `build_modules`

`build_modules.m` does the complete build.

Or:
1. `make -j all` (best on a bulky machine, it will launch MANY instances of Matlab)

### Message definitions / topics:

controller.msr (msr_bus): measures

for poses the 7th value is a singularity flag for the euler angles

- theta: motor pos
- q: joint pos
- tau: torque (from deflection)


- x: (orientation from imu software)
- dx: imu (not filtered yet: rotation velocities)
- ddx: (not filtered yet: acceleration)


- x_tracking: position and orientation from tracking system (euler angles)
- x_tracking_rotmat: (with rotation matrix)


- current_bridge (measured on the servos)
- voltage_bridge (measured on the servos)

- power = current_bridge * voltage_bridge

- temp_bridge (cutoff at 85deg Celsius) measured inside the motor
- temp_motor (measured at the outside)

- tau_z: torque summed up per leg (radial direction)

polar coord of the feet:
alpha r0

controller.ctrl: what is really sent to the robot

## Startup

### Start ln manager:

- source the links_and_nodes environment
- check which robot you want to use:
    - 21: stiff
    - 23: less stiff

- run the startup script with the id of the robot

```
./start.sh 21
```

### Start controller

1. start `applications/controller` in ln_manager

   you can also start the robot kernels first: `servo_rk_{front/hind}`, `imu_rk` and `gamepad_rk`
2. start `applications/controller`
3. start tracking and treadmill controller

### Check motor position limits

manually check if all motors move in the right direction and if the limits are set correctly

-> use ./custom_envs/scripts/test_motors.py to move joints in maximum positions

### Calibrate the leg (optionally)

(set to zeros if there is an offset)

1. Put it on a box
2. deactivate motor power (ln parameters)
3. align the in the leg
4. `utilities/reset_servo_positions`
5. make sure that the reset_position in `bert/util/constants.py` are still valid

## Shutdown

long press the green button on the robot

or

`power_unit/pdu_shutdown`

## Playing around

After successful startup start
- `applications/gamepad_control`
- `utilities/controller_scheme`

to control the robot via gamepad.


## Commanding the robot via Script

change parameter motor_pos_test, position key

- set `controller.app_select = 4`
- set `controller.motor_power = True`
- set `motor_pos_test.select = 1.0`
- set `motor_pos_test.dtheta_d` to desired max speed
- override `motor_pos_test.position` to move robot

take a look at
the [gamepad control](https://rmc-github.robotic.dlr.de/quadruped/controller_quadruped/blob/master/python_control/gamepad_control.py)
and
the [wrapper](https://rmc-github.robotic.dlr.de/quadruped/controller_quadruped/blob/master/python_control/simulink_com.py)

```python
import os

import links_and_nodes as ln
import numpy as np

LN_MANAGER = os.environ.get("LN_MANAGER", "localhost:4444")

ln_client = ln.client("test_client", LN_MANAGER)
parameter = ln_client.get_parameters("motor_pos_test.position")

n_motors = 8
motor_params = np.zeros(n_motors)
parameter.set_override("value", motor_params)
```

# Known Issues

## Robotkernel Fail

Sometimes the robot kernel fail and thus the motors do not respond to command anymore.

In this case follow these steps to avoid a crash of the ln_manager and continue training.

- FIRST! stop (press stop although they are already stopped) the failed kernels (servo_rk_front /hind)
- stop imu_rk and gamepad_rk (should be still running)
- after all dependencies of controller are stopped, stop controller itself
- restart controller, wait for it to be started
- restart gamepad_rk (used for publishing the logitechf710 buttons)
- start gamepad_control and press back button several times (will set correct mode in state_machine)
- wait for several seconds (seems to work more reliable then)
- stop gamepad_control (so it doesn't interfere with our control)
- press A on the gamepad (several times) to continue training
