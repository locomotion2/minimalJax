# EigenHunt

EigenHunt is a framework for discovery and control for eigenmodes and similar 
equipotential oscillations in dynamical systems in a model-free fashion as much as 
possible.

## Structure of the project
Inside `src` are all the relevant files for discovering eigenmodes given a model These are broken into `controllers` 
which has the controllers connecting the generated trajectories and the simulated systems. `environments` are 
wrappers that contain the utilities to simulate, track and plot the results. `learning` has the additions to rlzoo3 
to get everything running and `models` has the mathematical definitions of dynamic systems being used.

## Installation
```
pip install rl_zoo3
pip install sbx-rl
```

## Eigenmode discovery

How to train the model:
```
python train.py --algo tqc --env TestEnvironment-v1 -P --conf-file hyperparams/tqc.yml --env-kwargs render:False mode:"'speed'" --eval-freq -1 -params train_freq:10 gradient_steps:10 --log-interval 100 -n 200000
```
How to run a trained model:
```
python enjoy.py --algo tqc --env TestEnvironment-v1  --env-kwargs render:True mode:"'speed'" solve:False energy_command:0.3 -f logs/ -n 100
```
Some useful flags (Defaults):
```
solve:False  # Plays the solution of the system instead of the CPG
mode:'equilibrium'  # {'speed', 'position'} Defines the starting conditions
render:False  # Allows plotting of the system
energy_command:None  # Rational value, between 0 and 1 J
folder:saved_models # Where to get the model from, the correct model is chosen automatically
```
How to plot the learning curve:
```
python plot_train.py --algo tqc --env TestEnvironment-v1 -f logs
```