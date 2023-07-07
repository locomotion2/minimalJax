# EigenHunt

EigenHunt is a framework for discovery and control for eigenmodes and similar 
equipotential oscilations in dinamical systems in a model-free fashion as much as 
possible.

## Structure of the project
in folder `sim` is all the relevant files for discovering eigenmodes given a model. 
And in folder `identification` are all the files relevant for model identification.

## Model identification
*This will be moved to another repository in the future.*

`hyperparams`: Centralized params file, contains the following variables:
```
settings = {'batch_size': 128 * 8 * k,  # size of each minibatch
            'num_minibatches': 5 * int(16 * m / k),  # number of mini batches per epoch
            'num_batches': 1,  # how many times to repeat the epoch (unused)
            'num_epochs': 1000,  # number of epochs
            'test_every': 10,  # after how many times to print the test data
            'loss_weights': [1, 10, 10],  # weights for the loss function (forw, inv,
            # energies)
            'lr_start': 1e-4 * np.sqrt(k),  # starting learning rate
            'lr_end': 1e-4,  # ending learning rate
            'weight_decay': 1e-5,  # weight for weight decay regularization
            'es_gain': 1000,  # gain for early stopping
            'h_dim': 64 * 4,  # size of the hidden layer
            'friction': False,  # use friction network or not
            'model_pot': True,  # use model potential function
            'seed': 50,  # random number generator's seed
            'buffer_length': 20,  # training buffer length
            'buffer_length_max': 20,  # maximal buffer length in database
            'ckpt_dir': 'tmp/current',  # directory of the current weights
            'system': 'snake',  # current system name being identified
            'num_dof': 4,  # number of DOFs of system
            'starting_point': [0, 0, 0, 0],  # staring point in trajectory generation
            'data_dir': 'tmp/data',  # dpend database locations (will be changed to
            # have unified dir)
            'data_source': 'database',  # whether to load date from a database or
            # generate it
            'eff_datasampling': 1,  # number of batches to load in memory at the time
            # (unused)
            'time_step': 0.01,  # time_step for data generation (dpendulum)
            'data_partition': [0.1 * m, 0.05, 0.05],  # train, val, test data partition
            'database_name': '/home/gonz_jm/Documents/thesis_workspace/databases'
                             '/database_150pts_20buff_command',  # location of the
            'table_name': 'data_scrambled',  # name of the table in the database
            'reload': False,  # whether to reload params from file or keep train new NN
            'save': True  # save currently trained model or not
            }
```
Inside `src` are the main functions of the project:
1. `lagranx`: Handles all the dynamics and loss functions (currently replicates (TODO: 
   ref DeLaNN))
2. `trainer`: Implements the training loop usinig google's flax
3. `utils`: Functions used around the project

Inside `systems` are the system dependent functions (data loading, network archs, etc.):
1. `snake_utils`: Snake system dependent functions
2. `dpend_utils`: Double pendulum dependent functions

Inside `scripts` are the ways to interact with the other classes:
1. `train_model`: trains the model based on the params in `hyperparams`
2. `test_snake`: calculates and plots multiple functions of interest for the snake
3. `test_model`: calculates and plots multiple functions of interest for the dpend

Inside `applications` are the learned model used for other purposes:
1. `model_observer`: This script calculates the dynamic terms, decouples them and 
   then sends them as topics
2. `model_predictor`: This script reads in the state of the system and then predicts 
   the current torques and accelerations.

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