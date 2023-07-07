import numpy as np

k = 1
m = 4.0

settings = {'batch_size': int(128 * 8 * k),  # size of each minibatch
            'num_minibatches': int(8 * 16 * m / k),  # number of mini batches per epoch
            'num_batches': 1,  # how many times to repeat the epoch (unused)
            'num_epochs': 1000000,  # number of epochs
            'test_every': 10,  # after how many times to print the test data
            'loss_weights': [1, 10, 10],  # weights for the loss function (forw, inv,
            # energies)
            'loss_weights_model': [1, 0, 0],  # weights for the loss function (forw,
            # inv,
            # energies)
            'loss_weights_red': [0, 0, 0, 10000000],  # weights for the loss
            # function (forw, inv, energies, boot)
            'stage': 1,  # minus power 10 to reduce the learning rate
            'lr_start': 1e-4 * np.sqrt(k),  # starting learning rate
            'lr_end': 1e-4 * np.sqrt(k),  # ending learning rate
            'weight_decay': 1e-5,  # weight for weight decay regularization
            'es_gain': 1000,  # gain for early stopping
            'h_dim': 64 * 4,  # size of the hidden layer
            'h_dim_model': 64 * 10,
            'friction': False,  # use friction network or not
            'friction_model': False,
            'model_pot': False,  # use model potential function
            'model_pot_model': False,
            'bootstrapping': False,  # learn a reduced model from the larger one
            'simulate': False,
            'goal': 'energy',
            'seed': 50,  # random number generator's seed
            'buffer_length': 20,  # training buffer length
            'buffer_length_max': 20,  # maximal buffer length in database
            'ckpt_dir': 'tmp/current',  # directory of the current weights
            'ckpt_dir_model': 'tmp/model',  # directory of the current weights
            'ckpt_dir_red': 'tmp/reduced',  # directory of the current weights
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
                             '/database_250pts_20buff_command_standard',  # location of
            # the training database
            'table_name': 'data_scrambled',  # name of the table in the database
            'reload': False,  # whether to reload params from file or keep train new NN
            'save': True  # save currently trained model or not
            }
