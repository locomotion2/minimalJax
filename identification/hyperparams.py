import numpy as np

k = 1
m = 4.0

# snake parameters
snake_system_settings = {
    # "calib_coeffs": [1, 0, 0.1795465350151062, -0.06468642503023148],
    "calib_coeffs": [0.7252218723297119, 6.979795932769775,
                     0.19171033799648285, -0.05231717973947525],
    "system": "snake",  # current system name being identified
    "num_dof": 4,  # number of DOFs of system
    "starting_point": [0, 0, 0, 0, 0, 0, 0, 0],  # staring point in trajectory
    # generation
    "time_step": 0.01,  # time_step for data generation (dpendulum)
}

dpend_system_settings = {
    # "calib_coeffs": [1, 0, 0.1795465350151062, -0.06468642503023148],
    "calib_coeffs": [0.7252218723297119, 6.979795932769775,
                     0.19171033799648285, -0.05231717973947525],
    "system": "dpendulum",  # current system name being identified
    "num_dof": 2,  # number of DOFs of system
    "starting_point": [0, 0, 0, 0],  # staring point in trajectory generation
    "time_step": 0.01,  # time_step for data generation (dpendulum)
}

training_settings = {
    "bootstrapping": False,  # learn a reduced model from the larger one
    "seed": 50,  # random number generator's seed
    "batch_size": int(128 * 8 * k),  # size of each minibatch
    "num_minibatches": int(8 * 16 * m / k),  # number of mini batches per epoch
    "num_batches": 1,  # how many times to repeat the epoch (unused)
    "num_epochs": 1000000,  # number of epochs
    "test_every": 10,  # after how many times to print the test data
    "loss_weights": [1, 100, 10],  # weights for the loss function (forw, inv, energies)
    "loss_weights_model": [1, 0, 0],  # weights for the loss function (forw, inv,
    # energies)
    "loss_weights_red": [0, 0, 0, 10000000],  # weights for the loss
    # function (forw, inv, energies, boot)
    "stage": 1,  # minus power 10 to reduce the learning rate
    "lr_start": 1e-4 * np.sqrt(k),  # starting learning rate
    "lr_end": 1e-4 * np.sqrt(k),  # ending learning rate
    "weight_decay": 1e-5,  # weight for weight decay regularization
    "es_gain": 1000,  # gain for early stopping
}

model_settings = {
    "goal": "energy",  # define the goal model being used
    "friction": False,  # use friction network or not
    "friction_model": False,  # use friction in the model or not
    "model_pot": False,  # use model potential function
    "model_pot_model": False,  # use the model of potential function
    "buffer_length": 20,  # training buffer length
    "buffer_length_max": 20,  # maximal buffer length in database
    "base_dir": "std_models",
    "ckpt_dir": "current",  # directory of the current weights
    "ckpt_dir_model": "model",  # directory of the current weights
    "ckpt_dir_red": "reduced",  # directory of the current weights
    "h_dim": 64 * 4,  # size of the hidden layer
    "h_dim_model": 64 * 10,
}

data_settings = {
    "data_partition": [0.1 * m, 0.05, 0.05],  # train, val, test data partition
    "database_name": "/home/gonz_jm/Documents/thesis_workspace/databases"
                     "/database_250pts_20buff_command_standard",  # location of
    # the training database
    "database_name_test": "/home/gonz_jm/Documents/thesis_workspace/databases"
                          "/database_points_20buff_command_standard",
    "table_name": "data_scrambled",  # name of the table in the database
    "data_dir": "tmp/data",  # dpend database locations (will be changed to have
    # unified dir)
    "data_source": "database",  # whether to load date from a database or generate it
    "eff_datasampling": 1,  # number of batches to load in memory at the time (unused)
    "num_sections": 100,
    "num_generators": 100
}

settings = {
    "system_settings": snake_system_settings,
    "training_settings": training_settings,
    "model_settings": model_settings,
    "data_settings": data_settings,
    "simulate": False,  # simulate in the test_script
    "reload": True,  # whether to reload params from file or keep train new NN
    "save": True,  # save currently trained model or not
}
