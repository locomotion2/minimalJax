import optax
import stable_baselines3.common.save_util as loader
from aim import Run

from identification.src.learning import trainer
from identification.src.learning import plotting
from identification.src.dynamix import optim as optim

from identification.systems import dpendulum_utils, snake_utils

from identification.hyperparams import settings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_training(settings):
    # Unpack settings
    settings_training = settings['training_settings']
    settings_model = settings['model_settings']
    settings_system = settings['system_settings']

    # Define training parameters
    stage = settings_training["stage"]
    num_iterations = (
            settings_training["num_epochs"] *
            settings_training["num_batches"] *
            settings_training["num_minibatches"]
    )
    learning_rate_fn = optax.linear_schedule(
        init_value=settings_training["lr_start"] * 10 ** (-stage),
        end_value=settings_training["lr_end"] * 10 ** (-stage),
        transition_steps=num_iterations,
    )
    settings['training_settings']["lr_func"] = learning_rate_fn
    params = None
    params_red = None
    params_model = None
    model_dir = f"{settings_model['base_dir']}/{settings_system['snake']}"
    path_energy = f"{model_dir}/{settings_model['ckpt_dir']}"
    path_model = f"{model_dir}/{settings_model['ckpt_dir_model']}"
    path_red = f"{model_dir}/{settings_model['ckpt_dir_red']}"
    if settings["reload"]:
        params = loader.load_from_pkl(path=path_energy, verbose=1)
        params_model = loader.load_from_pkl(path=path_model,
                                            verbose=1)
        params_red = loader.load_from_pkl(path=path_red,
                                          verbose=1)
        print(f"Params loaded from file.")

    # Define sys_utils (functions that depend on the particular system)
    if settings_system["system"] == "snake":
        settings['system_settings']["sys_utils"] = snake_utils
    elif settings_system["system"] == "dpend":
        settings['system_settings']["sys_utils"] = dpendulum_utils

    # Create the training state
    train_state = None
    if settings_model["goal"] == "energy":
        train_state = trainer.create_train_state(
            "energy", settings, learning_rate_fn, params=params
        )
    elif settings_model["goal"] == "model":
        train_state = trainer.create_train_state(
            "model", settings, learning_rate_fn, params=params_model
        )
    train_state_red = trainer.create_train_state_red(
        settings, learning_rate_fn, params=params_red
    )

    # Define data & dataloader TODO: change everything to work with databases
    dataloader = trainer.choose_data_loader(settings)

    # Build the dynamic model and get a loss_func
    loss_func, loss_func_red = optim.build_loss(settings)
    print("Setup completed, training will now begin.")

    return (
        (train_state, train_state_red),
        (loss_func, loss_func_red),
        dataloader,
        settings,
    )


def save_model(settings):
    settings_model = settings['model_settings']
    settings_system = settings['system_settings']

    model_dir = f"{settings_model['base_dir']}/{settings_system['snake']}"
    path_energy = f"{model_dir}/{settings_model['ckpt_dir']}"
    path_model = f"{model_dir}/{settings_model['ckpt_dir_model']}"
    path_red = f"{model_dir}/{settings_model['ckpt_dir_red']}"

    if settings["save"]:
        if settings_model["goal"] == "energy":
            print(f'Saving the model params in {path_energy}.')
            loader.save_to_pkl(path=path_energy, obj=best_params,
                               verbose=1)
        elif settings_model["goal"] == "model":
            print(f'Saving the model params in {path_model}.')
            loader.save_to_pkl(
                path=path_model, obj=best_params, verbose=1
            )
        if settings['training_settings']["bootstrapping"]:
            loader.save_to_pkl(
                path=path_red, obj=best_params_red, verbose=1
            )


if __name__ == "__main__":
    print("Training is starting.")
    # Set up run
    run = Run()
    run["hparams"] = settings

    # Set up the training parameters
    (
        (train_state, train_state_red),
        (loss_func, loss_func_red),
        dataloader,
        settings,
    ) = setup_training(settings)

    # Train the model
    best_params, best_params_red, losses, losses_red = trainer.run_training(
        train_state,
        train_state_red,
        dataloader,
        loss_func,
        loss_func_red,
        settings,
        run,
    )
    print("Training completed.")

    # Save params from model
    save_model(settings)

    # Display training curves
    plotting.plot_training_results(losses)
    if settings["training_settings"]["bootstrapping"]:
        plotting.plot_training_results(losses_red)
