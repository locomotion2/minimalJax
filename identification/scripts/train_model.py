import optax
import stable_baselines3.common.save_util as loader
from aim import Run

from identification.src import trainer
from identification.src import lagranx as lx

from identification.systems import dpendulum_utils, snake_utils

from identification.hyperparams import settings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":

    print('Training is starting.')
    # Set up run
    run = Run()
    run['hparams'] = settings

    # Define training parameters
    stage = settings['stage']
    num_iterations = settings['num_epochs'] * settings['num_batches'] * settings[
        'num_minibatches']
    learning_rate_fn = optax.linear_schedule(
        init_value=settings['lr_start'] * 10 ** (-stage),
        end_value=settings['lr_end'] * 10 ** (-stage),
        transition_steps=num_iterations)
    settings['lr_func'] = learning_rate_fn
    params = None
    params_red = None
    params_model = None
    if settings['reload']:
        params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
        params_model = loader.load_from_pkl(path=settings['ckpt_dir_model'], verbose=1)
        params_red = loader.load_from_pkl(path=settings['ckpt_dir_red'], verbose=1)
        print(f"Params loaded from file: {settings['ckpt_dir']}")

    # Define sys_utils (functions that depend on the particular system)
    if settings['system'] == 'snake':
        settings['sys_utils'] = snake_utils
    elif settings['system'] == 'dpend':
        settings['sys_utils'] = dpendulum_utils

    # Create the training state
    train_state = None
    if settings['goal'] == 'energy':
        train_state = trainer.create_train_state_DeLaNN(settings,
                                                        learning_rate_fn,
                                                        params=params)
    elif settings['goal'] == 'model':
        train_state = trainer.create_train_state_PowNN(settings,
                                                       learning_rate_fn,
                                                       params=params_model)
    train_state_red = trainer.create_train_state_red(settings,
                                                     learning_rate_fn,
                                                     params=params_red)

    # Define data & dataloader TODO: change everything to work with databases
    dataloader = trainer.choose_data_loader(settings)

    # Build the dynamic model and get a loss_func
    loss_func, loss_func_red = lx.build_loss(settings)
    print('Setup completed, training will now begin.')

    # Train the model
    best_params, \
        best_params_red, \
        losses, \
        losses_red = trainer.run_training(train_state,
                                          train_state_red,
                                          dataloader,
                                          loss_func,
                                          loss_func_red,
                                          settings, run)
    print('Training completed.')

    # Save params from model
    if settings['save']:
        if settings['goal'] == 'energy':
            print(f'Saving the model params in {settings["ckpt_dir"]}.')
            loader.save_to_pkl(path=settings['ckpt_dir'],
                               obj=best_params,
                               verbose=1)
        elif settings['goal'] == 'model':
            print(f'Saving the model params in {settings["ckpt_dir_model"]}.')
            loader.save_to_pkl(path=settings['ckpt_dir_model'],
                               obj=best_params,
                               verbose=1)
        if settings['bootstrapping']:
            loader.save_to_pkl(path=settings['ckpt_dir_red'],
                               obj=best_params_red,
                               verbose=1)

    # Display training curves
    trainer.display_results(losses)
    if settings['bootstrapping']:
        trainer.display_results(losses_red)
