import optax
import stable_baselines3.common.save_util as loader
from aim import Run

from src import trainer
from src import lagranx as lx

from src import dpendulum_utils
from src import snake_utils

from hyperparams import settings

if __name__ == "__main__":

    print('Training is starting.')
    # Set up run
    run = Run()
    run['hparams'] = settings

    # Define training parameters
    stage = 0
    num_iterations = settings['num_epochs'] * settings['num_batches'] * settings[
        'num_minibatches']
    learning_rate_fn = optax.linear_schedule(
        init_value=settings['lr_start'] * 10 ** (-stage),
        end_value=settings['lr_end'] * 10 ** (-stage),
        transition_steps=num_iterations)
    settings['lr_func'] = learning_rate_fn
    params = None
    if settings['reload']:
        params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
        print(f"Params loaded from file: {settings['ckpt_dir']}")

    # Define sys_utils (functions that depend on the particular system)
    if settings['system'] == 'snake':
        settings['sys_utils'] = snake_utils
    elif settings['system'] == 'dpend':
        settings['sys_utils'] = dpendulum_utils

    # Create the training state
    train_state = trainer.create_train_state(settings,
                                             learning_rate_fn,
                                             params=params)

    # Define data & dataloader TODO: change everything to work with databases
    dataloader = trainer.choose_data_loader(settings)

    # Build the dynamic model and get a loss_func
    loss_func = lx.build_loss(settings)
    print('Setup completed, training will now begin.')

    # Train the model
    best_params, losses = trainer.run_training(train_state,
                                               dataloader,
                                               loss_func,
                                               settings, run)
    print('Training completed.')

    # Save params from model
    if settings['save']:
        print(f'Saving the model params in {settings["ckpt_dir"]}.')
        loader.save_to_pkl(path=settings['ckpt_dir'], obj=best_params, verbose=1)

    # Display training curves
    trainer.display_results(losses)
