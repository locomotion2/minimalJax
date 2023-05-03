from src import lagranx as lx
import jax
import optax
import stable_baselines3.common.save_util as loader
from hyperparams import settings

if __name__ == "__main__":

    print('Compilation completed, training is starting.')

    # Load the data
    batch_train, batch_test = loader.load_from_pkl(path=settings['data_dir'], verbose=1)

    # Create a training state
    stage = 2
    num_iterations = settings['num_epochs'] * settings['num_batches'] * settings['num_minibatches']
    # learning_rate_fn = lambda t: jnp.select([t < num_iterations * 1 // 4,
    #                                          t < num_iterations * 2 // 4,
    #                                          t < num_iterations * 3 // 4,
    #                                          t > num_iterations * 3 // 4],
    #                                         [5e-4, 3e-4, 1e-4, 3e-5])
    learning_rate_fn = optax.linear_schedule(init_value=settings['lr_start']*10**(-stage),
                                             end_value=settings['lr_end']*10**(-stage),
                                             transition_steps=num_iterations)
    settings['lr_func'] = learning_rate_fn
    params = None
    if settings['reload']:
        params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
        print(f"Params loaded from file: {settings['ckpt_dir']}")
    train_state = lx.create_train_state(jax.random.PRNGKey(settings['seed']),
                                        learning_rate_fn,
                                        params=params)

    # Define dataloader
    dataloader = lx.build_simple_dataloader(batch_train, batch_test, settings)
    if settings['generalize']:
        dataloader = lx.build_general_dataloader(batch_train, batch_test, settings)

    print('Setup completed, training will now begin.')

    # Train the model
    best_params, losses = lx.run_training(train_state, dataloader, settings)

    print('Training completed.')

    # Save params from model
    if settings['save']:
        print(f'Saving the model params in {settings["ckpt_dir"]}.')
        loader.save_to_pkl(path=settings['ckpt_dir'], obj=best_params, verbose=1)

    # Display training curves
    lx.display_results(losses)