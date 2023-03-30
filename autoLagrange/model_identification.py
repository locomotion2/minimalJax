import autoLagrange as al
import numpy as np
import jax.numpy as jnp
import jax, optax
import matplotlib.pyplot as plt
import stable_baselines3.common.save_util as loader
from copy import deepcopy as copy

if __name__ == "__main__":
    # Generate training data
    time_step = 0.001
    N = 1000 * 100
    x_0 = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)
    noise = np.random.RandomState(0).randn(x_0.size)
    x_0_test = x_0 + 1e-3 * noise
    # x_train, xt_train, x_test, xt_test, y_train, y_test = al.generate_train_test_data(time_step, N, x_0, x_0_test)

    # # Show training and test data
    # train_vis = jax.vmap(al.normalize_dp)(x_train)
    # test_vis = jax.vmap(al.normalize_dp)(x_test)
    # al.vizualize_train_test_data(train_vis, test_vis)
    #
    # # Standardize
    # x_train = jax.device_put(jax.vmap(al.normalize_dp)(x_train))
    # x_test = jax.device_put(jax.vmap(al.normalize_dp)(x_test))

    # Train
    seed = 0  # needless to say these should be in a config or defined like flags
    test_every = 10
    num_batches = 150 * 5

    eval_every = 1
    report_every = 10
    total_epochs = 100
    batch_size = 128 * 1
    minibatch_per_batch = 1

    # learning_rate_fn = optax.linear_schedule(1e-3, 3e-4, batch_size * num_batches + 1, transition_begin=0)
    # learning_rate_fn = lambda count: jnp.select([count < total_epochs * (minibatch_per_batch // 6),
    #                                              count < total_epochs * (2 * minibatch_per_batch // 6),
    #                                              count < total_epochs * (3 * minibatch_per_batch // 6),
    #                                              count < total_epochs * (4 * minibatch_per_batch // 6),
    #                                              count < total_epochs * (5 * minibatch_per_batch // 6),
    #                                              count > total_epochs * (5 * minibatch_per_batch // 6)],
    #                                             [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    learning_rate_fn = lambda count: 1e-3

    reload = False
    if reload:
        ckpt_dir = 'tmp/flax-checkpointing'
        params = loader.load_from_pkl(path=ckpt_dir, verbose=1)
        train_state, _ = al.create_train_state(jax.random.PRNGKey(seed), 0, params=params)
    else:
        train_state, model = al.create_train_state(jax.random.PRNGKey(seed), learning_rate_fn)

    train_losses = []
    test_losses = []

    best_loss = np.inf
    best_params = None
    random_key = jax.random.PRNGKey(0)
    data_generator = al.train_test_data_generator(batch_size, minibatch_per_batch, time_step=0)
    for epoch in range(1, total_epochs):
        batch = data_generator(random_key)
        train_state, train_metrics = al.train_one_epoch(train_state, learning_rate_fn, batch,
                                                        batch_size, minibatch_per_batch, epoch,
                                                        total_epochs)
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            best_params = copy(train_state.params)

        train_losses.append(train_metrics['loss'])

        random_key += 10
        print(f"Epoch={epoch},"
              f" train_loss={train_metrics['loss']:.6f},"
              # f" test_loss={test_metrics['loss']:.6f},"
              f" lr={train_metrics['learning_rate']:.6f}")
        # if epoch % eval_every == 0:
        #     test_batch = data_generator(random_key)
        #     test_metrics = al.evaluate_model(train_state, test_batch)
        #     train_losses.append(train_metrics['loss'])
        #     test_losses.append(test_metrics['loss'])
        #     if iteration % (batch_size * test_every) == 0:
        #         # print(f"Train epoch: {epoch}, loss: {train_metrics}")
        #         # print(f"Test epoch: {epoch}, loss: {test_metrics}")
        #         print(f"Step={iteration},"
        #               f" train_loss={train_metrics['loss']:.6f},"
        #               f" test_loss={test_metrics['loss']:.6f},"
        #               f" lr={train_metrics['learning_rate']:.6f}")

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    # plt.plot(test_losses, label='Test loss')
    plt.yscale('log')
    plt.ylim(0, None)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()

    # Save params from model
    ckpt_dir = 'tmp/flax-checkpointing'
    loader.save_to_pkl(path=ckpt_dir, obj=best_params, verbose=1)
