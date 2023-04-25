import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from flax import linen as nn
from flax.training import train_state as ts
import optax

import matplotlib.pyplot as plt
from functools import partial
from copy import deepcopy as copy
import stable_baselines3.common.save_util as loader


class MLP(nn.Module):
    # @nn.compact
    # def __call__(self, x):
    #     x1 = self.layer(x, features=64)
    #     x2 = self.layer(x1, features=128)
    #     # x3 = self.layer(x2, features=256)
    #
    #     # x2_skip = self.layer(x3, features=128)
    #     x1_skip = self.layer(x2, features=64)
    #     x0_skip = self.layer(x1 + x1_skip, features=4)
    #
    #     # Output Layer
    #     x_out = self.layer(x0_skip + x, features=1)
    #
    #     return x_out

    # @nn.compact
    # def __call__(self, x):
    #     x_1 = self.layer(x, features=128)
    #     x_2 = self.layer(x_1, features=128)
    #     x_out = self.layer(x_1 + x_2, features=4)
    #     x_out = self.layer(x_out + x, features=2)
    #     return x_out

    # @nn.compact
    # def __call__(self, x):
    #     x = self.layer(x, features=128)
    #     x = self.layer(x, features=128)
    #     x_out = self.layer(x, features=2)
    #     return x_out

    @nn.compact
    def __call__(self, x):
        x = self.layer(x, features=256)
        x = self.layer(x, features=256)
        x_out = self.layer(x, features=2)
        return x_out

    def layer(self, x, features=128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.softplus(x)
        return x


def lagrangian(q, q_dot, m1, m2, l1, l2, g, energies=False):
    t1, t2 = q  # theta 1 and theta 2
    w1, w2 = q_dot  # omega 1 and omega 2

    # kinetic energy (T)
    T1 = 0.5 * m1 * (l1 * w1) ** 2
    T2 = 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 +
                     2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
    T = T1 + T2

    # potential energy (V)
    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2

    if energies:
        return T, V

    return T - V


def f_analytical(state, t=0, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8):
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
    a2 = (l1 / l2) * jnp.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * jnp.sin(t1 - t2) - \
         (g / l1) * jnp.sin(t1)
    f2 = (l1 / l2) * (w1 ** 2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return jnp.stack([w1, w2, g1, g2])


def equation_of_motion(lagrangian, state):
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate([q_t, q_tt])


def solve_lagrangian(lagrangian, initial_state, **kwargs):
    # We currently run odeint on CPUs only, because its cost is dominated by
    # control flow, which is slow on GPUs.
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(equation_of_motion, lagrangian),
                      initial_state, **kwargs)

    return f(initial_state)


# Double pendulum dynamics via the rewritten Euler-Lagrange
@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8):
    L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    return solve_lagrangian(L, initial_state, t=times, rtol=1e-10, atol=1e-10)


# Double pendulum dynamics via analytical forces taken from Diego's blog
@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)


def normalize_dp(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])


def rk4_step(f, x, t, h):
    # one step of runge-kutta integration
    k1 = h * f(x, t)
    k2 = h * f(x + k1 / 2, t + h / 2)
    k3 = h * f(x + k2 / 2, t + h / 2)
    k4 = h * f(x + k3, t + h)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def learned_lagrangian(params, train_state, output='lagrangian'):
    def lagrangian(q, q_t):
        assert q.shape == (2,)
        state = jnp.concatenate([q, q_t])
        # state = normalize_dp(jnp.concatenate([q, q_t]))
        out = train_state.apply_fn({'params': params}, x=state)
        if output == 'energies':
            return out[0], out[1]
        elif output == 'lagrangian':
            return out[0] - out[1]
        elif output == 'potential':
            return out[1]

    return lagrangian


def learned_energies(state, params=None, train_state=None):
    q = state[0:2]
    q_dot = state[2:]
    T, V = learned_lagrangian(params, train_state, output='energies')(q, q_dot)
    V_dot = jax.grad(learned_lagrangian(params, train_state, output='potential'), 1)(q, q_dot)
    return T, V, V_dot


def analytic_energies(state):
    q = state[0:2]
    q_dot = state[2:]
    return lagrangian(q, q_dot, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8, energies=True)


def kin_energy_lagrangian(state, lagrangian=None):
    q, q_dot = jnp.split(state, 2)
    In = jax.hessian(lagrangian, 1)(q, q_dot)
    T = jnp.abs(1 / 2 * jnp.transpose(q_dot) @ In @ q_dot)
    return T


# define the loss of the model (MSE between predicted q, \dot q and targets)
@jax.jit
def loss(params, train_state, batch):
    state, targets = batch

    # calculate energies
    T, V, V_dot = jax.vmap(partial(learned_energies, params=params, train_state=train_state))(state)
    T_rec = jax.vmap(partial(kin_energy_lagrangian, lagrangian=learned_lagrangian(params, train_state)))(state)
    H = T + V

    # predict joint accelerations
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params, train_state)))(state)
    L_acc = jnp.mean((preds - targets) ** 2)

    # # impose energy conservation
    # L_con = jnp.mean(H ** 2)

    # impose clean derivative
    L_kin = jnp.mean((T - T_rec) ** 2)

    # # impose independence form q_dot on V due to mechanichal system
    # L_pot = jnp.mean(V_dot ** 2)
    #
    # # impose positive kin. energy
    # L_pos = jnp.mean(jnp.clip(T, a_min=None, a_max=0) ** 2)

    return L_acc + 0.0 * L_kin


@jax.jit
def loss_sample(pair, params=None, train_state=None):
    state, targets = pair

    # calculate energies
    T, V, V_dot = partial(learned_energies, params=params, train_state=train_state)(state)
    T_rec = partial(kin_energy_lagrangian, lagrangian=learned_lagrangian(params, train_state))(state)
    H = T + V

    # predict joint accelerations
    preds = partial(equation_of_motion, learned_lagrangian(params, train_state))(state)
    L_acc = jnp.mean((preds - targets) ** 2)
    # print(L_acc.shape)

    # impose energy conservation
    L_con = H ** 2
    # print(L_con.shape)

    # impose clean derivative
    L_kin = (T - T_rec) ** 2
    # print(L_kin.shape)

    # impose independence form q_dot on V due to mechanichal system
    L_pot = jnp.mean(V_dot ** 2)
    # print(L_pot.shape)

    # impose positive kin. energy
    L_pos = jnp.clip(T, a_min=None, a_max=0) ** 2
    # print(L_pos.shape)

    L_total = L_acc + L_kin + L_pos + L_con
    # print(L_total.shape)
    return L_total, L_acc, L_con, L_kin, L_pot, L_pos


def create_train_state(key, learning_rate_fn, params=None):
    network = MLP()
    if params is None:
        params = network.init(key, jax.random.normal(key, (4,)))['params']
    adam_opt = optax.adamw(learning_rate=learning_rate_fn)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)


# @jax.jit
@partial(jax.jit, static_argnums=2)
def train_step(state, batch, learning_rate_fn):
    @jax.jit
    def loss_fn(params):
        return loss(params, state, batch)

    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    lr = learning_rate_fn(state.step)
    metrics = {'learning_rate': lr, 'loss': loss_value}
    # metrics = {'loss': loss_value}
    return state, metrics


@jax.jit
def eval_step(state, test_batch):
    loss_value = loss(state.params, state, test_batch)
    return {'loss': loss_value}


def run_training(train_state, batch_train, batch_test, settings):
    # batch_size = settings['batch_size']
    test_every = settings['test_every']
    num_batches = settings['num_batches']
    num_minibatches = settings['num_minibatches']
    num_epochs = settings['num_epochs']
    lr_func = settings['lr_func']

    train_losses = []
    test_losses = []
    best_loss = np.inf
    best_params = None

    try:
        # Iterate over every iteration
        for epoch in range(num_epochs + 1):
            train_loss = 0
            x_train, xt_train = batch_train
            x_train_minibatches = jnp.split(x_train, num_minibatches)
            xt_train_minibatches = jnp.split(xt_train, num_minibatches)
            for batch in range(num_batches + 1):
                for minibatch in range(num_minibatches):
                    minibatch_current = (x_train_minibatches[minibatch], xt_train_minibatches[minibatch])
                    train_state, train_metrics = train_step(train_state, minibatch_current, lr_func)
                    train_loss += train_metrics['loss'] / num_minibatches

                # When a batch is done
                train_loss = train_loss
                print(f"epoch:{epoch}, batch:{batch}, loss:{train_loss:.6f}")
                train_losses.append(train_loss)

                # Check for the best params
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = copy(train_state.params)

            # If error explosion
            # if train_loss > best_loss*100:
            #     print('Error explosion!')
            #     train_state.params = copy(best_params)

            test_loss = eval_step(train_state, batch_test)['loss']
            test_losses.append(test_loss)

            # Output results now and then for debugging
            if epoch % test_every == 0:
                print(
                    f"epoch={epoch}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}, lr={train_metrics['learning_rate']:.6f}")

    except KeyboardInterrupt:
        # Save params from model
        print(f'Saving the model params in {settings["ckpt_dir"]}.')
        loader.save_to_pkl(path=settings['ckpt_dir'], obj=best_params, verbose=1)

    return best_params, (train_losses, test_losses)


def generate_data(settings):
    # load settings
    N = settings['data_size']
    x0 = settings['starting_point']
    time_step = settings['time_step']
    key = jax.random.PRNGKey(settings['seed'])

    # create time
    t_train = np.arange(N, dtype=np.float32) * time_step  # time steps 0 to N
    t_test = np.arange(N, 2 * N, dtype=np.float32) * time_step  # time steps N to 2N

    # create train data
    x_train = solve_analytical(x0, t_train)  # dynamics for first N time steps
    x_train = jax.random.permutation(key, x_train)  # randomly change the order of samples
    xt_train = jax.vmap(f_analytical)(x_train)  # time derivatives of each state
    print('Generated train data.')

    # create test data
    noise = np.random.RandomState(0).randn(x0.size)
    # x0_test = x0 + noise * 1e-3
    x0_test = x0
    x_test = solve_analytical(x0_test, t_test)  # dynamics for next N time steps
    x_test = jax.random.permutation(key, x_test)  # randomly change the order of samples
    xt_test = jax.vmap(f_analytical)(x_test)  # time derivatives of each state
    print('Generated test data.')

    # normalize data
    x_train = jax.vmap(normalize_dp)(x_train)
    x_test = jax.vmap(normalize_dp)(x_test)
    print('Normalized data.')

    return (x_train, xt_train), (x_test, xt_test)


def display_results(losses):
    train_losses, test_losses = losses
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.yscale('log')
    plt.ylim(None, 200)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Define all settings
    settings = {'batch_size': 100,
                'test_every': 1,
                'num_batches': 100,
                'num_minibatches': 1000,
                'num_epochs': 200,
                'time_step': 0.01,
                'data_size': 1000 * 100,
                'starting_point': np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32),
                'data_dir': 'tmp/data',
                'reload': False,
                'ckpt_dir': 'tmp/flax-checkpointing',
                'seed': 0
                }

    # Load the data
    batch_train, batch_test = loader.load_from_pkl(path=settings['data_dir'], verbose=1)
    # print(batch_train)

    # Create a training state
    num_iterations = settings['batch_size'] * settings['num_batches'] * settings['num_minibatches']
    learning_rate_fn = lambda t: jnp.select([t < num_iterations * 1 // 4,
                                             t < num_iterations * 2 // 4,
                                             t < num_iterations * 3 // 4,
                                             t > num_iterations * 3 // 4],
                                            [1e-4, 3e-5, 1e-5, 3e-6])
    settings['lr_func'] = learning_rate_fn
    params = None
    if settings['reload']:
        params = loader.load_from_pkl(path=settings['ckpt_dir'], verbose=1)
    train_state = create_train_state(jax.random.PRNGKey(settings['seed']),
                                     learning_rate_fn,
                                     params=params)

    # Train the model
    best_params, losses = run_training(train_state, batch_train, batch_test, settings)

    # Save params from model
    print(f'Saving the model params in {settings["ckpt_dir"]}.')
    loader.save_to_pkl(path=settings['ckpt_dir'], obj=best_params, verbose=1)

    # Display training curves
    display_results(losses)
