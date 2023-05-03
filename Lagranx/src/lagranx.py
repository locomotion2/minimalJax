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

# from src import dpend_model_cramer as model
from Lagranx.src import dpend_model_arne as model


class MLP(nn.Module):
    # def __init__(self, h_dim, *args, **kwargs):
    #     self.h_dim = h_dim
    #     super().__init__(*args, **kwargs)

    @nn.compact
    def __call__(self, x):
        # skip_layers = 0
        size = 128 * 5
        x_prev = self.layer(x, features=size)
        # x_cur = self.layer(x_prev, features=size)
        # # x_cur = self.layer(x_cur, features=size)
        # for i in range(skip_layers):
        #     x_temp = x_cur
        #     x_cur = self.layer(x_cur + x_prev, features=size)
        #     x_prev = x_temp

        x_out = nn.Dense(features=2)(x_prev)
        return x_out

    def layer(self, x, features=128):
        x = nn.Dense(features=features)(x)
        x = nn.activation.softplus(x)
        return x


def equation_of_motion(lagrangian, state):
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate([q_t, q_tt])


# def solve_lagrangian(lagrangian, initial_state, **kwargs):
#     # We currently run odeint on CPUs only, because its cost is dominated by
#     # control flow, which is slow on GPUs.
#     @partial(jax.jit, backend='cpu')
#     def f(initial_state):
#         return odeint(partial(equation_of_motion, lagrangian),
#                       initial_state, **kwargs)
#
#     return f(initial_state)


# Double pendulum dynamics via the rewritten Euler-Lagrange
# @partial(jax.jit, backend='cpu')
# def solve_autograd(initial_state, times, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8):
#     L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
#     return solve_lagrangian(L, initial_state, t=times, rtol=1e-10, atol=1e-10)


# Double pendulum dynamics via analytical forces taken from Diego's blog
@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    return odeint(model.f_analytical, initial_state, t=times, rtol=1e-8, atol=1e-9)


# def rk4_step(f, x, t, h):
#     # one step of runge-kutta integration
#     k1 = h * f(x, t)
#     k2 = h * f(x + k1 / 2, t + h / 2)
#     k3 = h * f(x + k2 / 2, t + h / 2)
#     k4 = h * f(x + k3, t + h)
#     return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def learned_lagrangian(params, train_state, output='lagrangian'):
    def lagrangian(q, q_t):
        assert q.shape == (2,)
        state = jnp.concatenate([q, q_t])
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
    H = T_rec + V

    # predict joint accelerations
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params, train_state)))(state)
    L_acc = jnp.mean((preds - targets) ** 2)

    # impose energy conservation
    L_con = jnp.mean(H ** 2)

    # impose clean derivative
    L_kin = jnp.mean((T - T_rec) ** 2)

    # impose independence form q_dot on V due to mechanical system
    L_pot = jnp.mean(V_dot ** 2)

    # # impose positive kin. energy
    # L_pos = jnp.mean(jnp.clip(T, a_min=None, a_max=0) ** 2)

    return L_acc + ((L_kin + L_pot) + 10000 * L_con)


@jax.jit
def loss_sample(pair, params=None, train_state=None):
    state, targets = pair

    # calculate energies
    T, V, V_dot = partial(learned_energies, params=params, train_state=train_state)(state)
    T_rec = partial(kin_energy_lagrangian, lagrangian=learned_lagrangian(params, train_state))(state)
    H = T_rec + V

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
        num_subbatches = 1
        x, xt = batch
        x_subbatches = jnp.split(x, num_subbatches)
        xt_subbatches = jnp.split(xt, num_subbatches)

        loss_val = 0
        for subbatch in range(num_subbatches):
            subbatch_current = (x_subbatches[subbatch], xt_subbatches[subbatch])
            loss_val += loss(params, state, subbatch_current)
        return loss_val / num_subbatches

    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    lr = learning_rate_fn(state.step)
    metrics = {'learning_rate': lr, 'loss': loss_value}
    return state, metrics


@jax.jit
def eval_step(state, test_batch):
    loss_value = loss(state.params, state, test_batch)
    return {'loss': loss_value}


def run_training(train_state, dataloader, settings):
    # batch_size = settings['batch_size']
    test_every = settings['test_every']
    num_batches = settings['num_batches']
    num_minibatches = settings['num_minibatches']
    num_epochs = settings['num_epochs']
    lr_func = settings['lr_func']
    early_stopping_gain = settings['es_gain']

    train_losses = []
    test_losses = []
    best_loss = np.inf
    best_params = None

    try:
        # Iterate over every iteration
        epoch_loss_last = np.inf
        epoch = 0
        while epoch < num_epochs:
            random_key = jax.random.PRNGKey(settings['seed'])
            epoch_loss = 0
            train_metrics = None
            batch_train, batch_test = dataloader(random_key)
            random_key += 10

            x_train, xt_train = batch_train
            x_train_minibatches = jnp.split(x_train, num_minibatches)
            xt_train_minibatches = jnp.split(xt_train, num_minibatches)
            for batch in range(num_batches):
                train_loss = 0
                for minibatch in range(num_minibatches):
                    minibatch_current = (x_train_minibatches[minibatch], xt_train_minibatches[minibatch])
                    train_state, train_metrics = train_step(train_state, minibatch_current, lr_func)
                    train_loss += train_metrics['loss'] / num_minibatches
                    # print(f"minibatch length: {train_state.step}")

                # When a batch is done
                epoch_loss += train_loss / num_batches

                # Check for the best params
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = copy(train_state.params)

            test_loss = 0
            if epoch_loss > epoch_loss_last * early_stopping_gain:
                print(f'Early stopping! loss: {epoch_loss}. Now resetting.')
                settings['seed'] += 10
                train_state = create_train_state(settings['seed'], settings['lr_func'], params=best_params)
                epoch = 0
                continue
            else:
                train_losses.append(epoch_loss)
                test_loss = eval_step(train_state, batch_test)['loss']
                test_losses.append(test_loss)
            epoch_loss_last = epoch_loss

            # Output results now and then for debugging
            if epoch % test_every == 0:
                print(
                    f"epoch={epoch}, train_loss={epoch_loss:.6f}, test_loss={test_loss:.6f}, lr={train_metrics['learning_rate']:.6f}")
            epoch += 1

    except KeyboardInterrupt:
        # Save params from model
        # print(f'Saving the model params in {settings["ckpt_dir"]}.')
        print(f'Terminating learning!')
        # loader.save_to_pkl(path=settings['ckpt_dir'], obj=best_params, verbose=1)

    return best_params, (train_losses, test_losses)


def generate_trajectory_data(x0, t_window, section_num, key):
    x_traj = None
    xt_traj = None
    x_start = x0
    for section in range(section_num):
        x_traj_sec = solve_analytical(x_start, t_window)
        x_start = x_traj_sec[-1, :]
        # x_start = normalize_dp(x_start)
        x_traj_sec = jax.random.permutation(key, x_traj_sec)
        xt_traj_sec = jax.vmap(model.f_analytical)(x_traj_sec)

        if jnp.any(jnp.isnan(x_traj_sec)) or jnp.any(jnp.isnan(xt_traj_sec)):
            print('Problem!')
            print(x_traj_sec)
            return

        if x_traj is None:
            x_traj = x_traj_sec
            xt_traj = xt_traj_sec
        else:
            x_traj = jnp.append(x_traj, x_traj_sec, axis=0)
            xt_traj = jnp.append(xt_traj, xt_traj_sec, axis=0)
    return x_traj, xt_traj, x_start


def generate_data(settings):
    # load settings
    N = settings['data_size']
    x0 = settings['starting_point']
    time_step = settings['time_step']
    section_num = settings['sections_num']
    key = jax.random.PRNGKey(settings['seed'])

    # create time
    t_window = np.arange(N / section_num, dtype=np.float32) * time_step
    # t_train = np.arange(N / section_num, dtype=np.float32) * time_step  # time steps 0 to N
    # t_test = np.arange(N, 2 * N, dtype=np.float32) * time_step  # time steps N to 2N
    # t_train_sec = jnp.split(t_train, section_num)
    # t_test_sec = jnp.split(t_test, section_num)

    # create training data
    x_train, xt_train, x0_test = generate_trajectory_data(x0, t_window, section_num, key)
    print('Generated train data.')

    # create test data
    # noise = np.random.RandomState(0).randn(x0.size)
    # x0_test = x0 + noise * 1e-3
    x_test, xt_test, _ = generate_trajectory_data(x0_test, t_window, section_num, key)
    print('Generated test data.')

    # normalize data
    x_train = jax.vmap(model.normalize)(x_train)
    x_test = jax.vmap(model.normalize)(x_test)
    print('Normalized data.')

    print(jnp.amax(x_test, axis=0))
    print(jnp.amin(x_test, axis=0))

    return (x_train, xt_train), (x_test, xt_test)


def build_simple_dataloader(batch_train, batch_test, settings):
    def dataloader(key):
        return batch_train, batch_test

    return dataloader


def build_general_dataloader(batch_train, batch_test, settings):
    # eqs_motion = jax.jit(jax.vmap(partial(poormans_solve, time_step)))
    batch_size = settings['batch_size']
    num_minibathces = settings['num_minibatches']
    data_size = batch_size * num_minibathces
    eqs_motion = jax.jit(jax.vmap(model.f_analytical))

    def dataloader(key):
        # randomly sample inputs
        y0 = jnp.concatenate([
            jax.random.uniform(key, (data_size, 2)) * 2.0 * np.pi,
            (jax.random.uniform(key + 10, (data_size, 1)) - 0.5) * 10 * 2,
            (jax.random.uniform(key + 20, (data_size, 1)) - 0.5) * 10 * 4], axis=1)
        y0 = jax.vmap(model.normalize)(y0)

        # return (y0, eqs_motion(y0)), batch_test
        return (y0, eqs_motion(y0)), batch_test

    return dataloader


def display_results(losses):
    train_losses, test_losses = losses
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.yscale('log')
    # plt.ylim(None, 1000)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()
