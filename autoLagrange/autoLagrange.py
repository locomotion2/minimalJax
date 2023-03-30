import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.ode import odeint

from flax import linen as nn, struct
from flax.training import train_state as ts
import optax
from clu import metrics
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from typing import Any


class TrainState(ts.TrainState):
    batch_stats: Any


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        # Input layer
        x_in = self.layer(x, train=train)

        # Skip Layer 1
        x1 = self.layer(x_in, train=train)
        # Skip Layer 2
        x2 = self.layer(x_in + x1, train=train)
        # # Skip Layer 3
        # x3 = self.layer(x_in + x1 + x2, train=train)

        # Narrowing down
        # x_nar = nn.Dense(features=128)(x_in + x1)
        # x_nar = nn.activation.softplus(x_nar)
        # x_nar = nn.Dense(features=64)(x_nar)
        # x_nar = nn.activation.softplus(x_nar)
        # x_nar = nn.Dense(features=4)(x_nar)
        # x_nar = nn.activation.softplus(x_nar)

        # Output layer
        x = nn.Dense(features=2)(x_in + x1 + x2)
        return x

    def layer(self, x, train: bool):
        x = nn.Dense(features=128)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
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


def equation_of_motion(lagrangian, state, t=None):
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


def poormans_solve(x, time_step):
    x_t = jnp.diff(x, axis=0) / time_step
    return x_t


# replace the lagrangian with a parametric model
def learned_lagrangian(params, batch_stats, train_state: TrainState, output='lagrangian', train=True):
    def lagrangian(q, q_t):
        assert q.shape == (2,)
        norm_state = normalize_dp(jnp.concatenate([q, q_t]))
        updates = None
        if train:
            out, updates = train_state.apply_fn({'params': params, 'batch_stats': batch_stats},
                                                x=norm_state, train=True, mutable=['batch_stats'])
        else:
            out = train_state.apply({'params': params, 'batch_stats': batch_stats},
                                    x=norm_state, train=False)
        # out = train_state.apply_fn({'params': params}, norm_state)
        if output == 'energies':
            return out[0], out[1]
        elif output == 'lagrangian':
            return out[0] - out[1]
        elif output == 'potential':
            return out[1]
        elif output == 'updates':
            if not train:
                return NotImplementedError
            return updates
        else:
            raise NotImplementedError

    return lagrangian


def analytic_energies(state):
    q, q_dot = jnp.split(state, 2)
    return lagrangian(q, q_dot, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8, energies=True)


def learned_energies(state, params=None, batch_stats=None, train_state=None, train=True):
    q, q_dot = jnp.split(state, 2)
    T, V = learned_lagrangian(params, batch_stats, train_state,
                              output='energies', train=train)(q, q_dot)
    V_dot = jax.grad(learned_lagrangian(params, batch_stats, train_state,
                                        output='potential', train=train), 1)(q, q_dot)
    return T, V, V_dot


def recon_kin(lagrangian, state):
    q, q_dot = jnp.split(state, 2)
    In = jax.hessian(lagrangian, 1)(q, q_dot)
    T = jnp.abs(1 / 2 * jnp.transpose(q_dot) @ In @ q_dot)
    return T


# define the loss of the model (MSE between predicted q, \dot q and targets)
# @jax.jit
def loss(params, batch_stats, train_state, batch, H_0=0, train=True):
    state, targets = batch

    # Predict the joint accelerations
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params, batch_stats, train_state,
                                                                    train=train)))(state)
    L_acc = jnp.mean((preds - targets) ** 2)

    # Reconstruct the kinetic energy
    T_recon = jax.vmap(partial(recon_kin, learned_lagrangian(params, batch_stats, train_state,
                                                             train=train)))(state)

    # Calculate the energies
    T, V, V_dot = jax.vmap(partial(learned_energies, params=params, batch_stats=batch_stats,
                                   train_state=train_state))(state)
    H = T_recon + V

    # Impose conservation of energy and shape
    L_con = jnp.mean((H - H_0) ** 2)
    L_kin = jnp.mean((T - T_recon) ** 2)

    # Impose q_dot independence on V_dot
    L_pot = jnp.mean(V_dot ** 2)

    # Calculate loss
    L_total = L_acc + 1000 * L_con + L_kin + L_pot

    # Calculate updates
    def dummy_lag(state):
        q, q_dot = jnp.split(state, 2)
        return learned_lagrangian(params, batch_stats, train_state, train=train, output='updates')(q, q_dot)

    updates = jax.vmap(dummy_lag)(state)

    return L_total, updates


def generate_train_test_data_toy(time_step, N, x_0, x_0_test):
    analytical_step = jax.jit(jax.vmap(partial(rk4_step, f_analytical, t=0.0, h=time_step)))

    # x0 = np.array([-0.3*np.pi, 0.2*np.pi, 0.35*np.pi, 0.5*np.pi], dtype=np.float32)

    t = np.arange(N, dtype=np.float32) * time_step  # time steps 0 to N
    x_train = jax.device_get(solve_analytical(x_0, t))  # dynamics for first N time steps
    # %time xt_train = jax.device_get(poormans_solve(x_train, time_step))
    xt_train = jax.device_get(jax.vmap(f_analytical)(x_train))  # time derivatives of each state
    y_train = jax.device_get(analytical_step(x_train))  # analytical next step
    # print(jnp.mean((xt_train - xt_train_ref[:N-1]) ** 2))

    # t_test = np.arange(N, 2*N, dtype=np.float32) * time_step # time steps N to 2N
    x_test = jax.device_get(solve_analytical(x_0_test, t))  # dynamics for next N time steps
    # %time xt_test = jax.device_get(poormans_solve(x_test, time_step))
    xt_test = jax.device_get(jax.vmap(f_analytical)(x_test))  # time derivatives of each state
    y_test = jax.device_get(analytical_step(x_test))  # analytical next step

    # Subsample arrays
    x_train = x_train[0::100]
    xt_train = xt_train[0::100]
    x_test = x_test[0::100]
    xt_test = xt_test[0::100]

    return x_train, xt_train, x_test, xt_test, y_train, y_test


def train_test_data_generator(batch_size, minibatch_per_batch, time_step):
    eqs_motion = jax.jit(jax.vmap(f_analytical))

    # eqs_motion = jax.jit(jax.vmap(partial(poormans_solve, time_step)))
    def get_derivative_dataset(rng):
        # randomly sample inputs

        y0 = jnp.concatenate([
            jax.random.uniform(rng, (batch_size * minibatch_per_batch, 2)) * 2.0 * np.pi,
            (jax.random.uniform(rng + 1, (batch_size * minibatch_per_batch, 2)) - 0.5) * 10 * 2
        ], axis=1)

        return y0, eqs_motion(y0)

    return get_derivative_dataset


def vizualize_train_test_data(train_vis, test_vis):
    vel_angle = lambda data: (np.arctan2(data[:, 3], data[:, 2]) / np.pi + 1) / 2
    vel_color = lambda vangle: np.stack([np.zeros_like(vangle), vangle, 1 - vangle]).T
    train_colors = vel_color(vel_angle(train_vis))
    test_colors = vel_color(vel_angle(test_vis))

    # plot
    SCALE = 80
    WIDTH = 0.006
    plt.figure(figsize=[8, 4], dpi=120)
    plt.subplot(1, 2, 1)
    plt.title("Train data")
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.quiver(*train_vis.T, color=train_colors, scale=SCALE, width=WIDTH)

    plt.subplot(1, 2, 2)
    plt.title("Test data")
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.quiver(*test_vis.T, color=test_colors, scale=SCALE, width=WIDTH)

    plt.tight_layout()
    plt.show()


@partial(jax.jit, static_argnums=2)
def train_step(state: TrainState, batch, learning_rate_fn):
    traj, traj_truth = batch

    def loss_fn(params):
        # logits = CNN().apply({'params': params}, imgs)
        # one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
        return loss(params, state.batch_stats, state, (traj, traj_truth), H_0=0)

    # variables = {'params': state.params, 'batch_stats': state.batch_stats}
    (loss_value, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    state = state.replace(batch_stats=updates['batch_stats'])
    lr = learning_rate_fn(state.step)
    metrics = {'learning_rate': lr, 'loss': loss_value}
    return state, metrics


@jax.jit
def eval_step(state, test_batch):
    traj_test, traj_test_truth = test_batch
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    loss_value = loss(variables, state, (traj_test, traj_test_truth), H_0=0)
    return {'loss': loss_value}


def train_one_epoch(state, learning_rate_fn, batch, batch_size, minibatch_per_batch, epoch, total_epochs):
    # """Train for 1 epoch on the training set."""
    # batch_metrics = []
    # for cnt, (imgs, labels) in enumerate(dataloader):
    #     state, metrics = train_step(state, imgs, labels)
    #     batch_metrics.append(metrics)
    #
    # # Aggregate the metrics
    # batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    # epoch_metrics_np = {
    #     k: np.mean([metrics[k] for metrics in batch_metrics_np])
    #     for k in batch_metrics_np[0]
    # }
    epoch_loss = 0.0
    num_samples = 0
    for minibatch_num in range(minibatch_per_batch):
        # fraction = (epoch + minibatch_num/minibatch_per_batch)/total_epochs
        minibatch = (batch[0][minibatch_num * batch_size:(minibatch_num + 1) * batch_size],
                     batch[1][minibatch_num * batch_size:(minibatch_num + 1) * batch_size])
        # opt_state, params = update_derivative(fraction, opt_state, batch_data, 1e-6)
        state, metrics = train_step(state, minibatch, learning_rate_fn)

        # print(f"Epoch={epoch},"
        #       f" minibatch_loss={metrics['loss']:.6f},"
        #       f" lr={metrics['learning_rate']:.6f}")
        epoch_loss += metrics['loss']
        num_samples += batch_size
    metrics_epoch = {'loss': epoch_loss / minibatch_per_batch, 'learning_rate': learning_rate_fn(state.step)}
    return state, metrics_epoch


def evaluate_model(state, test_batch):
    """Evaluate on the validation set."""
    metrics = eval_step(state, test_batch)
    metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    # metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
    return metrics


# This one will keep things nice and tidy compared to our previous examples
def create_train_state(key, learning_rate_fn, params=None, batch_stats=None):
    network = MLP()
    if params is None:
        variables = network.init(key, random.normal(key, (4,)), train=False)
        params = variables['params']
        if batch_stats is None:
            batch_stats = variables['batch_stats']
    adam_opt = optax.adam(learning_rate=learning_rate_fn)
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return TrainState.create(apply_fn=network.apply, params=params, batch_stats=batch_stats, tx=adam_opt), network
