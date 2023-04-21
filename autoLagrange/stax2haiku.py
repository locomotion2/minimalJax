import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial  # reduces arguments to function by making some subset implicit

from jax.example_libraries import stax
from jax.example_libraries import optimizers

from flax import linen as nn
from flax.training import train_state as ts
import optax

# visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from functools import partial
from PIL import Image


def lagrangian(q, q_dot, m1, m2, l1, l2, g):
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

    return T - V


def f_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
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
def solve_autograd(initial_state, times, m1=1, m2=1, l1=1, l2=1, g=9.8):
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

def learned_lagrangian(params, train_state):
    def lagrangian(q, q_t):
        assert q.shape == (2,)
        state = normalize_dp(jnp.concatenate([q, q_t]))
        # return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
        return jnp.squeeze(train_state.apply_fn({'params': params}, x=state))

    return lagrangian


# define the loss of the model (MSE between predicted q, \dot q and targets)
@jax.jit
def loss(params, train_state, batch):
    state, targets = batch
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params, train_state)))(state)
    return jnp.mean((preds - targets) ** 2)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.activation.softplus(x)
        x = nn.Dense(features=128)(x)
        x = nn.activation.softplus(x)
        x = nn.Dense(features=1)(x)
        return x


def create_train_state(key, learning_rate_fn):
    network = MLP()
    params = network.init(key, jax.random.normal(key, (4,)))['params']
    adam_opt = optax.adam(learning_rate=learning_rate_fn)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=adam_opt)

@jax.jit
# @partial(jax.jit, static_argnums=2)
def train_step(state, batch):
    @jax.jit
    def loss_fn(params):
        return loss(params, state, batch)

    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    # lr = learning_rate_fn(state.step)
    # metrics = {'learning_rate': lr, }
    metrics = {'loss': loss_value}
    return state, metrics


@jax.jit
def eval_step(state, test_batch):
    loss_value = loss(state.params, state, test_batch)
    return {'loss': loss_value}

def run_flax(x_train, xt_train, x_test, xt_test):
    # numbers in comments denote stephan's settings
    batch_size = 100
    test_every = 10
    num_batches = 15

    train_losses = []
    test_losses = []

    # adam w learn rate decay
    learning_rate_fn = lambda t: jnp.select([t < batch_size * (num_batches // 3),
                                             t < batch_size * (2 * num_batches // 3),
                                             t > batch_size * (2 * num_batches // 3)],
                                            [1e-3, 3e-4, 1e-4])

    train_state = create_train_state(jax.random.PRNGKey(0), learning_rate_fn)

    for iteration in range(batch_size * num_batches + 1):
        train_state, train_metrics = train_step(train_state, (x_train, xt_train))

        if iteration % batch_size == 0:

            train_loss = train_metrics['loss']
            train_losses.append(train_loss)

            test_loss = eval_step(train_state, (x_test, xt_test))['loss']
            test_losses.append(test_loss)

            if iteration % (batch_size * test_every) == 0:
                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

@jax.jit
def update_derivative(i, opt_state, batch, get_params, opt_update):
  params = get_params(opt_state)
  return opt_update(i, jax.grad(loss)(params, batch, None), opt_state)

if __name__ == "__main__":
    time_step = 0.01
    N = 1500

    x0 = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)
    t = np.arange(N, dtype=np.float32)  # time steps 0 to N
    x_train = solve_analytical(x0, t)  # dynamics for first N time steps
    xt_train = jax.vmap(f_analytical)(x_train)  # time derivatives of each state

    noise = np.random.RandomState(0).randn(x0.size)
    t_test = np.arange(N, 2 * N, dtype=np.float32)  # time steps N to 2N
    x_test = solve_analytical(x0, t_test)  # dynamics for next N time steps
    xt_test = jax.vmap(f_analytical)(x_test)  # time derivatives of each state

    x_train = jax.vmap(normalize_dp)(x_train)
    x_test = jax.vmap(normalize_dp)(x_test)

    run_flax(x_train, xt_train, x_test, xt_test)