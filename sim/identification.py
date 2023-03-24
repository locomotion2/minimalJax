import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial  # reduces arguments to function by making some subset implicit

from jax.example_libraries import stax
from jax.example_libraries import optimizers

# New flax stuff
from jax import random
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state
import optax

# visualization
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from moviepy.editor import ImageSequenceClip
from functools import partial


# import proglog
# from PIL import Image

# from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)


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


@partial(jax.jit, static_argnums=0)
def apply_fun(ma, w, x):
    return ma.apply(w, x)


# replace the lagrangian with a parametric model
def learned_lagrangian(params, energies=False):
    def lagrangian(q, q_t, energies=energies):
        assert q.shape == (2,)
        state = normalize_dp(jnp.concatenate([q, q_t]))
        # return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
        out = nn_forward_fn(params, state)
        # out = model.apply(params, state)
        # out = apply_fun(model, params, state)
        if energies:
            return out[0], out[1]
        return out[0] - out[1]

    return lagrangian


def analytic_energies(state):
    q = state[0:2]
    q_dot = state[2:4]
    return lagrangian(q, q_dot, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8, energies=True)


def learned_energies(params, state):
    q = state[0:2]
    q_dot = state[2:4]
    L_fun = learned_lagrangian(params, energies=True)
    return L_fun(q, q_dot)


def recon_kin(lagrangian, state):
    q, q_t = jnp.split(state, 2)
    In = -jax.hessian(lagrangian, 1)(q, q_t)
    T = 1 / 2 * jnp.transpose(q_t) @ In @ q_t
    return T


# define the loss of the model (MSE between predicted q, \dot q and targets)
@jax.jit
def loss(params, batch, H_0=0, time_step=None):
    state, targets = batch
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(state)
    V, T = jax.vmap(partial(learned_energies, params))(state)
    H = T + V
    T_recon = jax.vmap(partial(recon_kin, learned_lagrangian(params)))(state)
    return jnp.mean((preds - targets) ** 2) + 1000 * jnp.mean((H - H_0) ** 2) + jnp.mean((T - T_recon) ** 2)


def make_mse_loss(xs, ys):
    def mse_loss(params):
        """Gives the value of the loss on the (xs, ys) dataset for the given model (params)."""

        # Batched version via vmap
        return loss(params, (xs, ys), H_0=0, time_step=None)

    return jax.jit(mse_loss)  # and finally we jit the result (mse_loss is a pure function)


# @jax.jit
# def update_timestep(i, opt_state, batch, get_params, opt_update):
#     params = get_params(opt_state)
#     return opt_update(i, jax.grad(loss)(params, batch, time_step), opt_state)
#
#
# @jax.jit
# def update_derivative(i, opt_state, batch, get_params, opt_update, H_0):
#     params = get_params(opt_state)
#     return opt_update(i, jax.grad(loss)(params, batch, H_0, None), opt_state)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256)(x)
        x = nn.activation.softplus(x)
        x = nn.Dense(features=128)(x)
        x = nn.activation.softplus(x)
        x = nn.Dense(features=64)(x)
        x = nn.activation.softplus(x)
        x = nn.Dense(features=2)(x)
        return x


def generate_trainig_test_date(time_step, N, x_0, x_0_test):
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


def plot_data(train_vis, test_vis):
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


def train_plot(batch_size, num_batches, test_every):
    train_losses = []
    test_losses = []

    # stax nn
    rng = jax.random.PRNGKey(0)
    _, init_params = init_random_params(rng, (-1, 4))
    opt_init, opt_update, get_params = optimizers.adam(
        lambda t: jnp.select([t < batch_size * (num_batches // 4),
                              t < batch_size * (2 * num_batches // 4),
                              t < batch_size * (3 * num_batches // 4),
                              t > batch_size * (3 * num_batches // 4)],
                             [1e-3, 3e-4, 1e-4, 3e-5]))
    opt_state = opt_init(init_params)
    # print(jax.tree_map(lambda x: x.shape, init_params))

    # flax nn
    # seed = 23
    # key1, key2 = random.split(random.PRNGKey(seed))
    # x = random.normal(key1, (4,))  # create a dummy input, a 4-dimensional random vector
    # init_flax_params = model.init(key2, x)
    # # print(jax.tree_map(lambda x: x.shape, init_flax_params))
    # opt_adam = optax.adam(learning_rate=1e-3)
    # opt_state = opt_adam.init(init_flax_params)

    @jax.jit
    def update_derivative(i, opt_state, batch, H_0):
        params = get_params(opt_state)
        return opt_update(i, jax.grad(loss)(params, batch, H_0, None), opt_state)

    for iteration in range(batch_size * num_batches + 1):
        # loss_value, grads = vg_loss(init_flax_params)
        # updates, opt_state = opt_adam.update(grads, opt_state)
        # init_flax_params = optax.apply_updates(init_flax_params, updates)
        # params = init_flax_params

        if iteration % batch_size == 0:
            params = get_params(opt_state)
            train_loss = loss(params, (x_train, xt_train), H_0=0.0)
            train_losses.append(train_loss)
            test_loss = loss(params, (x_test, xt_test), H_0=0.0)
            test_losses.append(test_loss)
            if iteration % (batch_size * test_every) == 0:
                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

        opt_state = update_derivative(iteration, opt_state, (x_train, xt_train), 0)


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

    return get_params(opt_state)


if __name__ == "__main__":
    # Generate training data
    time_step = 0.001
    N = 1000 * 100
    x_0 = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)
    noise = np.random.RandomState(0).randn(x_0.size)
    x_0_test = x_0 + 1e-3 * noise
    x_train, xt_train, x_test, xt_test, y_train, y_test = generate_trainig_test_date(time_step, N, x_0, x_0_test)

    # Standarize
    train_vis = jax.vmap(normalize_dp)(x_train)
    test_vis = jax.vmap(normalize_dp)(x_test)

    # Show training and test data
    plot_data(train_vis, test_vis)

    # Calculate the starting energies
    # T_0, V_0 = analytic_energies(x_0)
    # H_train = T_0 + V_0
    # T_0, V_0 = analytic_energies(x_0_test)
    # H_test = T_0 + V_0

    x_train = jax.device_put(jax.vmap(normalize_dp)(x_train))
    x_test = jax.device_put(jax.vmap(normalize_dp)(x_test))
    # y_train = jax.device_put(y_train)
    # y_test = jax.device_put(y_test)

    # Train
    # Build the loss
    mse_loss = make_mse_loss(x_train, xt_train)
    vg_loss = jax.jit(jax.value_and_grad(mse_loss))

    # build a neural network model
    init_random_params, nn_forward_fn = stax.serial(
        stax.Dense(256), stax.Softplus,
        stax.Dense(256), stax.Softplus,
        stax.Dense(128), stax.Softplus,
        stax.Dense(64), stax.Softplus,
        stax.Dense(2),
    )

    model = MLP()

    # numbers in comments denote stephan's settings
    batch_size = 100
    test_every = 10
    num_batches = 150 * 5

    params = train_plot(batch_size, num_batches, test_every)

    # Test system
    N_sim = 1000 * 5
    x_0_sim = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)

    # Simulate system
    t_sim = np.arange(N_sim, dtype=np.float32) * time_step  # time steps 0 to N
    x_sim = solve_analytical(x_0_sim, t_sim)
    x_sim = jax.vmap(normalize_dp)(x_sim)

    # Analytic energies from trajectory
    T_ana, V_ana = jax.device_get(jax.vmap(analytic_energies)(x_sim))
    H_ana = T_ana + V_ana
    L_ana = T_ana - V_ana

    # Learned energies from trajectory
    V_lnn, T_lnn = jax.device_get(jax.vmap(partial(learned_energies, params))(x_sim))
    H_lnn = T_lnn + V_lnn
    L_lnn = T_lnn - V_lnn

    # Reconstructed energies from trajectory

    # Calibration
    x_origin = np.array([0, 0, 0, 0], dtype=np.float32)
    _, V_origin = jax.device_get(analytic_energies(x_origin))
    V_lnn_origin, _ = jax.device_get(partial(learned_energies, params)(x_origin))
    x_bottom = np.array([np.pi / 2, np.pi / 2, 0, 0], dtype=np.float32)
    _, V_bottom = jax.device_get(analytic_energies(x_origin))
    V_lnn_bottom, _ = jax.device_get(partial(learned_energies, params)(x_bottom))
    alpha = (V_origin - V_bottom) / (V_lnn_origin - V_lnn_bottom)
    beta = V_origin - alpha*V_lnn_origin
    H_rec_0 = jnp.mean(alpha*(T_lnn + V_lnn) + beta)

    T_rec = alpha * jax.device_get(jax.vmap(partial(recon_kin, learned_lagrangian(params)))(x_sim)) + beta
    V_rec = H_rec_0 - T_rec
    H_rec = V_rec + T_rec
    L_rec = T_rec - V_rec

    # # Calibrated energies from trajectory
    # mean_ana = jnp.mean(T_ana)
    # mean_lnn = jnp.mean(T_rec)
    # max_ana = jnp.max(T_ana - mean_ana)
    # max_lnn = jnp.max(T_rec - mean_lnn)
    # T_cal = ((T_rec - mean_lnn) / max_lnn) * max_ana + mean_ana
    # # T_cal = T_rec * (max_ana / mean_lnn) - (mean_lnn * max_ana / max_lnn - mean_ana)
    #
    # mean_ana = jnp.mean(V_ana)
    # mean_lnn = jnp.mean(V_rec)
    # max_ana = jnp.max(V_ana - mean_ana)
    # max_lnn = jnp.max(V_rec - mean_lnn)
    # V_cal = ((V_rec - mean_lnn) / max_lnn) * max_ana + mean_ana
    # # V_cal = V_rec * (max_ana / mean_lnn) - (mean_lnn * max_ana / max_lnn - mean_ana)
    #
    # L_cal = T_cal - V_cal
    # H_cal = T_cal + V_cal

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, H_ana, label='H. Analytic')
    plt.plot(t_sim, H_lnn, label='H. Lnn')
    # plt.plot(t_sim, H_cal, label='H. Calibrated')
    plt.plot(t_sim, H_rec, label='H. Recon')
    plt.title('Hamiltonians')
    plt.ylim(0, 0.1)
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, L_ana, label='L. Analytic')
    plt.plot(t_sim, L_lnn, label='L. Lnn')
    plt.plot(t_sim, L_rec, label='L. Reconstructed')
    # plt.plot(t_sim, L_cal, label='L. Calibrated')
    plt.title('Lagrangians')
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()

    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(t_sim, T_ana, label='Kin. Analytic')
    plt.plot(t_sim, V_ana, label='Pot. Analytic')
    plt.plot(t_sim, T_lnn, label='Kin. Lnn.')
    plt.plot(t_sim, V_lnn, label='Pot. Lnn.')
    plt.plot(t_sim, T_rec, label='Kin. Reconstructed')
    plt.plot(t_sim, V_rec, label='Pot. Reconstructed')
    # plt.plot(t_sim, T_cal, label='Kin. Calibrated')
    # plt.plot(t_sim, V_cal, label='Pot. Calibrated')
    plt.title('Energies')
    plt.ylabel('Energy Level (J)')
    plt.xlabel('Time (s)')
    plt.legend(loc="best")
    plt.show()
