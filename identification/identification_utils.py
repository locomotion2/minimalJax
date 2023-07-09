import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


@jax.jit
def wrap_angle(q):
    return (q + np.pi) % (2 * np.pi) - np.pi


@jax.jit
def calibrate(ref, val):
    # Calculate means
    mean_ref = jnp.mean(ref)
    mean_val = jnp.mean(val)

    # Calculate signal height
    height_ref = (jnp.max(ref - mean_ref) -
                  jnp.min(ref - mean_ref)) / 2
    height_val = (jnp.max(val - mean_val) -
                  jnp.min(val - mean_val)) / 2

    # Calculate coefficients for linear correction
    alpha = height_ref / height_val
    beta = - mean_val * alpha + mean_ref
    val_calibrated = val * alpha + beta
    coeffs = [alpha, beta]

    return coeffs, val_calibrated

@jax.jit
def update_format_buffers(variable_buffer, variable):
    # var_list_save = []
    # var_list_send = []
    # for index, element in enumerate(variable):
    # update buffer
    vector = variable_buffer[:, 0:-1]
    vector = jnp.insert(vector, 0, variable, axis=1)
    var_list_save = vector

    # filter
    # vector = np.array(vector)
    # vector_filtered = filter_svg(vector)

    # format buffer to be sent
    var_list_send = vector
    # var_list_send = vector[:, ::10]
    # var_list_send.append(vector)

    return jnp.array(var_list_save), jnp.concatenate(var_list_send)


@jax.jit
def format_state(q, q_buff, dq, dq_buff, tau):
    # format into the right sizes
    q = wrap_angle(q)

    # update buffers, subsample and flatten
    q_buff, q_out = update_format_buffers(q_buff, q)
    dq_buff, dq_out = update_format_buffers(dq_buff, dq)

    # prepare package
    state = jnp.concatenate([q_out, dq_out, tau])

    return state, q_buff, dq_buff

@jax.jit
def format_state_sim(q, q_buff, dq, dq_buff):
    # update q
    q_buff_old = q_buff[:, 0:-1]
    q_buff_new = jnp.insert(q_buff_old, 0, q, axis=1)

    # update dq
    dq_buff_old = dq_buff[:, 0:-1]
    dq_buff_new = jnp.insert(dq_buff_old, 0, dq, axis=1)

    return (jnp.concatenate(q_buff_old),
            jnp.concatenate(dq_buff_old)),\
        (q_buff_new, dq_buff_new)
