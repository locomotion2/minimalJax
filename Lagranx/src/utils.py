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
