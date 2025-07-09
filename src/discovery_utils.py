import numpy as np # Keep for general numpy compatibility if needed elsewhere
import jax
import jax.numpy as jnp


def gaus(value: float, width: float = 0.3):
    return float(jnp.exp(-(value / width) ** 2)) # Use jnp.exp


def debug_print(text: str = None, obj=None):
    if True:
        print(f'{text}: {obj}')


@jax.jit
def hom2xyphi(hom):
    fkin = jnp.empty((hom.shape[0], 3)) # Use jnp.empty
    fkin = fkin.at[:, 0].set(hom[:, 0, 2])
    fkin = fkin.at[:, 1].set(hom[:, 1, 2])
    fkin = fkin.at[:, 2].set(jnp.angle(jnp.exp(1j * (jnp.arctan2(hom[:, 1, 0], hom[:, 0, 0]))))) # Use jnp.angle, jnp.exp, jnp.arctan2
    return fkin


@jax.jit
def project(key, u: jnp.ndarray, v: jnp.ndarray): # Added key parameter, type hint changed to jnp.ndarray
    v_norm = jnp.linalg.norm(v) # Use jnp.linalg.norm
    return (u @ v / v_norm ** 2) * v


@jax.jit
def inverse_grad_desc(key, var_des, error_func, jacobian_func, name: str = 'variable', # Added key parameter
                      num_dims: int = 2, q0=None, K=1.0, tol=1e-3, max_steps=100,
                      max_tries=5):
    # Handle starting point
    if q0 is not None:
        q = q0
    else:
        key, subkey = jax.random.split(key) # Split key for randomness
        q = jax.random.uniform(subkey, shape=(num_dims,), minval=-jnp.pi, maxval=jnp.pi) # Use jax.random.uniform, jnp.pi

    # Shorten the functions for practicality
    f = lambda x: error_func(x)
    A = lambda x: jnp.linalg.pinv(jacobian_func(x)) # Use jnp.linalg.pinv

    # Iterate until solution is found
    step = 0
    counter = 1
    while True:
        # Did not find a solution this try? Start from another seed and try
        if step > max_steps:
            step = 0
            counter += 1
            if counter > max_tries:
                print(
                    f"No {name} solution found for {var_des} in final try, problem is "
                    f"hard.")
                return q

            key, subkey = jax.random.split(key) # Split key for randomness
            q = jax.random.uniform(subkey, shape=(num_dims,), minval=-jnp.pi, maxval=jnp.pi) # Use jax.random.uniform, jnp.pi
            print(f"No {name} solution found for {var_des}, try # {counter}")

        # Check if solution is good enough
        e = var_des - f(q)
        if jnp.linalg.norm(e) < tol: # Use jnp.linalg.norm
            break

        # If not, improve solution
        try:
            inc = A(q) @ jnp.squeeze(K * e) # Use jnp.squeeze
        except ValueError:
            inc = (A(q) * jnp.squeeze(K * e))[:, 0] # Use jnp.squeeze
        q += inc
        step += 1

    return q