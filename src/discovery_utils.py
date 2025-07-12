import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def gaus(value: float, width: float = 0.3):
    """A JAX-compatible Gaussian function."""
    return float(jnp.exp(-(value / width) ** 2))

def debug_print(text: str = None, obj=None):
    """A simple debug print function (not for use inside JIT)."""
    if True:
        print(f'{text}: {obj}')

@jax.jit
def hom2xyphi(hom: jax.Array) -> jax.Array:
    """
    Converts a batch of homogeneous transformation matrices to [x, y, phi] coordinates.
    This function is JIT-compiled for performance.
    """
    # Pre-allocate the output array
    fkin = jnp.empty((hom.shape[0], 3))
    # Efficiently update slices of the array
    fkin = fkin.at[:, 0].set(hom[:, 0, 2])
    fkin = fkin.at[:, 1].set(hom[:, 1, 2])
    # Use jnp.angle on a complex representation for a stable angle calculation
    fkin = fkin.at[:, 2].set(jnp.angle(jnp.exp(1j * (jnp.arctan2(hom[:, 1, 0], hom[:, 0, 0])))))
    return fkin

@jax.jit
def project(u: jax.Array, v: jax.Array) -> jax.Array:
    """
    Projects vector u onto vector v using a JIT-compatible implementation.
    """
    v_norm_sq = jnp.dot(v, v)
    # Add a small epsilon to prevent division by zero
    return (jnp.dot(u, v) / (v_norm_sq + 1e-8)) * v

# This is the inner, JIT-compiled function that contains the core logic.
@partial(jax.jit, static_argnames=['error_func', 'jacobian_func', 'name', 'num_dims', 'K', 'tol', 'max_steps', 'max_tries'])
def _jitted_inverse_grad_desc(key, var_des, q0, error_func, jacobian_func, name: str,
                              num_dims: int, K: float, tol: float, max_steps: int,
                              max_tries: int):
    """
    The core JIT-compiled logic for inverse kinematics using gradient descent.
    Uses jax.lax.while_loop for efficient iteration on device.
    """
    f = error_func
    # The state tuple (q, step_count, try_count, prng_key) is passed through the loop
    init_state = (q0, 0, 1, key)

    def loop_cond_fun(state):
        """Condition to continue the while loop."""
        q, _, counter, _ = state
        e = var_des - f(q)
        # Continue if error is too high and we haven't exceeded max tries
        return (jnp.linalg.norm(e) >= tol) & (counter <= max_tries)

    def loop_body_fun(state):
        """Body of the while loop, performs one step or retries."""
        q, step, counter, key = state

        def retry_fn(state_inner):
            """Logic to execute when max_steps is reached: reset and retry."""
            _, _, counter_inner, key_inner = state_inner
            key_inner, subkey = jax.random.split(key_inner)
            # Generate a new random guess for q
            q_new = jax.random.uniform(subkey, shape=(num_dims,), minval=-jnp.pi, maxval=jnp.pi)
            return q_new, 0, counter_inner + 1, key_inner

        def step_fn(state_inner):
            """Logic to execute for a normal gradient descent step."""
            q_inner, step_inner, counter_inner, key_inner = state_inner
            e = var_des - f(q_inner)
            J = jacobian_func(q_inner)

            def grad_case(grad):
                """Handle the case where the Jacobian is just a gradient vector."""
                norm_grad = grad / (jnp.linalg.norm(grad) + 1e-8)
                return norm_grad * jnp.squeeze(K * e)

            def jac_case(jac):
                """Handle the standard Jacobian matrix case using pseudo-inverse."""
                return jnp.linalg.pinv(jac) @ jnp.squeeze(K * e)

            # Use lax.cond to choose the update rule based on Jacobian's dimension
            inc = jax.lax.cond(J.ndim == 1, grad_case, jac_case, J)
            q_new = q_inner + inc
            return q_new, step_inner + 1, counter_inner, key_inner

        # If we've taken too many steps, retry. Otherwise, take another step.
        return jax.lax.cond(step > max_steps, retry_fn, step_fn, state)

    # Run the while loop until the condition is false
    final_q, _, _, _ = jax.lax.while_loop(loop_cond_fun, loop_body_fun, init_state)
    return final_q

# This is the outer, user-facing function that handles Python-level logic.
def inverse_grad_desc(key, var_des, error_func, jacobian_func, name: str = 'variable',
                      num_dims: int = 2, q0=None, K: float = 1.0, tol: float = 1e-3, max_steps: int = 100,
                      max_tries: int = 5):
    """
    A user-friendly wrapper for the JIT-compiled inverse kinematics solver.
    Handles default argument values and PRNG key splitting.
    """
    if q0 is None:
        key, subkey = jax.random.split(key)
        q0 = jax.random.uniform(subkey, shape=(num_dims,), minval=-jnp.pi, maxval=jnp.pi)
    
    # Call the fast, JIT-compiled core function
    return _jitted_inverse_grad_desc(key, var_des, q0, error_func, jacobian_func, name,
                                     num_dims, K, tol, max_steps, max_tries)

## --- NEW SPECIALIZED FUNCTION --- ##
@partial(jax.jit, static_argnames=['value_func', 'grad_func', 'max_steps', 'max_tries'])
def find_by_grad_desc(key,
                      target_value,
                      q0,
                      value_func,
                      grad_func,
                      tol: float = 1e-3,
                      max_steps: int = 100,
                      max_tries: int = 10,
                      learning_rate: float = 0.1):
    """
    A JIT-compatible function to find an input `q` that yields a `target_value`
    from a `value_func`, using simple gradient descent.
    """
    init_state = (q0, 0, 0, key)

    def loop_cond(state):
        """Continue if error is above tolerance and we haven't exceeded retries."""
        q, _, try_count, _ = state
        error = jnp.abs(value_func(q) - target_value)
        return (error > tol) & (try_count < max_tries)

    def loop_body(state):
        """One iteration of the search loop."""
        q, step, try_count, key = state

        def do_step(operand):
            """Perform one gradient descent step."""
            q_in, step_in, try_in, key_in = operand
            grad = grad_func(q_in)
            # Normalize gradient to get direction
            norm_grad = grad / (jnp.linalg.norm(grad) + 1e-8)
            # Move in the direction that reduces the error
            direction = jnp.sign(target_value - value_func(q_in))
            q_new = q_in + direction * norm_grad * learning_rate
            return q_new, step_in + 1, try_in, key_in

        def do_retry(operand):
            """Reset `q` to a new random value if stuck."""
            _, _, try_in, key_in = operand
            key_in, subkey = jax.random.split(key_in)
            q_new = jax.random.uniform(subkey, shape=q0.shape, minval=-jnp.pi, maxval=jnp.pi)
            return q_new, 0, try_in + 1, key_in

        # If max_steps is reached, retry; otherwise, take a step.
        return jax.lax.cond(step >= max_steps, do_retry, do_step, state)

    final_q, _, _, _ = jax.lax.while_loop(loop_cond, loop_body, init_state)
    return final_q
