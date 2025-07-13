import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def gaus(value: float, width: float = 0.3):
    return float(jnp.exp(-(value / width) ** 2))

def debug_print(text: str = None, obj=None):
    if True:
        print(f'{text}: {obj}')

@jax.jit
def hom2xyphi(hom):
    fkin = jnp.empty((hom.shape[0], 3))
    fkin = fkin.at[:, 0].set(hom[:, 0, 2])
    fkin = fkin.at[:, 1].set(hom[:, 1, 2])
    fkin = fkin.at[:, 2].set(jnp.angle(jnp.exp(1j * (jnp.arctan2(hom[:, 1, 0], hom[:, 0, 0])))))
    return fkin

@jax.jit
def project(key, u: jnp.ndarray, v: jnp.ndarray):
    v_norm = jnp.linalg.norm(v)
    return (u @ v / v_norm ** 2) * v

# This is the inner, JIT-compiled function that contains the core logic.
@partial(jax.jit, static_argnames=['error_func', 'jacobian_func', 'name', 'num_dims', 'K', 'tol', 'max_steps', 'max_tries'])
def _jitted_inverse_grad_desc(key, var_des, q0, error_func, jacobian_func, name: str,
                              num_dims: int, K: float, tol: float, max_steps: int,
                              max_tries: int):
    f = error_func
    init_state = (q0, 0, 1, key)

    def loop_cond_fun(state):
        q, _, counter, _ = state
        e = var_des - f(q)
        return (jnp.linalg.norm(e) >= tol) & (counter <= max_tries)

    def loop_body_fun(state):
        q, step, counter, key = state
        def retry_fn(state_inner):
            _, _, counter_inner, key_inner = state_inner
            key_inner, subkey = jax.random.split(key_inner)
            q_new = jax.random.uniform(subkey, shape=(num_dims,), minval=-jnp.pi, maxval=jnp.pi)
            return q_new, 0, counter_inner + 1, key_inner
        def step_fn(state_inner):
            q_inner, step_inner, counter_inner, key_inner = state_inner
            e = var_des - f(q_inner)
            J = jacobian_func(q_inner)
            def grad_case(grad):
                norm_grad = grad / (jnp.linalg.norm(grad) + 1e-8)
                return norm_grad * jnp.squeeze(K * e)
            def jac_case(jac):
                return jnp.linalg.pinv(jac) @ jnp.squeeze(K * e)
            inc = jax.lax.cond(J.ndim == 1, grad_case, jac_case, J)
            q_new = q_inner + inc
            return q_new, step_inner + 1, counter_inner, key_inner
        return jax.lax.cond(step > max_steps, retry_fn, step_fn, state)

    final_q, _, _, _ = jax.lax.while_loop(loop_cond_fun, loop_body_fun, init_state)
    return final_q

# This is the outer, user-facing function that handles Python-level logic.
def inverse_grad_desc(key, var_des, error_func, jacobian_func, name: str = 'variable',
                      num_dims: int = 2, q0=None, K: float = 1.0, tol: float = 1e-3, max_steps: int = 100,
                      max_tries: int = 5):
    if q0 is None:
        key, subkey = jax.random.split(key)
        q0 = jax.random.uniform(subkey, shape=(num_dims,), minval=-jnp.pi, maxval=jnp.pi)
    
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
    if target_value is None:
        raise ValueError("target_value (E) must not be None")

    init_state = (q0, 0, 0, key)

    def loop_cond(state):
        q, _, try_count, _ = state
        error = jnp.abs(value_func(q) - target_value)
        return (error > tol) & (try_count < max_tries)

    def loop_body(state):
        q, step, try_count, key = state

        def do_step(operand):
            q_in, step_in, try_in, key_in = operand
            grad = grad_func(q_in)
            norm_grad = grad / (jnp.linalg.norm(grad) + 1e-8)
            direction = jnp.sign(target_value - value_func(q_in))
            q_new = q_in + direction * norm_grad * learning_rate
            return q_new, step_in + 1, try_in, key_in

        def do_retry(operand):
            _, _, try_in, key_in = operand
            key_in, subkey = jax.random.split(key_in)
            q_new = jax.random.uniform(subkey, shape=q0.shape, minval=-jnp.pi, maxval=jnp.pi)
            return q_new, 0, try_in + 1, key_in

        return jax.lax.cond(step >= max_steps, do_retry, do_step, state)

    final_q, _, _, _ = jax.lax.while_loop(loop_cond, loop_body, init_state)
    return final_q