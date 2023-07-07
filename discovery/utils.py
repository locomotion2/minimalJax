import numpy as np


def gaus(value: float, width: float = 0.3):
    return float(np.exp(-(value / width) ** 2))


def debug_print(text: str = None, obj=None):
    if True:
        print(f'{text}: {obj}')


def hom2xyphi(hom):
    fkin = np.empty((hom.shape[0], 3))
    fkin[:, 0] = hom[:, 0, 2]
    fkin[:, 1] = hom[:, 1, 2]
    fkin[:, 2] = np.angle(np.exp(1j * (np.arctan2(hom[:, 1, 0], hom[:, 0, 0]))))
    return fkin


def project(u: np.ndarray, v: np.ndarray):
    v_norm = np.linalg.norm(v)
    return (u @ v / v_norm ** 2) * v


def inverse_grad_desc(var_des, error_func, jacobian_func, name: str = 'variable',
                      num_dims: int = 2, q0=None, K=1.0, tol=1e-3, max_steps=100,
                      max_tries=5):
    # Handle starting point
    if q0 is not None:
        q = q0
    else:
        q = np.random.uniform(-np.pi, np.pi, num_dims)

    # Shorten the functions for practicality
    f = lambda x: error_func(x)
    A = lambda x: np.linalg.pinv(jacobian_func(x))

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

            q = np.random.uniform(-np.pi, np.pi, num_dims)
            print(f"No {name} solution found for {var_des}, try # {counter}")

        # Check if solution is good enough
        e = var_des - f(q)
        if np.linalg.norm(e) < tol:
            break

        # If not, improve solution
        try:
            inc = A(q) @ np.squeeze(K * e)
        except ValueError:
            inc = (A(q) * np.squeeze(K * e))[:, 0]
        q += inc
        step += 1

    return q
