import jax.numpy as jnp
import numpy as np

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
    q1, q2, w1, w2 = state
    l = jnp.array([l1, l2])
    m = jnp.array([m1, m2])
    k = jnp.array([0, 0])
    tau_in = jnp.array([0, 0])
    qr = (0, 0)
    kf = 0.0

    q = jnp.array([q1, q2])
    dq = jnp.array([w1, w2])

    ddq_cons = jnp.array([[dq[0]],[dq[1]],[1.0 * (
            0.5 * dq[0] ** 2 * l[0] ** 2 * l[1] * m[1] * np.sin(2 * q[1]) + 1.0 * dq[0] ** 2 * l[0] * l[
        1] ** 2 * m[1] * np.sin(q[1]) + 2.0 * dq[0] * dq[1] * l[0] * l[1] ** 2 * m[1] * np.sin(
        q[1]) + 1.0 * dq[1] ** 2 * l[0] * l[1] ** 2 * m[1] * np.sin(q[1]) - 1.0 * g * l[0] * l[1] * m[
                0] * np.cos(q[0]) - 0.5 * g * l[0] * l[1] * m[1] * np.cos(q[0]) + 0.5 * g * l[0] * l[1] *
            m[1] * np.cos(q[0] + 2 * q[1]) - 1.0 * k[0] * l[1] * q[0] + 1.0 * k[0] * l[1] * qr[0] + 1.0 * k[
                1] * l[0] * q[1] * np.cos(q[1]) - 1.0 * k[1] * l[0] * qr[1] * np.cos(q[1]) + 1.0 * k[1] *
            l[1] * q[1] - 1.0 * k[1] * l[1] * qr[1] - 1.0 * l[0] * tau_in[1] * np.cos(q[1]) + 1.0 * l[1] *
            tau_in[0] - 1.0 * l[1] * tau_in[1]) / (l[0] ** 2 * l[1] * (m[0] + m[1] * np.sin(q[1]) ** 2))], [
                             2.0 * (-1.0 * dq[0] ** 2 * l[0] ** 3 * l[1] * m[0] * m[1] * np.sin(q[1]) - 1.0 * dq[
                                 0] ** 2 * l[0] ** 3 * l[1] * m[1] ** 2 * np.sin(q[1]) - 1.0 * dq[0] ** 2 * l[
                                        0] ** 2 * l[1] ** 2 * m[1] ** 2 * np.sin(2 * q[1]) - 1.0 * dq[0] ** 2 * l[
                                        0] * l[1] ** 3 * m[1] ** 2 * np.sin(q[1]) - 1.0 * dq[0] * dq[1] * l[
                                        0] ** 2 * l[1] ** 2 * m[1] ** 2 * np.sin(2 * q[1]) - 2.0 * dq[0] * dq[1] *
                                    l[0] * l[1] ** 3 * m[1] ** 2 * np.sin(q[1]) - 0.5 * dq[1] ** 2 * l[0] ** 2 *
                                    l[1] ** 2 * m[1] ** 2 * np.sin(2 * q[1]) - 1.0 * dq[1] ** 2 * l[0] * l[
                                        1] ** 3 * m[1] ** 2 * np.sin(q[1]) + 0.5 * g * l[0] ** 2 * l[1] * m[0] *
                                    m[1] * np.cos(q[0] - q[1]) - 0.5 * g * l[0] ** 2 * l[1] * m[0] * m[
                                        1] * np.cos(q[0] + q[1]) + 0.5 * g * l[0] ** 2 * l[1] * m[
                                        1] ** 2 * np.cos(q[0] - q[1]) - 0.5 * g * l[0] ** 2 * l[1] * m[
                                        1] ** 2 * np.cos(q[0] + q[1]) + 1.0 * g * l[0] * l[1] ** 2 * m[0] * m[
                                        1] * np.cos(q[0]) + 0.5 * g * l[0] * l[1] ** 2 * m[1] ** 2 * np.cos(
                                         q[0]) - 0.5 * g * l[0] * l[1] ** 2 * m[1] ** 2 * np.cos(
                                         q[0] + 2 * q[1]) + 1.0 * k[0] * l[0] * l[1] * m[1] * q[0] * np.cos(
                                         q[1]) - 1.0 * k[0] * l[0] * l[1] * m[1] * qr[0] * np.cos(q[1]) + 1.0 * k[
                                        0] * l[1] ** 2 * m[1] * q[0] - 1.0 * k[0] * l[1] ** 2 * m[1] * qr[0] - 1.0 *
                                    k[1] * l[0] ** 2 * m[0] * q[1] + 1.0 * k[1] * l[0] ** 2 * m[0] * qr[1] - 1.0 * k[
                                        1] * l[0] ** 2 * m[1] * q[1] + 1.0 * k[1] * l[0] ** 2 * m[1] * qr[1] - 2.0 *
                                    k[1] * l[0] * l[1] * m[1] * q[1] * np.cos(q[1]) + 2.0 * k[1] * l[0] * l[1] *
                                    m[1] * qr[1] * np.cos(q[1]) - 1.0 * k[1] * l[1] ** 2 * m[1] * q[1] + 1.0 * k[
                                        1] * l[1] ** 2 * m[1] * qr[1] + 1.0 * l[0] ** 2 * m[0] * tau_in[1] + 1.0 * l[
                                        0] ** 2 * m[1] * tau_in[1] - 1.0 * l[0] * l[1] * m[1] * tau_in[
                                        0] * np.cos(q[1]) + 2.0 * l[0] * l[1] * m[1] * tau_in[1] * np.cos(
                                         q[1]) - 1.0 * l[1] ** 2 * m[1] * tau_in[0] + 1.0 * l[1] ** 2 * m[1] *
                                    tau_in[1]) / (l[0] ** 2 * l[1] ** 2 * m[1] * (
                                     2 * m[0] - m[1] * np.cos(2 * q[1]) + m[1]))]])
    ddq = ddq_cons - kf * np.asarray([[0], [0], dq[0], [dq[1]]])
    return ddq

    # a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
    # a2 = (l1 / l2) * jnp.cos(t1 - t2)
    # f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * jnp.sin(t1 - t2) - \
    #      (g / l1) * jnp.sin(t1)
    # f2 = (l1 / l2) * (w1 ** 2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
    # g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    # g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    # return jnp.stack([w1, w2, g1, g2])

def normalize(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])

def analytic_energies(state):
    q = state[0:2]
    q_dot = state[2:]
    return lagrangian(q, q_dot, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8, energies=True)