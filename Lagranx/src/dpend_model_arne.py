import jax.numpy as jnp
import numpy as np


def lagrangian(q, q_dot, m1, m2, l1, l2, g, energies=False):

    def potential_energy(q, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8):
        def _potential(q):
            l = jnp.array([l1, l2])
            m = jnp.array([m1, m2])
            k = jnp.array([0, 0])
            qr = (0, 0)

            return g * l[0] * m[0] * jnp.sin(q[0]) + g * m[1] *\
                (l[0] * jnp.sin(q[0]) + l[1] * jnp.sin(q[0] + q[1])) + 0.5 * k[0] * (-q[0] + qr[0]) ** 2 + 0.5 * \
                k[1] * (-q[1] + qr[1]) ** 2

        return _potential(q.T) - _potential(np.array([-jnp.pi / 2, 0]))

    def kinetic_energy(q, dq, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8):
        l = jnp.array([l1, l2])
        m = jnp.array([m1, m2])

        return 0.5 * dq[0] ** 2 * l[0] ** 2 * m[0] + 0.5 * m[1] * \
            (dq[0] ** 2 * l[0] ** 2 + 2 * dq[0] ** 2 * l[0] * l[1] * jnp.cos(q[1]) + dq[0] ** 2 *
             l[1] ** 2 + 2 * dq[0] * dq[1] * l[0] * l[1] * jnp.cos(q[1]) + 2 * dq[0] * dq[1] *
             l[1] ** 2 + dq[1] ** 2 * l[1] ** 2)

    # Calculate energies
    V = potential_energy(q, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    T = kinetic_energy(q, q_dot, m1=m1, m2=m2, l1=l1, l2=l2, g=g)

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

    ddq_cons = jnp.array([dq[0], dq[1], 1.0 * (
            0.5 * dq[0] ** 2 * l[0] ** 2 * l[1] * m[1] * jnp.sin(2 * q[1]) + 1.0 * dq[0] ** 2 * l[0] * l[
        1] ** 2 * m[1] * jnp.sin(q[1]) + 2.0 * dq[0] * dq[1] * l[0] * l[1] ** 2 * m[1] * jnp.sin(
        q[1]) + 1.0 * dq[1] ** 2 * l[0] * l[1] ** 2 * m[1] * jnp.sin(q[1]) - 1.0 * g * l[0] * l[1] * m[
                0] * jnp.cos(q[0]) - 0.5 * g * l[0] * l[1] * m[1] * jnp.cos(q[0]) + 0.5 * g * l[0] * l[1] *
            m[1] * jnp.cos(q[0] + 2 * q[1]) - 1.0 * k[0] * l[1] * q[0] + 1.0 * k[0] * l[1] * qr[0] + 1.0 * k[
                1] * l[0] * q[1] * jnp.cos(q[1]) - 1.0 * k[1] * l[0] * qr[1] * jnp.cos(q[1]) + 1.0 * k[1] *
            l[1] * q[1] - 1.0 * k[1] * l[1] * qr[1] - 1.0 * l[0] * tau_in[1] * jnp.cos(q[1]) + 1.0 * l[1] *
            tau_in[0] - 1.0 * l[1] * tau_in[1]) / (l[0] ** 2 * l[1] * (m[0] + m[1] * jnp.sin(q[1]) ** 2)),
                              2.0 * (-1.0 * dq[0] ** 2 * l[0] ** 3 * l[1] * m[0] * m[1] * jnp.sin(q[1]) - 1.0 * dq[
                                  0] ** 2 * l[0] ** 3 * l[1] * m[1] ** 2 * jnp.sin(q[1]) - 1.0 * dq[0] ** 2 * l[
                                         0] ** 2 * l[1] ** 2 * m[1] ** 2 * jnp.sin(2 * q[1]) - 1.0 * dq[0] ** 2 * l[
                                         0] * l[1] ** 3 * m[1] ** 2 * jnp.sin(q[1]) - 1.0 * dq[0] * dq[1] * l[
                                         0] ** 2 * l[1] ** 2 * m[1] ** 2 * jnp.sin(2 * q[1]) - 2.0 * dq[0] * dq[1] *
                                     l[0] * l[1] ** 3 * m[1] ** 2 * jnp.sin(q[1]) - 0.5 * dq[1] ** 2 * l[0] ** 2 *
                                     l[1] ** 2 * m[1] ** 2 * jnp.sin(2 * q[1]) - 1.0 * dq[1] ** 2 * l[0] * l[
                                         1] ** 3 * m[1] ** 2 * jnp.sin(q[1]) + 0.5 * g * l[0] ** 2 * l[1] * m[0] *
                                     m[1] * jnp.cos(q[0] - q[1]) - 0.5 * g * l[0] ** 2 * l[1] * m[0] * m[
                                         1] * jnp.cos(q[0] + q[1]) + 0.5 * g * l[0] ** 2 * l[1] * m[
                                         1] ** 2 * jnp.cos(q[0] - q[1]) - 0.5 * g * l[0] ** 2 * l[1] * m[
                                         1] ** 2 * jnp.cos(q[0] + q[1]) + 1.0 * g * l[0] * l[1] ** 2 * m[0] * m[
                                         1] * jnp.cos(q[0]) + 0.5 * g * l[0] * l[1] ** 2 * m[1] ** 2 * jnp.cos(
                                          q[0]) - 0.5 * g * l[0] * l[1] ** 2 * m[1] ** 2 * jnp.cos(
                                          q[0] + 2 * q[1]) + 1.0 * k[0] * l[0] * l[1] * m[1] * q[0] * jnp.cos(
                                          q[1]) - 1.0 * k[0] * l[0] * l[1] * m[1] * qr[0] * jnp.cos(q[1]) + 1.0 * k[
                                         0] * l[1] ** 2 * m[1] * q[0] - 1.0 * k[0] * l[1] ** 2 * m[1] * qr[0] - 1.0 *
                                     k[1] * l[0] ** 2 * m[0] * q[1] + 1.0 * k[1] * l[0] ** 2 * m[0] * qr[1] - 1.0 * k[
                                         1] * l[0] ** 2 * m[1] * q[1] + 1.0 * k[1] * l[0] ** 2 * m[1] * qr[1] - 2.0 *
                                     k[1] * l[0] * l[1] * m[1] * q[1] * jnp.cos(q[1]) + 2.0 * k[1] * l[0] * l[1] *
                                     m[1] * qr[1] * jnp.cos(q[1]) - 1.0 * k[1] * l[1] ** 2 * m[1] * q[1] + 1.0 * k[
                                         1] * l[1] ** 2 * m[1] * qr[1] + 1.0 * l[0] ** 2 * m[0] * tau_in[1] + 1.0 * l[
                                         0] ** 2 * m[1] * tau_in[1] - 1.0 * l[0] * l[1] * m[1] * tau_in[
                                         0] * jnp.cos(q[1]) + 2.0 * l[0] * l[1] * m[1] * tau_in[1] * jnp.cos(
                                          q[1]) - 1.0 * l[1] ** 2 * m[1] * tau_in[0] + 1.0 * l[1] ** 2 * m[1] *
                                     tau_in[1]) / (l[0] ** 2 * l[1] ** 2 * m[1] * (
                                      2 * m[0] - m[1] * jnp.cos(2 * q[1]) + m[1]))])
    ddq = ddq_cons - kf * jnp.asarray([0, 0, dq[0], dq[1]])
    return ddq

def normalize(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])


def analytic_energies(state):
    q = state[0:2]
    q_dot = state[2:]
    return lagrangian(q, q_dot, m1=0.05, m2=0.05, l1=0.5, l2=0.5, g=9.8, energies=True)
