import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# # tau * dy2/dt2 + 2*zeta*tau*dy/dt + y = Kp*u
# Kp = 2.0  # gain
# tau = 1.0  # time constant
# zeta = 0.25  # damping factor
# theta = 0.0  # no time delay
# du = 1.0  # change in u
#
# # (1) Transfer Function
# num = [Kp]
# den = [tau ** 2, 2 * zeta * tau, 1]
# sys1 = signal.TransferFunction(num, den)
# t1, y1 = signal.step(sys1)
#
# # (2) State Space
# A = [[0.0, 1.0], [-1.0 / tau ** 2, -2.0 * zeta / tau]]
# B = [[0.0], [Kp / tau ** 2]]
# C = [1.0, 0.0]
# D = 0.0
# sys2 = signal.StateSpace(A, B, C, D)
# t2, y2 = signal.step(sys2)


# (3) ODE Integrator
# def model3(x, t):
#     y = x[0]
#     dydt = x[1]
#     dy2dt2 = (-2.0 * zeta * tau * dydt - y + Kp * du) / tau ** 2
#     return [dydt, dy2dt2]


def hopf(x, t, args=[1, 1]):
    mu = args[0]
    omega = args[1]
    rho = x[0] ** 2 + x[1] ** 2

    circleDist = mu ** 2 - rho
    dx1 = -x[1] * omega + x[0] * circleDist
    dx2 = x[0] * omega + x[1] * circleDist

    return [dx1, dx2]


t3 = np.linspace(0, 20, 100)
x_traj = odeint(hopf, [2, 2], t3)
x1 = x_traj[:, 0]
x2 = x_traj[:, 1]

plt.figure(1)
# plt.plot(t1, y1 * du, 'b--', linewidth=3, label='Transfer Fcn')
# plt.plot(t2, y2 * du, 'g:', linewidth=2, label='State Space')
plt.plot(x1, x2, 'r-', linewidth=1, label='ODE Integrator')
# y_ss = Kp * du
# plt.plot([0, max(t1)], [y_ss, y_ss], 'k:')
plt.xlim([min(x1), max(x1)])
plt.ylim([min(x2), max(x2)])
plt.xlabel('Time')
plt.ylabel('Response (y)')
plt.legend(loc='best')
# plt.savefig('2nd_order.png')
plt.show()
