import matplotlib.pyplot as plt


def plot_joint_positions(q, q_sim=None, settings=None):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(q, linewidth=2, label="q")
    if settings["simulate"]:
        plt.plot(q_sim, linewidth=2, label="q_sim")
    plt.legend()
    plt.title("Joint positions.")
    plt.ylabel("rad")
    plt.xlabel("sample (n)")
    plt.show()


def plot_joint_speeds(dq, dq_sim=None, settings=None):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(dq, linewidth=2, label="dq")
    if settings["simulate"]:
        plt.plot(dq_sim, linewidth=2, label="dq_sim")
    plt.legend()
    plt.title("Joint speeds.")
    plt.ylabel("rad/s")
    plt.xlabel("sample (n)")
    plt.show()


def plot_friction_coeffs(tau_loss, dq):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(-tau_loss / dq, linewidth=2, label='k_f')
    plt.legend()
    plt.title('Friction coeffs.')
    plt.ylabel('Something')
    plt.xlabel('sample (n)')
    plt.legend([r'$q_1$', r'$q_2$', r'$\theta_1$', r'$\theta_2$'], loc="best")
    plt.show()


def plot_accelerations(ddq_target, ddq_pred):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(ddq_target[:, 4:8], linewidth=3, label="target")
    plt.plot(ddq_pred[:, 4:8], linewidth=3, linestyle="--", label="pred")
    plt.legend()
    plt.title("Accelerations")
    plt.ylabel("rad/s^2")
    plt.xlabel("sample (n)")
    plt.legend(
        [
            r"$q_1$",
            r"$q_2$",
            r"$\theta_1$",
            r"$\theta_2$",
            r"$q^p_1$",
            r"$q^p_2$",
            r"$\theta^p_1$",
            r"$\theta^p_2$",
        ],
        loc="best",
    )
    plt.show()


def plot_motor_torques(tau_target, tau_pred, tau_loss):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(tau_target[:, 2:4], linewidth=3, label="target")
    plt.plot(tau_pred[:, 2:4], linewidth=3, linestyle="--", label="pred")
    plt.plot(tau_loss[:, 2:4], linewidth=2, linestyle="--", label="loss")
    plt.legend()
    plt.title("Motor torques")
    plt.ylabel("Nm")
    plt.xlabel("sample (n)")
    plt.show()


def plot_losses(L_acc_qdd, L_acc_tau, L_pot):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(L_acc_qdd, linewidth=2, label='ddq')
    plt.plot(10 * L_acc_tau, linewidth=2, label='tau')
    plt.plot(100 * L_pot, linewidth=2, label='pot')
    plt.legend()
    plt.title('Losses')
    plt.show()


def plot_powers(pow_input):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(pow_input, linewidth=2, label='Power input.')
    plt.title('Powers')
    plt.ylabel('Power (J/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()


def plot_hamiltonians(H_ana, H_mec, H_loss, H_cal, H_f):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(H_ana, linewidth=2, label="H. Ana")
    plt.plot(H_mec, linewidth=2, label="H. Mec")
    plt.plot(H_loss, linewidth=2, label="H. Loss")
    plt.plot(H_cal, linewidth=2, label="H. Cal")
    plt.plot(H_f, linewidth=3, label="H. Final")
    plt.title("Hamiltonians")
    plt.ylabel("Energy (J)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def plot_lagrangians(L_ana, L_cal, L_f):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(L_ana, linewidth=2, label="L. Ana")
    plt.plot(L_cal, linewidth=2, label="L. Cal")
    plt.plot(L_f, linewidth=2, label="L. Final")
    plt.title("Lagrangians from the Snake")
    plt.ylabel("Energy (J)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def plot_energies(V_ana, T_cal, V_cal, T_f, V_f):
    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(V_ana, linewidth=2, label="Pot. Ana")
    plt.plot(T_cal, linewidth=2, label="Kin. Cal")
    plt.plot(V_cal, linewidth=2, label="Pot. Cal")
    plt.plot(T_f, linewidth=2, label="Kin. Final")
    plt.plot(V_f, linewidth=2, label="Pot. Final")
    plt.title("Energies from the Snake")
    plt.ylabel("Energy Level (J)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def display_results(losses: tuple):
    train_losses, test_losses = losses
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.yscale('log')
    # plt.ylim(None, 1000)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.show()
