import pickle
import numpy as np
import matplotlib.pyplot as plt
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel, SthModel
import safe_mpc.animator
from rich.traceback import install
install(show_locals=True)

def plotter(file_path=None, animate=False, Duration=10.0, dt=0.005):
    ### USER PARAMETERS ###
    if file_path is None:
        file_path = "./data_noise/sth_naiveSth_45hor_10sm_use_netNone__q_collision_margins_0_0_guess.pkl"  

    u_max = 108**2                          # actuator saturation (if known)
    u_max = u_max * 1.1                     # for plotting limits

    # Optional reference (set to None if unavailable)
    yref = None                          # shape (N+1, nx) or None

    ### LOAD DATA  ###
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    xHistory = data["xg"]        # (Ntraj, N+1, 12)
    uHistory = data["ug"]        # (Ntraj, N, 6)

    print(f"Loaded data from {file_path}")
    print(f"xHistory shape: {xHistory.shape}")

    if xHistory.ndim == 3:
        N = xHistory.shape[1] - 1
        xHistory = xHistory[-1, :, :]   # (N+1, 12)
        uHistory = uHistory[-1, :, :]   # (N, 6)
    else:
        N = xHistory.shape[0] - 1
        xHistory = xHistory[:N, :]   # (N, 12)
        uHistory = uHistory[:, :]   # (N, 6)

    # time = np.arange(0, N * Ts, Ts)
    time = np.arange(0, Duration, dt)

    # Helper
    rad2deg = lambda x: x * 180.0 / np.pi

    ### PLOTTER ###
    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("First part of state = [positions, euler angles]")

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    ylabels = ["pos [m]", "pos [m]", "pos [m]",
            "angle [deg]", "angle [deg]", "angle [deg]"]

    for i, ax in enumerate(axs.flat):
        if i < 3:
            ax.plot(time, xHistory[:, i])
            if yref is not None and yref.shape[1] > i:
                ax.plot(time, yref[:, i])
        else:
            ax.plot(time, rad2deg(xHistory[:, i]))
            if yref is not None and yref.shape[1] > i:
                ax.plot(time, rad2deg(yref[:, i]))

        ax.set_title(labels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel("time [s]")
        ax.grid(True)
        ax.legend(["actual", "reference"] if yref is not None else ["actual"])

    # States [7:12]
    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("Second part of state = [linear velocities, angular velocities]")

    labels = ["v_x", "v_y", "v_z", "ω_x", "ω_y", "ω_z"]
    ylabels = ["vel [m/s]", "vel [m/s]", "vel [m/s]",
            "ang vel [deg/s]", "ang vel [deg/s]", "ang vel [deg/s]"]

    for i, ax in enumerate(axs.flat):
        idx = i + 6
        if i < 3:
            ax.plot(time, xHistory[:, idx])
            if yref is not None and yref.shape[1] > idx:
                ax.plot(time, yref[:, idx])
        else:
            ax.plot(time, rad2deg(xHistory[:, idx]))
            if yref is not None and yref.shape[1] > idx:
                ax.plot(time, rad2deg(yref[:, idx]))

        ax.set_title(labels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel("time [s]")
        ax.grid(True)
        ax.legend(["actual", "reference"] if yref is not None else ["actual"])

    # Control inputs
    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("Control Inputs (ω²)")

    for i, ax in enumerate(axs.flat):
        ax.step(time, uHistory[:, i], where="post")
        ax.set_ylim([0, u_max])
        ax.set_title(f"Input {i+1}")
        ax.set_ylabel("ω² [rad²/s²]")
        ax.set_xlabel("time [s]")
        ax.grid(True)
        ax.legend(["actual"])

    # Produced wrench (if allocation matrices available)
    args = parse_args()
    model_name = args['system'] 
    params = Parameters(args,model_name, rti=False)
    model = SthModel(params) 

    Force = (model.F @ uHistory.T).T   
    Torque = (model.M @ uHistory.T).T  

    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("Produced Wrench (forces and torques)")

    force_labels = ["F_x", "F_y", "F_z"]
    torque_labels = ["τ_x", "τ_y", "τ_z"]

    for i in range(3):
        axs[0, i].step(time, Force[:, i], where="post")
        axs[0, i].set_title(force_labels[i])
        axs[0, i].set_ylabel("force [N]")
        axs[0, i].grid(True)
        axs[0, i].legend(["actual"])

        axs[1, i].step(time, Torque[:, i], where="post")
        axs[1, i].set_title(torque_labels[i])
        axs[1, i].set_ylabel("torque [Nm]")
        axs[1, i].set_xlabel("time [s]")
        axs[1, i].grid(True)
        axs[1, i].legend(["actual"])

    ### ANIMATION ###
    # pos = xHistory[:, 0:3]        # (N, 3)
    # angles = xHistory[:, 3:6]     # (N, 3)

    # animator.animator(pos, angles, dt=Ts, num_steps=N)

    ### SHOW ALL FIGURES ###
    plt.show()