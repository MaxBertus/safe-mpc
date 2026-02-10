import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.animator import animator
from rich.traceback import install
install(show_locals=True)

def plotter(file_path=None, model=None, params=None, animate=False):
    '''Plot states, input and wrench of STH simulation. '''

    # *** CHECK INPUTS ***
    if any(v is None for v in (file_path, model, params)):
        missing = [n for n, v in (("file_path", file_path), ("model", model), ("params", params)) if v is None]
        raise ValueError(f"ERROR [plotter]: missing {', '.join(missing)}")

    # *** LOAD AND PREPARE DATA ***
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    xHistory = data["xg"]        # (Ntraj, N+1, 12)
    uHistory = data["ug"]        # (Ntraj, N, 6)

    print(f"Loaded data from {file_path}")

    # Extract first simulation of batch
    if xHistory.ndim == 3:
        N = xHistory.shape[1] - 1
        xHistory = xHistory[-1, :, :]   # (N+1, 12)
        uHistory = uHistory[-1, :, :]   # (N, 6)
    else:
        N = xHistory.shape[0] - 1
        xHistory = xHistory[:N, :]      # (N, 12)
        uHistory = uHistory[:, :]       # (N, 6)

    # Define time array
    time = params.time

    # Define state and input references
    x_ref = params.x_ref
    if params.use_u_ref_hovering:
        uref = np.linalg.pinv(model.R(params.x_ref).full() @ model.F) @ np.array([0, 0, params.mass * params.g])
    else:
        uref = np.zeros(params.nu)

    # Helper function
    rad2deg = lambda x: x * 180.0 / np.pi


    # *** PLOTTER ***
    # States [0:6]
    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("First part of state = [positions, euler angles]")

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    ylabels = ["pos [m]", "pos [m]", "pos [m]",
            "angle [deg]", "angle [deg]", "angle [deg]"]

    for i, ax in enumerate(axs.flat):
        if i < 3:
            ax.axhline(y=x_ref[i], color='r', linestyle='--')
            ax.plot(time, xHistory[:, i], color='b')
        else:
            ax.axhline(y=rad2deg(x_ref[i]), color='r', linestyle='--')
            ax.plot(time, rad2deg(xHistory[:, i]), color='b')
            
        ax.set_title(labels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel("time [s]")
        ax.grid(True)
        ax.legend(["reference", "actual"])

    # States [7:12] 
    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("Second part of state = [linear velocities, angular velocities]")

    labels = ["v_x", "v_y", "v_z", "ω_x", "ω_y", "ω_z"]
    ylabels = ["vel [m/s]", "vel [m/s]", "vel [m/s]",
            "ang vel [deg/s]", "ang vel [deg/s]", "ang vel [deg/s]"]

    for i, ax in enumerate(axs.flat):
        idx = i + 6
        if i < 3:
            ax.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax.plot(time, xHistory[:, idx], color='b')
        else:
            ax.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax.plot(time, rad2deg(xHistory[:, idx]), color='b')

        ax.set_title(labels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel("time [s]")
        ax.grid(True)
        ax.legend(["actual"])

    # Control inputs 
    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("Control Inputs (ω²)")

    for i, ax in enumerate(axs.flat):
        ax.axhline(y=params.u_bar, color='g', linestyle='-.')
        ax.axhline(y= params.u_bar * uref[i], color='r', linestyle='--', label="reference")
        ax.step(time, params.u_bar * uHistory[:, i], color='b', where="post", label="actual")
        ax.set_ylim([0, params.u_bar*1.1])
        ax.set_title(f"Input {i+1}")
        ax.set_ylabel("ω² [rad²/s²]")
        ax.set_xlabel("time [s]")
        ax.grid(True)
        ax.legend()

    # Produced wrench
    Force = (model.F @ uHistory.T).T   
    Torque = (model.M @ uHistory.T).T  

    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    fig.suptitle("Produced Wrench (forces and torques)")

    force_labels = ["F_x", "F_y", "F_z"]
    torque_labels = ["τ_x", "τ_y", "τ_z"]

    for i in range(3):
        axs[0, i].step(time, Force[:, i], where="post", color="b")
        axs[0, i].set_title(force_labels[i])
        axs[0, i].set_ylabel("force [N]")
        axs[0, i].grid(True)
        axs[0, i].legend(["actual"])

        axs[1, i].step(time, Torque[:, i], where="post", color="b")
        axs[1, i].set_title(torque_labels[i])
        axs[1, i].set_ylabel("torque [Nm]")
        axs[1, i].set_xlabel("time [s]")
        axs[1, i].grid(True)
        axs[1, i].legend(["actual"])

    # *** SHOW ALL FIGURES ***
    plt.show()

    # *** ANIMATION ***
    if animate:
        pos = xHistory[:, 0:3]        # (N, 3)
        angles = xHistory[:, 3:6]     # (N, 3)

        if params.maxRad != 0.0:
            ell_axes = [params.maxRad, params.maxRad, params.maxRad]
        else:
            ell_axes = None

        animator(pos, angles, params, ellipsoid_axes=ell_axes)

