import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from utils.animator import animator
from utils.animator_vboc import animator
from rich.traceback import install
import os
install(show_locals=True)

# --- LaTeX + font size ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

def plotter(file_path=None, model=None, params=None, animate=False):

    actualColor = "#1171BE" # "#3271DF"
    actualWidth = 1.8

    if any(v is None for v in (file_path, model, params)):
        missing = [n for n, v in (("file_path", file_path), ("model", model), ("params", params)) if v is None]
        raise ValueError(f"ERROR [plotter]: missing {', '.join(missing)}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data["xg"], list):
        xHistory = data["xg"][0]
        uHistory = data["ug"][0]
        if "bg" in data:
            bHistory = data["bg"][0]
    else:
        xHistory = data["xg"]
        uHistory = data["ug"]
        if "bg" in data:
            bHistory = data["bg"]

    print(f"Loaded data from {file_path}")

    N = xHistory.shape[0] - 1
    xHistory = xHistory[:N, :]
    uHistory = uHistory[:, :]
    time = np.arange(N) * params.dt

    x_ref = params.x_ref

    if params.use_u_ref_hovering:
        uref = np.linalg.pinv(model.R(params.x_ref).full() @ model.F) @ \
               np.array([0, 0, params.mass * params.g])
    else:
        uref = np.zeros(params.nu)

    rad2deg = lambda x: x * 180.0 / np.pi

    os.makedirs(params.plots_dir, exist_ok=True)

    def refine_time_axis(ax):
        ax.set_xlim([0, time[-1]+params.dt])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=3)

    # ==========================================================
    # STATES PART 1
    # ==========================================================

    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)

    labels1 = [r"$x$", r"$y$", r"$z$", r"$\phi$", r"$\theta$", r"$\psi$"]
    units1 = [r"[m]", r"[m]", r"[m]", r"[deg]", r"[deg]", r"[deg]"]

    for i, ax in enumerate(axs.flat):

        if i < 3:
            ax.axhline(y=x_ref[i], color='r', linestyle='--')
            ax.plot(time, xHistory[:, i], color=actualColor, linewidth=actualWidth)
        else:
            ax.axhline(y=rad2deg(x_ref[i]), color='r', linestyle='--')
            ax.plot(time, rad2deg(xHistory[:, i]), color=actualColor, linewidth=actualWidth)

        ax.set_ylabel(labels1[i] + " " + units1[i])
        ax.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax)
        ax.grid(True)
        ax.legend(["reference", "actual"])

    fig.savefig(os.path.join(params.plots_dir, "states_part1.pdf"),
                format="pdf", bbox_inches="tight")

    # --- single ---
    for i in range(6):
        fig_s, ax_s = plt.subplots(figsize=(8,5))

        if i < 3:
            ax_s.axhline(y=x_ref[i], color='r', linestyle='--')
            ax_s.plot(time, xHistory[:, i], color=actualColor, linewidth=actualWidth)
        else:
            ax_s.axhline(y=rad2deg(x_ref[i]), color='r', linestyle='--')
            ax_s.plot(time, rad2deg(xHistory[:, i]), color=actualColor, linewidth=actualWidth)

        ax_s.set_ylabel(labels1[i] + " " + units1[i])
        ax_s.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_s)
        ax_s.grid(True)

        fig_s.savefig(os.path.join(params.plots_dir, f"state_{i+1}.pdf"),
                      format="pdf", bbox_inches="tight")
        plt.close(fig_s)

    # ==========================================================
    # STATES PART 2
    # ==========================================================

    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)

    labels2 = [r"$v_x$", r"$v_y$", r"$v_z$",
               r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    units2 = [r"[m/s]", r"[m/s]", r"[m/s]",
              r"[deg/s]", r"[deg/s]", r"[deg/s]"]

    for i, ax in enumerate(axs.flat):

        idx = i + 6

        if i < 3:
            ax.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax.plot(time, xHistory[:, idx], color=actualColor, linewidth=actualWidth)
        else:
            ax.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax.plot(time, rad2deg(xHistory[:, idx]), color=actualColor, linewidth=actualWidth)

        ax.set_ylabel(labels2[i] + " " + units2[i])
        ax.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax)
        ax.grid(True)
        ax.legend(["reference", "actual"])

    fig.savefig(os.path.join(params.plots_dir, "states_part2.pdf"),
                format="pdf", bbox_inches="tight")

    for i in range(6):
        fig_s, ax_s = plt.subplots(figsize=(8,5))
        idx = i + 6

        if i < 3:
            ax_s.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax_s.plot(time, xHistory[:, idx], color=actualColor, linewidth=actualWidth)
        else:
            ax_s.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax_s.plot(time, rad2deg(xHistory[:, idx]), color=actualColor, linewidth=actualWidth)

        ax_s.set_ylabel(labels2[i] + " " + units2[i])
        ax_s.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_s)
        ax_s.grid(True)

        fig_s.savefig(os.path.join(params.plots_dir, f"state_{i+7}.pdf"),
                      format="pdf", bbox_inches="tight")
        plt.close(fig_s)

    # ==========================================================
    # INPUTS
    # ==========================================================

    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)

    for i, ax in enumerate(axs.flat):

        ax.axhline(y=params.u_bar, color='g', linestyle='-.', label = r"\bar{\omega}^2")
        ax.axhline(y=params.u_bar * uref[i], color='r', linestyle='--', label = "reference")
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=actualWidth)
        ax.step(time, params.u_bar * uHistory[:, i],
                color=actualColor, where="post", linewidth=actualWidth, label = "actual")

        ax.set_ylabel(rf"$u_{i+1}$ [Hz$^2$]")
        ax.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax)
        ax.grid(True)

    fig.savefig(os.path.join(params.plots_dir, "inputs.pdf"),
                format="pdf", bbox_inches="tight")

    for i in range(6):
        fig_s, ax_s = plt.subplots(figsize=(8,5))

        ax_s.axhline(y=params.u_bar, color='g', linestyle='-.')
        ax_s.axhline(y=params.u_bar * uref[i], color='r', linestyle='--')
        ax_s.axhline(y=0, color='gray', linestyle='-')
        ax_s.step(time, params.u_bar * uHistory[:, i],
                  color=actualColor, where="post", linewidth=actualWidth)

        ax_s.set_ylabel(rf"$u_{i+1}$ [Hz$^2$]")
        ax_s.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_s)
        ax_s.grid(True)

        fig_s.savefig(os.path.join(params.plots_dir, f"input_{i+1}.pdf"),
                      format="pdf", bbox_inches="tight")
        plt.close(fig_s)

    # ==========================================================
    # WRENCH
    # ==========================================================

    Force = (model.F @ uHistory.T).T
    Torque = (model.M @ uHistory.T).T

    fig, axs = plt.subplots(2, 3, figsize=(24, 10), sharex=True)

    force_labels = [r"$F_x$", r"$F_y$", r"$F_z$"]
    torque_labels = [r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

    for i in range(3):

        axs[0, i].step(time, Force[:, i], where="post",
                       color=actualColor, linewidth=actualWidth)
        axs[0, i].set_ylabel(force_labels[i] + r" [N]")
        axs[0, i].set_xlabel(r"$t$ [s]")
        refine_time_axis(axs[0, i])
        axs[0, i].grid(True)

        axs[1, i].step(time, Torque[:, i], where="post",
                       color=actualColor, linewidth=actualWidth)
        axs[1, i].set_ylabel(torque_labels[i] + r" [Nm]")
        axs[1, i].set_xlabel(r"$t$ [s]")
        refine_time_axis(axs[1, i])
        axs[1, i].grid(True)

    fig.savefig(os.path.join(params.plots_dir, "wrench.pdf"),
                format="pdf", bbox_inches="tight")

    for i in range(3):

        fig_f, ax_f = plt.subplots(figsize=(8,5))
        ax_f.step(time, Force[:, i], where="post",
                  color=actualColor, linewidth=actualWidth)
        ax_f.set_ylabel(force_labels[i] + r" [N]")
        ax_f.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_f)
        ax_f.grid(True)
        fig_f.savefig(os.path.join(params.plots_dir, f"force_{i+1}.pdf"),
                      format="pdf", bbox_inches="tight")
        plt.close(fig_f)

        fig_t, ax_t = plt.subplots(figsize=(8,5))
        ax_t.step(time, Torque[:, i], where="post",
                  color=actualColor, linewidth=actualWidth)
        ax_t.set_ylabel(torque_labels[i] + r" [Nm]")
        ax_t.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_t)
        ax_t.grid(True)
        fig_t.savefig(os.path.join(params.plots_dir, f"torque_{i+1}.pdf"),
                      format="pdf", bbox_inches="tight")
        plt.close(fig_t)

    plt.show()

    # ==========================================================
    # ANIMATION
    # ==========================================================

    if animate:
        pos = xHistory[:, 0:3]
        angles = xHistory[:, 3:6]

        if "bg" in data:
            animator(pos, angles, bHistory, params)
        else:
            animator(pos, angles, params)