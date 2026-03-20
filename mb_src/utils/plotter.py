import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter
from utils.animator import animator
from utils.animator_vboc import animator
from rich.traceback import install
import os

install(show_locals=True)

# --- LaTeX rendering and global font settings ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


def plotter(
    file_path: str | None = None,
    model: object | None = None,
    params: object | None = None,
    animate: bool = False,
) -> None:
    """Load a saved trajectory and produce publication-quality PDF plots.

    Generates two 2×3 grid figures and individual single-axis PDFs for:
    - States part 1: position (x, y, z) and orientation (φ, θ, ψ)
    - States part 2: linear (vx, vy, vz) and angular (ωx, ωy, ωz) velocities
    - Inputs: six normalised squared rotor speeds
    - Wrench: body-frame forces (Fx, Fy, Fz) and torques (τx, τy, τz)

    Optionally launches an animation if ``animate=True``.

    Parameters
    ----------
    file_path : str or None
        Path to the ``trajectory.pkl`` file produced by the MPC simulation.
    model : object or None
        Robot model exposing ``R``, ``F``, and ``M`` attributes.
    params : object or None
        Configuration object providing ``dt``, ``x_ref``, ``u_bar``,
        ``mass``, ``g``, ``nu``, ``plots_dir``, and
        ``use_u_ref_hovering``.
    animate : bool, optional
        If True, launch the 3-D animation after saving the plots.
        Default is False.

    Raises
    ------
    ValueError
        If any of ``file_path``, ``model``, or ``params`` is None.
    """
    # --- Figure size constants ---
    FIG_MULTI = (24, 10)    # 2×3 grid figures
    FIG_SINGLE = (8, 3)     # Individual subplot figures

    # --- Plot style ---
    ACTUAL_COLOR = "#1171BE"
    ACTUAL_WIDTH = 1.8

    # --- Validate inputs ---
    if any(v is None for v in (file_path, model, params)):
        missing = [
            name for name, val in (
                ("file_path", file_path),
                ("model", model),
                ("params", params),
            )
            if val is None
        ]
        raise ValueError(f"ERROR [plotter]: missing {', '.join(missing)}")

    # --- Load trajectory data ---
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Support both single-trajectory (ndarray) and multi-IC (list) formats
    if isinstance(data["xg"], list):
        xHistory = data["xg"][0]
        uHistory = data["ug"][0]
        bHistory = data.get("bg", [None])[0]
    else:
        xHistory = data["xg"]
        uHistory = data["ug"]
        bHistory = data.get("bg", None)

    print(f"Loaded data from {file_path}")

    # --- Build time vector and trim state history ---
    N = xHistory.shape[0] - 1
    xHistory = xHistory[:N, :]
    uHistory = uHistory[:, :]
    time = np.arange(N) * params.dt

    x_ref = params.x_ref

    # --- Compute hovering reference input ---
    if params.use_u_ref_hovering:
        uref = (
            np.linalg.pinv(model.R(params.x_ref).full() @ model.F)
            @ np.array([0, 0, params.mass * params.g])
        )
    else:
        uref = np.zeros(params.nu)

    rad2deg = lambda x: x * 180.0 / np.pi   # noqa: E731

    os.makedirs(params.plots_dir, exist_ok=True)

    def refine_time_axis(ax: plt.Axes) -> None:
        """Apply consistent time-axis formatting to an Axes object.

        Parameters
        ----------
        ax : plt.Axes
            The axes whose x-axis will be formatted.
        """
        ax.set_xlim([0, time[-1] + params.dt])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # =========================================================================
    # STATES PART 1 — position and orientation
    # =========================================================================

    labels1 = [r"$x$", r"$y$", r"$z$", r"$\phi$", r"$\theta$", r"$\psi$"]
    units1 = [r"[m]", r"[m]", r"[m]", r"[deg]", r"[deg]", r"[deg]"]

    fig, axs = plt.subplots(2, 3, figsize=FIG_MULTI, sharex=True)

    for i, ax in enumerate(axs.flat):
        if i < 3:
            # Position: plot in metres
            ax.axhline(y=x_ref[i], color='r', linestyle='--')
            ax.plot(time, xHistory[:, i],
                    color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        else:
            # Orientation: convert radians to degrees
            ax.axhline(y=rad2deg(x_ref[i]), color='r', linestyle='--')
            ax.plot(time, rad2deg(xHistory[:, i]),
                    color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)

        ax.set_ylabel(labels1[i] + " " + units1[i])
        ax.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax)
        ax.grid(True)
        ax.legend(["reference", "actual"])

    fig.savefig(
        os.path.join(params.plots_dir, "states_part1.pdf"),
        format="pdf", bbox_inches="tight",
    )

    # Individual single-axis PDFs for states 1–6
    for i in range(6):
        fig_s, ax_s = plt.subplots(figsize=FIG_SINGLE)

        if i < 3:
            ax_s.axhline(y=x_ref[i], color='r', linestyle='--')
            ax_s.plot(time, xHistory[:, i],
                      color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        else:
            ax_s.axhline(y=rad2deg(x_ref[i]), color='r', linestyle='--')
            ax_s.plot(time, rad2deg(xHistory[:, i]),
                      color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)

        ax_s.set_ylabel(labels1[i] + " " + units1[i])
        ax_s.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_s)
        ax_s.grid(True)

        fig_s.savefig(
            os.path.join(params.plots_dir, f"state_{i + 1}.pdf"),
            format="pdf", bbox_inches="tight",
        )
        plt.close(fig_s)

    # =========================================================================
    # STATES PART 2 — linear and angular velocities
    # =========================================================================

    labels2 = [
        r"$v_x$", r"$v_y$", r"$v_z$",
        r"$\omega_x$", r"$\omega_y$", r"$\omega_z$",
    ]
    units2 = [
        r"[m/s]", r"[m/s]", r"[m/s]",
        r"[deg/s]", r"[deg/s]", r"[deg/s]",
    ]

    fig, axs = plt.subplots(2, 3, figsize=FIG_MULTI, sharex=True)

    for i, ax in enumerate(axs.flat):
        idx = i + 6   # Offset into the full state vector

        if i < 3:
            # Linear velocity: plot in m/s
            ax.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax.plot(time, xHistory[:, idx],
                    color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        else:
            # Angular velocity: convert radians to degrees
            ax.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax.plot(time, rad2deg(xHistory[:, idx]),
                    color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)

        ax.set_ylabel(labels2[i] + " " + units2[i])
        ax.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax)
        ax.grid(True)
        ax.legend(["reference", "actual"])

    fig.savefig(
        os.path.join(params.plots_dir, "states_part2.pdf"),
        format="pdf", bbox_inches="tight",
    )

    # Individual single-axis PDFs for states 7–12
    for i in range(6):
        fig_s, ax_s = plt.subplots(figsize=FIG_SINGLE)
        idx = i + 6

        if i < 3:
            ax_s.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax_s.plot(time, xHistory[:, idx],
                      color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        else:
            ax_s.axhline(y=x_ref[idx], color='r', linestyle='--')
            ax_s.plot(time, rad2deg(xHistory[:, idx]),
                      color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)

        ax_s.set_ylabel(labels2[i] + " " + units2[i])
        ax_s.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_s)
        ax_s.grid(True)

        fig_s.savefig(
            os.path.join(params.plots_dir, f"state_{i + 7}.pdf"),
            format="pdf", bbox_inches="tight",
        )
        plt.close(fig_s)

    # =========================================================================
    # INPUTS — normalised squared rotor speeds
    # =========================================================================

    fig, axs = plt.subplots(2, 3, figsize=FIG_MULTI, sharex=True)

    for i, ax in enumerate(axs.flat):
        ax.axhline(y=params.u_bar, color='g', linestyle='-.', label=r"$\bar{\omega}^2$")
        ax.axhline(y=params.u_bar * uref[i], color='r', linestyle='--', label="reference")
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=ACTUAL_WIDTH)
        ax.step(time, params.u_bar * uHistory[:, i],
                color=ACTUAL_COLOR, where="post",
                linewidth=ACTUAL_WIDTH, label="actual")

        ax.set_ylabel(rf"$u_{i + 1}$ [Hz$^2$]")
        ax.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax)
        ax.grid(True)

    fig.savefig(
        os.path.join(params.plots_dir, "inputs.pdf"),
        format="pdf", bbox_inches="tight",
    )

    # Individual single-axis PDFs for inputs 1–6
    for i in range(6):
        fig_s, ax_s = plt.subplots(figsize=FIG_SINGLE)

        ax_s.axhline(y=params.u_bar, color='g', linestyle='-.')
        ax_s.axhline(y=params.u_bar * uref[i], color='r', linestyle='--')
        ax_s.axhline(y=0, color='gray', linestyle='-')
        ax_s.step(time, params.u_bar * uHistory[:, i],
                  color=ACTUAL_COLOR, where="post", linewidth=ACTUAL_WIDTH)

        ax_s.set_ylabel(rf"$u_{i + 1}$ [Hz$^2$]")
        ax_s.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_s)
        ax_s.grid(True)

        fig_s.savefig(
            os.path.join(params.plots_dir, f"input_{i + 1}.pdf"),
            format="pdf", bbox_inches="tight",
        )
        plt.close(fig_s)

    # =========================================================================
    # WRENCH — body-frame forces and torques
    # =========================================================================

    # Compute force and torque histories from the allocation matrices
    Force = (model.F @ uHistory.T).T     # Shape (N, 3) [N]
    Torque = (model.M @ uHistory.T).T    # Shape (N, 3) [Nm]

    force_labels = [r"$F_x$", r"$F_y$", r"$F_z$"]
    torque_labels = [r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

    fig, axs = plt.subplots(2, 3, figsize=FIG_MULTI, sharex=True)

    for i in range(3):
        axs[0, i].step(time, Force[:, i], where="post",
                       color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        axs[0, i].set_ylabel(force_labels[i] + r" [N]")
        axs[0, i].set_xlabel(r"$t$ [s]")
        refine_time_axis(axs[0, i])
        axs[0, i].grid(True)

        axs[1, i].step(time, Torque[:, i], where="post",
                       color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        axs[1, i].set_ylabel(torque_labels[i] + r" [Nm]")
        axs[1, i].set_xlabel(r"$t$ [s]")
        refine_time_axis(axs[1, i])
        axs[1, i].grid(True)

    fig.savefig(
        os.path.join(params.plots_dir, "wrench.pdf"),
        format="pdf", bbox_inches="tight",
    )

    # Individual single-axis PDFs for forces and torques
    for i in range(3):
        fig_f, ax_f = plt.subplots(figsize=FIG_SINGLE)
        ax_f.step(time, Force[:, i], where="post",
                  color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        ax_f.set_ylabel(force_labels[i] + r" [N]")
        ax_f.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_f)
        ax_f.grid(True)
        fig_f.savefig(
            os.path.join(params.plots_dir, f"force_{i + 1}.pdf"),
            format="pdf", bbox_inches="tight",
        )
        plt.close(fig_f)

        fig_t, ax_t = plt.subplots(figsize=FIG_SINGLE)
        ax_t.step(time, Torque[:, i], where="post",
                  color=ACTUAL_COLOR, linewidth=ACTUAL_WIDTH)
        ax_t.set_ylabel(torque_labels[i] + r" [Nm]")
        ax_t.set_xlabel(r"$t$ [s]")
        refine_time_axis(ax_t)
        ax_t.grid(True)
        fig_t.savefig(
            os.path.join(params.plots_dir, f"torque_{i + 1}.pdf"),
            format="pdf", bbox_inches="tight",
        )
        plt.close(fig_t)

    plt.show()

    # =========================================================================
    # ANIMATION
    # =========================================================================

    if animate:
        pos = xHistory[:, 0:3]      # Position trajectory [m]
        angles = xHistory[:, 3:6]   # Euler angles trajectory [rad]

        if bHistory is not None:
            animator(pos, angles, bHistory, params)
        else:
            animator(pos, angles, params)