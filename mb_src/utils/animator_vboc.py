import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


# =============================================================================
# ROTATION UTILITIES
# =============================================================================

def rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Build a ZYX rotation matrix from Euler angles.

    Parameters
    ----------
    roll : float
        Rotation about the x-axis [rad].
    pitch : float
        Rotation about the y-axis [rad].
    yaw : float
        Rotation about the z-axis [rad].

    Returns
    -------
    R : np.ndarray
        3×3 rotation matrix: world ← body.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0, 0, 1]])
    Ry = np.array([[cp,  0, sp], [ 0,   1, 0], [-sp, 0, cp]])
    Rx = np.array([[ 1,  0,  0], [ 0,  cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def axis_angle_rotation(
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Build a rotation matrix from an axis–angle representation.

    Uses the Rodrigues rotation formula.

    Parameters
    ----------
    axis : np.ndarray
        Rotation axis (need not be unit length), shape (3,).
    angle : float
        Rotation angle [rad].

    Returns
    -------
    R : np.ndarray
        3×3 rotation matrix corresponding to the given axis and angle.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [        0, -axis[2],  axis[1]],
        [ axis[2],         0, -axis[0]],
        [-axis[1],  axis[0],         0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotor_disc(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float = 0.08,
    n_points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Generate the geometry of a filled rotor disc as a triangle fan.

    Parameters
    ----------
    center : np.ndarray
        Centre of the disc in world coordinates, shape (3,).
    normal : np.ndarray
        Normal vector of the disc plane (need not be unit length), shape (3,).
    radius : float, optional
        Disc radius [m]. Default is 0.08.
    n_points : int, optional
        Number of points on the circumference. Default is 40.

    Returns
    -------
    x, y, z : np.ndarray
        Vertex coordinates (centre first, then circumference), shape (n_points+1,).
    triangles : list of list of int
        Triangle connectivity list for ``plot_trisurf``.
    """
    normal = normal / np.linalg.norm(normal)

    # Build an orthonormal basis in the disc plane
    t1 = np.cross(normal, [0, 0, 1] if abs(normal[2]) < 0.9 else [0, 1, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle_pts = np.array([
        center + radius * (np.cos(t) * t1 + np.sin(t) * t2) for t in theta
    ])

    # Vertex 0 is the centre; vertices 1…n_points are on the circumference
    x = np.hstack([center[0], circle_pts[:, 0]])
    y = np.hstack([center[1], circle_pts[:, 1]])
    z = np.hstack([center[2], circle_pts[:, 2]])

    # Fan triangles: (centre, i, i+1) for each edge
    triangles = [[0, i, i + 1] for i in range(1, n_points)] + [[0, n_points, 1]]
    return x, y, z, triangles


def build_box_faces(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> list[list[tuple[float, float, float]]]:
    """Return the six quad faces of an axis-aligned box as vertex lists.

    Parameters
    ----------
    xmin, xmax : float
        Box extents along the x-axis [m].
    ymin, ymax : float
        Box extents along the y-axis [m].
    zmin, zmax : float
        Box extents along the z-axis [m].

    Returns
    -------
    faces : list of list of tuple
        Six faces, each defined by four (x, y, z) corner tuples, suitable
        for passing directly to ``Poly3DCollection``.
    """
    return [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
    ]


# =============================================================================
# ANIMATION UPDATE
# =============================================================================

def update(
    i: int,
    pos: np.ndarray,
    angles: np.ndarray,
    axx: plt.Artist,
    axy: plt.Artist,
    axz: plt.Artist,
    trail_line: plt.Artist,
    time_text: plt.Text,
    arms: list,
    rotor_surfs: list,
    sphere_surf: list,
    dt: float,
    ax: plt.Axes,
    params: object,
    box_collection: Poly3DCollection | None,
    box_array: np.ndarray | None,
    save_frames: bool = False,
    frames_dir: str | None = None,
    save_every: int = 10,
) -> list:
    """Update all animated artists for frame ``i``.

    Called by ``FuncAnimation`` at each frame.  Updates the body-frame
    axes, arm lines, rotor discs, safety sphere, animated bounding box,
    position trail, and elapsed-time label.  Optionally saves individual
    frames as PDF files.

    Parameters
    ----------
    i : int
        Current frame index.
    pos : np.ndarray
        Position trajectory, shape (T, 3) [m].
    angles : np.ndarray
        Euler-angle trajectory (roll, pitch, yaw), shape (T, 3) [rad].
    axx, axy, axz : plt.Artist
        Line artists for the body x, y, z axes respectively.
    trail_line : plt.Artist
        Line artist for the position trail.
    time_text : plt.Text
        Text artist displaying the elapsed time.
    arms : list of plt.Artist
        Six line artists, one per rotor arm.
    rotor_surfs : list
        Mutable list of ``TriSurf`` objects (cleared and rebuilt each frame).
    sphere_surf : list of length 1
        Mutable single-element list holding the safety sphere surface,
        or ``[None]`` if not yet created.
    dt : float
        Control time step [s].
    ax : plt.Axes
        The 3-D axes object.
    params : object
        Configuration object exposing ``alpha_tilt``, ``propRad``, and
        ``maxRad``.
    box_collection : Poly3DCollection or None
        Mutable collection for the animated free-space box, or None if
        no box data is provided.
    box_array : np.ndarray or None
        Array of box parameters ``[xmin, xmax, ymin, ymax, zmin, zmax]``
        per frame, shape (T, 6), or None.
    save_frames : bool, optional
        If True, save a PDF snapshot every ``save_every`` frames.
        Default is False.
    frames_dir : str or None, optional
        Directory in which to save frame PDFs.  Required when
        ``save_frames`` is True.
    save_every : int, optional
        Interval between saved frames. Default is 10.

    Returns
    -------
    artists : list
        Empty list (blitting not used).
    """
    p = pos[i]
    roll, pitch, yaw = angles[i]
    R = rotation_matrix(roll, pitch, yaw)

    # --- Body-frame axis arrows ---
    scale = 0.15
    ex = p + scale * R[:, 0]
    ey = p + scale * R[:, 1]
    ez = p + scale * R[:, 2]

    axx.set_data([p[0], ex[0]], [p[1], ex[1]])
    axx.set_3d_properties([p[2], ex[2]])
    axy.set_data([p[0], ey[0]], [p[1], ey[1]])
    axy.set_3d_properties([p[2], ey[2]])
    axz.set_data([p[0], ez[0]], [p[1], ez[1]])
    axz.set_3d_properties([p[2], ez[2]])

    # --- Arms and rotor centres ---
    arm_len = 0.385
    arm_angles = np.arange(6) * np.pi / 3   # 60° apart
    rotor_centers, arm_dirs_body = [], []

    for k, ang in enumerate(arm_angles):
        p_body = np.array([arm_len * np.cos(ang), arm_len * np.sin(ang), 0])
        pw = p + R @ p_body
        arms[k].set_data([p[0], pw[0]], [p[1], pw[1]])
        arms[k].set_3d_properties([p[2], pw[2]])
        rotor_centers.append(pw)
        arm_dirs_body.append(p_body)

    # --- Rotor discs (rebuilt each frame to reflect attitude changes) ---
    for surf in rotor_surfs:
        surf.remove()
    rotor_surfs.clear()

    colors = ["cornflowerblue", "red"] * 3   # Alternating CW/CCW rotor colours

    for k, (pc, arm_b) in enumerate(zip(rotor_centers, arm_dirs_body)):
        arm_dir_world = R @ (arm_b / np.linalg.norm(arm_b))
        sign = (-1) ** k
        R_tilt = axis_angle_rotation(arm_dir_world, sign * params.alpha_tilt)
        rotor_normal = R_tilt @ R[:, 2]
        x, y, z, tris = rotor_disc(pc, rotor_normal, radius=params.propRad)
        surf = ax.plot_trisurf(x, y, z, triangles=tris,
                               color=colors[k], alpha=0.5, linewidth=0)
        rotor_surfs.append(surf)

    # --- Safety sphere (follows the drone centre) ---
    if sphere_surf[0] is not None:
        sphere_surf[0].remove()
        sphere_surf[0] = None

    if params.maxRad > 0:
        u_grid, v_grid = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        r = params.maxRad
        xs = p[0] + r * np.cos(u_grid) * np.sin(v_grid)
        ys = p[1] + r * np.sin(u_grid) * np.sin(v_grid)
        zs = p[2] + r * np.cos(v_grid)
        sphere_surf[0] = ax.plot_surface(
            xs, ys, zs, color="cyan", alpha=0.2, linewidth=0
        )

    # --- Animated free-space bounding box ---
    if box_collection is not None and box_array is not None:
        box_collection.set_verts(build_box_faces(*box_array[i]))

    # --- Position trail ---
    trail_line.set_data(pos[:i + 1, 0], pos[:i + 1, 1])
    trail_line.set_3d_properties(pos[:i + 1, 2])

    # --- Elapsed time label ---
    time_text.set_text(f"t = {i * dt:.2f} s")

    # --- Optional frame snapshot ---
    if save_frames and frames_dir is not None and i % save_every == 0:
        os.makedirs(frames_dir, exist_ok=True)
        plt.savefig(
            os.path.join(frames_dir, f"frame_{i:05d}.pdf"),
            format="pdf", bbox_inches="tight",
        )

    return []


# =============================================================================
# ANIMATOR
# =============================================================================

def animator(
    pos: np.ndarray,
    angles: np.ndarray,
    box: np.ndarray | None,
    params: object,
    save_frames: bool = True,
    save_every: int = 10,
    save_video: bool = True,
) -> None:
    """Launch a 3-D animation of the drone trajectory with obstacles and bounding box.

    Sets up the Matplotlib figure in a lateral (X–Z) view, draws static
    scene elements (world frame, obstacles, reference point), initialises
    animated artists, and starts the ``FuncAnimation`` loop.
    Optionally saves individual PDF frames and/or an MP4 video.

    Parameters
    ----------
    pos : np.ndarray
        Position trajectory in world frame, shape (T, 3) [m].
    angles : np.ndarray
        Euler-angle trajectory (roll, pitch, yaw), shape (T, 3) [rad].
    box : np.ndarray or None
        Free-space box parameters.  Either a constant 1-D array of shape
        (6,) or a time-varying 2-D array of shape (T, 6) with columns
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``.  Pass None to disable.
    params : object
        Configuration object exposing ``dt``, ``xlim``, ``ylim``,
        ``zlim``, ``alpha_tilt``, ``propRad``, ``maxRad``,
        optionally ``x_ref``, ``obstacles``, and ``plots_dir``.
    save_frames : bool, optional
        If True, save a PDF snapshot every ``save_every`` frames to
        ``<plots_dir>/frames/``. Default is True.
    save_every : int, optional
        Interval between saved frames. Default is 10.
    save_video : bool, optional
        If True, render and save an MP4 video to
        ``<plots_dir>/animation.mp4``. Default is True.
    """
    dt = params.dt
    num_steps = pos.shape[0]

    frames_dir = os.path.join(getattr(params, "plots_dir", "."), "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # --- Figure and axes setup ---
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(params.xlim)
    ax.set_ylim(params.ylim)
    ax.set_zlim(params.zlim)

    x_range = params.xlim[1] - params.xlim[0]
    y_range = params.ylim[1] - params.ylim[0]
    z_range = params.zlim[1] - params.zlim[0]
    ax.set_box_aspect([x_range, y_range, z_range])

    # Lateral view: look along the y-axis to show the X–Z plane
    ax.view_init(elev=0, azim=-90)

    # --- Tick and label formatting ---
    LABEL_SIZE = 20
    TICK_SIZE = 18
    tick_spacing = 1.0

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.tick_params(axis="x", labelsize=TICK_SIZE, pad=2)
    ax.tick_params(axis="z", labelsize=TICK_SIZE, pad=10)   # Extra padding for z labels

    ax.set_xlabel("x [m]", fontsize=LABEL_SIZE, labelpad=6)
    ax.set_ylabel("",       fontsize=LABEL_SIZE)
    ax.set_zlabel("z [m]", fontsize=LABEL_SIZE, labelpad=16)

    # Hide y-axis tick labels (grid lines remain visible)
    ax.tick_params(axis="y", labelsize=0, length=0)
    ax.yaxis.set_ticklabels([])
    ax.grid(True)

    fig.tight_layout()

    # --- Static world reference frame ---
    L = 0.4
    ax.plot([0, L], [0, 0], [0, 0], color="r", linewidth=2)
    ax.plot([0, 0], [0, L], [0, 0], color="g", linewidth=2)
    ax.plot([0, 0], [0, 0], [0, L], color="b", linewidth=2)

    # --- Animated body-frame axes ---
    axx, = ax.plot([], [], [], color="r", linewidth=2)
    axy, = ax.plot([], [], [], color="g", linewidth=2)
    axz, = ax.plot([], [], [], color="b", linewidth=2)

    # --- Animated arm lines (one per rotor) ---
    arms = [ax.plot([], [], [], color="k", linewidth=2)[0] for _ in range(6)]

    # Rotor surfaces rebuilt each frame; safety sphere in mutable list
    rotor_surfs = []
    sphere_surf = [None]

    trail_line, = ax.plot([], [], [], color="cornflowerblue", linewidth=0.8)

    # --- Reference position marker ---
    if hasattr(params, "x_ref"):
        xr = params.x_ref[:3]
        ax.plot([xr[0]], [xr[1]], [xr[2]],
                "rx", markersize=12, markeredgewidth=2.5)

    time_text = fig.text(0.52, 0.12, "", ha="center", fontsize=16)

    # --- Animated free-space bounding box ---
    box_collection, box_array = None, None

    if box is not None:
        box_np = np.asarray(box)
        # Broadcast a constant box to every frame if a single set of params is given
        box_array = np.tile(box_np, (num_steps, 1)) if box_np.ndim == 1 else box_np
        box_collection = Poly3DCollection(
            build_box_faces(*box_array[0]),
            facecolor=(1, 0, 0, 0.15),
            edgecolor=(1, 0, 0, 0.6),
            linewidth=0.8,
        )
        ax.add_collection3d(box_collection)

    # --- Static obstacle geometry ---
    for obs in getattr(params, "obstacles", []):
        if obs["type"] == "sphere":
            u_grid, v_grid = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            c, r = obs["center"], obs["radius"]
            ax.plot_surface(
                c[0] + r * np.cos(u_grid) * np.sin(v_grid),
                c[1] + r * np.sin(u_grid) * np.sin(v_grid),
                c[2] + r * np.cos(v_grid),
                color="gray", alpha=0.3, linewidth=0,
            )

        elif obs["type"] == "box":
            c, d = obs["center"], obs["dimensions"]
            x_min, y_min, z_min = c[0] - d[0] / 2, c[1] - d[1] / 2, c[2] - d[2] / 2
            x_max, y_max, z_max = c[0] + d[0] / 2, c[1] + d[1] / 2, c[2] + d[2] / 2
            ax.add_collection3d(
                Poly3DCollection(
                    build_box_faces(x_min, x_max, y_min, y_max, z_min, z_max),
                    facecolor="gray", edgecolor="darkgray",
                    alpha=0.3, linewidths=1,
                )
            )

    # --- Launch animation ---
    ani = FuncAnimation(   # noqa: F841  (kept alive by plt.show / ani.save)
        fig,
        update,
        frames=num_steps,
        interval=dt * 1000,
        fargs=(
            pos, angles,
            axx, axy, axz,
            trail_line, time_text,
            arms, rotor_surfs,
            sphere_surf,
            dt, ax, params,
            box_collection, box_array,
            save_frames, frames_dir, save_every,
        ),
    )

    # --- Optional MP4 export ---
    if save_video:
        video_path = os.path.join(getattr(params, "plots_dir", "."), "animation.mp4")
        ani.save(video_path, writer=FFMpegWriter(fps=int(1 / dt)))
        print(f"Video saved at: {video_path}")

    plt.show()