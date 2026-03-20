import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

    Rz = np.array([
        [ cy, -sy, 0],
        [ sy,  cy, 0],
        [  0,   0, 1],
    ])
    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp],
    ])
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr],
    ])
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
    if abs(normal[2]) < 0.9:
        t1 = np.cross(normal, [0, 0, 1])
    else:
        t1 = np.cross(normal, [0, 1, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle_pts = np.array([
        center + radius * (np.cos(t) * t1 + np.sin(t) * t2)
        for t in theta
    ])

    # Vertex 0 is the centre; vertices 1…n_points are on the circumference
    x = np.hstack([center[0], circle_pts[:, 0]])
    y = np.hstack([center[1], circle_pts[:, 1]])
    z = np.hstack([center[2], circle_pts[:, 2]])

    # Fan triangles: (centre, i, i+1) for each edge
    triangles = [[0, i, i + 1] for i in range(1, n_points)]
    triangles.append([0, n_points, 1])   # Closing triangle

    return x, y, z, triangles


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
) -> list:
    """Update all animated artists for frame ``i``.

    Called by ``FuncAnimation`` at each frame.  Updates the body-frame
    axes, arm lines, rotor discs, safety sphere, position trail, and
    elapsed-time label.

    Parameters
    ----------
    i : int
        Current frame index.
    pos : np.ndarray
        Position trajectory, shape (T, 3).
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
        Mutable single-element list holding the safety sphere surface
        (or None if not yet created).
    dt : float
        Control time step [s], used to compute elapsed time.
    ax : plt.Axes
        The 3-D axes object.
    params : object
        Configuration object exposing ``alpha_tilt``, ``propRad``, and
        ``maxRad``.

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

    rotor_centers = []
    arm_dirs_body = []

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

    colors = ["blue", "red"] * 3   # Alternating colours for CW/CCW rotors

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

    # --- Position trail ---
    trail_line.set_data(pos[:i + 1, 0], pos[:i + 1, 1])
    trail_line.set_3d_properties(pos[:i + 1, 2])

    # --- Elapsed time label ---
    time_text.set_text(f"t = {i * dt:.2f} s")

    return []


# =============================================================================
# ANIMATOR
# =============================================================================

def animator(
    pos: np.ndarray,
    angles: np.ndarray,
    params: object,
) -> None:
    """Launch a 3-D animation of the drone trajectory with obstacles.

    Sets up the Matplotlib figure, draws static scene elements (world frame,
    obstacles), initialises animated artists, and starts the
    ``FuncAnimation`` loop.

    Parameters
    ----------
    pos : np.ndarray
        Position trajectory in world frame, shape (T, 3) [m].
    angles : np.ndarray
        Euler-angle trajectory (roll, pitch, yaw), shape (T, 3) [rad].
    params : object
        Configuration object exposing ``dt``, ``xlim``, ``ylim``,
        ``zlim``, ``alpha_tilt``, ``propRad``, ``maxRad``, and
        optionally ``obstacles``.
    """
    dt = params.dt
    num_steps = pos.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # --- Axis labels and tick formatting ---
    ax.set_xlabel("X", fontsize=18)
    ax.set_ylabel("Y", fontsize=18)
    ax.set_zlabel("Z", fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)

    ax.set_xlim(params.xlim)
    ax.set_ylim(params.ylim)
    ax.set_zlim(params.zlim)

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.zaxis.set_major_locator(MultipleLocator(1.0))

    # Equal aspect ratio based on axis ranges
    x_range = params.xlim[1] - params.xlim[0]
    y_range = params.ylim[1] - params.ylim[0]
    z_range = params.zlim[1] - params.zlim[0]
    ax.set_box_aspect([x_range, y_range, z_range])

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

    # Rotor surfaces rebuilt each frame; safety sphere stored in a list for mutability
    rotor_surfs = []
    sphere_surf = [None]

    trail_line, = ax.plot([], [], [], color="red", linewidth=0.8)
    time_text = fig.text(0.5, 0.02, "", ha="center")

    # --- Static obstacle geometry ---
    obstacles = getattr(params, "obstacles", [])

    for obs in obstacles:
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
            x_min = c[0] - d[0] / 2
            y_min = c[1] - d[1] / 2
            z_min = c[2] - d[2] / 2
            x_max = c[0] + d[0] / 2
            y_max = c[1] + d[1] / 2
            z_max = c[2] + d[2] / 2

            vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ])
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
            ]
            ax.add_collection3d(
                Poly3DCollection(faces, facecolor="gray",
                                 edgecolor="darkgray", alpha=0.3, linewidths=1)
            )

    # --- Launch animation ---
    ani = FuncAnimation(   # noqa: F841  (kept alive by plt.show)
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
        ),
    )

    plt.show()