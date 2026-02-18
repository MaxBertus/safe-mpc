import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================================================
# Rotation utilities
# =========================================================

def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rz @ Ry @ Rx


def axis_angle_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


def rotor_disc(center, normal, radius=0.08, n_points=40):
    normal = normal / np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        t1 = np.cross(normal, [0, 0, 1])
    else:
        t1 = np.cross(normal, [0, 1, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    theta = np.linspace(0, 2*np.pi, n_points)
    circle_pts = np.array([
        center + radius*(np.cos(t)*t1 + np.sin(t)*t2) for t in theta
    ])

    x = np.hstack([center[0], circle_pts[:,0]])
    y = np.hstack([center[1], circle_pts[:,1]])
    z = np.hstack([center[2], circle_pts[:,2]])

    triangles = [[0, i, i+1] for i in range(1, n_points)]
    triangles.append([0, n_points, 1])

    return x, y, z, triangles


# =========================================================
# Bounding Box Helper
# =========================================================

def build_box_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    """Return the 6 faces of an axis-aligned box as a list of quad vertices."""
    faces = [
        # Bottom (z = zmin)
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        # Top (z = zmax)
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        # Front (y = ymin)
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        # Back (y = ymax)
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        # Left (x = xmin)
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        # Right (x = xmax)
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
    ]
    return faces


# =========================================================
# Update Function
# =========================================================

def update(i, pos, angles, axx, axy, axz,
           trail_line, time_text,
           arms, rotor_surfs,
           sphere_surf,
           dt, ax, params,
           box_collection, box_array):  # box_array: None or ndarray of shape (N, 6)

    p = pos[i]
    roll, pitch, yaw = angles[i]
    R = rotation_matrix(roll, pitch, yaw)

    # Body frame
    scale = 0.15
    ex, ey, ez = p + scale*R[:,0], p + scale*R[:,1], p + scale*R[:,2]

    axx.set_data([p[0], ex[0]], [p[1], ex[1]])
    axx.set_3d_properties([p[2], ex[2]])

    axy.set_data([p[0], ey[0]], [p[1], ey[1]])
    axy.set_3d_properties([p[2], ey[2]])

    axz.set_data([p[0], ez[0]], [p[1], ez[1]])
    axz.set_3d_properties([p[2], ez[2]])

    # Arms and rotors
    arm_len = 0.385
    angles_arm = np.arange(6) * np.pi / 3

    rotor_centers = []
    arm_dirs_body = []

    for k, ang in enumerate(angles_arm):
        p_body = np.array([arm_len*np.cos(ang), arm_len*np.sin(ang), 0])
        pw = p + R @ p_body

        arms[k].set_data([p[0], pw[0]], [p[1], pw[1]])
        arms[k].set_3d_properties([p[2], pw[2]])

        rotor_centers.append(pw)
        arm_dirs_body.append(p_body)

    # Remove old rotors
    for surf in rotor_surfs:
        surf.remove()
    rotor_surfs.clear()

    colors = ["blue", "red"] * 3

    for k, (pc, arm_b) in enumerate(zip(rotor_centers, arm_dirs_body)):
        arm_dir_world = R @ (arm_b / np.linalg.norm(arm_b))
        sign = (-1)**k
        R_tilt = axis_angle_rotation(arm_dir_world, sign * params.alpha_tilt)
        rotor_normal = R_tilt @ R[:,2]

        x, y, z, tris = rotor_disc(pc, rotor_normal, radius=params.propRad)

        surf = ax.plot_trisurf(x, y, z, triangles=tris,
                               color=colors[k], alpha=0.5, linewidth=0)
        rotor_surfs.append(surf)

    # Moving safety sphere
    if sphere_surf[0] is not None:
        sphere_surf[0].remove()
        sphere_surf[0] = None

    if params.maxRad > 0:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = params.maxRad

        xs = p[0] + r * np.cos(u) * np.sin(v)
        ys = p[1] + r * np.sin(u) * np.sin(v)
        zs = p[2] + r * np.cos(v)

        sphere_surf[0] = ax.plot_surface(
            xs, ys, zs,
            color="cyan",
            alpha=0.2,
            linewidth=0
        )

    # --- Animated bounding box: update faces from row i of box_array ---
    if box_collection is not None and box_array is not None:
        box_params_i = box_array[i]   # shape (6,): xmin,xmax,ymin,ymax,zmin,zmax
        faces = build_box_faces(*box_params_i)
        box_collection.set_verts(faces)

    # Trail
    trail_line.set_data(pos[:i+1, 0], pos[:i+1, 1])
    trail_line.set_3d_properties(pos[:i+1, 2])

    time_text.set_text(f"t = {i*dt:.2f} s")

    return []


# =========================================================
# Main animator with obstacles
# =========================================================

def animator(pos, angles, box, params):
    """
    Animate a hexacopter along a pre-computed trajectory.

    A red semi-transparent bounding box is drawn if box is set.
    box can be:
      - a tuple/list (xmin, xmax, ymin, ymax, zmin, zmax): static box
      - an ndarray of shape (N, 6): per-step animated box
    """

    dt = params.dt
    num_steps = pos.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(params.xlim)
    ax.set_ylim(params.ylim)
    ax.set_zlim(params.zlim)

    # Calculate the actual dimensions for each axis
    x_range = params.xlim[1] - params.xlim[0]
    y_range = params.ylim[1] - params.ylim[0]
    z_range = params.zlim[1] - params.zlim[0]

    ax.set_box_aspect([x_range, y_range, z_range])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Global reference frame
    L = 0.4
    ax.plot([0, L], [0, 0], [0, 0], color="r", linewidth=2)
    ax.plot([0, 0], [0, L], [0, 0], color="g", linewidth=2)
    ax.plot([0, 0], [0, 0], [0, L], color="b", linewidth=2)

    # Drone reference frame
    axx, = ax.plot([], [], [], color="r", linewidth=2)
    axy, = ax.plot([], [], [], color="g", linewidth=2)
    axz, = ax.plot([], [], [], color="b", linewidth=2)

    # Arms
    arms = [ax.plot([], [], [], color="k", linewidth=2)[0] for _ in range(6)]

    # Rotors
    rotor_surfs = []

    # Safety sphere
    sphere_surf = [None]

    trail_line, = ax.plot([], [], [], color="red", linewidth=0.8)
    time_text = fig.text(0.5, 0.02, "", ha="center")

    # === Bounding box ===
    # Normalize box input: accept tuple/list (static) or ndarray (N, 6) (animated)
    box_collection = None
    box_array = None  # will be ndarray (N, 6) or None

    if box is not None:
        box_np = np.asarray(box)

        if box_np.ndim == 1:
            # Static box: replicate across all steps
            box_array = np.tile(box_np, (num_steps, 1))
        else:
            # Per-step box: shape (N, 6)
            box_array = box_np

        # Create the collection using the first frame
        faces = build_box_faces(*box_array[0])
        box_collection = Poly3DCollection(
            faces,
            facecolor=(1.0, 0.0, 0.0, 0.15),   # red, semi-transparent fill
            edgecolor=(1.0, 0.0, 0.0, 0.6),     # red edges, slightly opaque
            linewidth=0.8,
        )
        ax.add_collection3d(box_collection)

    # === Obstacles ===
    obstacles = getattr(params, "obstacles", [])
    for obs in obstacles:
        if obs["type"] == "sphere":
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            c = obs["center"]
            r = obs["radius"]
            x_sphere = c[0] + r * np.cos(u) * np.sin(v)
            y_sphere = c[1] + r * np.sin(u) * np.sin(v)
            z_sphere = c[2] + r * np.cos(v)
            ax.plot_surface(x_sphere, y_sphere, z_sphere,
                            color="gray", alpha=0.3, linewidth=0)
        elif obs["type"] == "box":
            c = obs["center"]
            d = obs["dimensions"]
            x_min, y_min, z_min = c[0]-d[0]/2, c[1]-d[1]/2, c[2]-d[2]/2
            x_max, y_max, z_max = c[0]+d[0]/2, c[1]+d[1]/2, c[2]+d[2]/2
            vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]]
            ]
            face_collection = Poly3DCollection(faces, facecolor="gray",
                                               edgecolor="darkgray", alpha=0.3, linewidths=1)
            ax.add_collection3d(face_collection)

    # === Animation ===
    ani = FuncAnimation(
        fig,
        update,
        frames=num_steps,
        interval=dt * 1000,
        fargs=(pos, angles,
               axx, axy, axz,
               trail_line, time_text,
               arms, rotor_surfs,
               sphere_surf,
               dt, ax, params,
               box_collection, box_array)
    )

    plt.show()