import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# =========================================================
# Rotation utilities
# =========================================================

def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def axis_angle_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

def rotor_disc(center, normal, radius=0.08, n_points=40):
    normal = normal / np.linalg.norm(normal)
    t1 = np.cross(normal, [0,0,1] if abs(normal[2])<0.9 else [0,1,0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    theta = np.linspace(0, 2*np.pi, n_points)
    circle_pts = np.array([center + radius*(np.cos(t)*t1 + np.sin(t)*t2) for t in theta])
    x = np.hstack([center[0], circle_pts[:,0]])
    y = np.hstack([center[1], circle_pts[:,1]])
    z = np.hstack([center[2], circle_pts[:,2]])
    triangles = [[0, i, i+1] for i in range(1, n_points)] + [[0, n_points, 1]]
    return x, y, z, triangles

def build_box_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    return [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
    ]

# =========================================================
# Update function
# =========================================================

def update(i, pos, angles, axx, axy, axz,
           trail_line, time_text,
           arms, rotor_surfs,
           sphere_surf,
           dt, ax, params,
           box_collection, box_array,
           save_frames=False, frames_dir=None, save_every=10):

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
    rotor_centers, arm_dirs_body = [], []
    for k, ang in enumerate(angles_arm):
        p_body = np.array([arm_len*np.cos(ang), arm_len*np.sin(ang), 0])
        pw = p + R @ p_body
        arms[k].set_data([p[0], pw[0]], [p[1], pw[1]])
        arms[k].set_3d_properties([p[2], pw[2]])
        rotor_centers.append(pw)
        arm_dirs_body.append(p_body)

    for surf in rotor_surfs: surf.remove()
    rotor_surfs.clear()
    colors = ["cornflowerblue", "red"]*3
    for k, (pc, arm_b) in enumerate(zip(rotor_centers, arm_dirs_body)):
        arm_dir_world = R @ (arm_b / np.linalg.norm(arm_b))
        sign = (-1)**k
        R_tilt = axis_angle_rotation(arm_dir_world, sign * params.alpha_tilt)
        rotor_normal = R_tilt @ R[:,2]
        x, y, z, tris = rotor_disc(pc, rotor_normal, radius=params.propRad)
        surf = ax.plot_trisurf(x, y, z, triangles=tris, color=colors[k], alpha=0.5, linewidth=0)
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
        sphere_surf[0] = ax.plot_surface(xs, ys, zs, color="cyan", alpha=0.2, linewidth=0)

    # Animated bounding box
    if box_collection is not None and box_array is not None:
        box_params_i = box_array[i]
        faces = build_box_faces(*box_params_i)
        box_collection.set_verts(faces)

    # Trail
    trail_line.set_data(pos[:i+1, 0], pos[:i+1, 1])
    trail_line.set_3d_properties(pos[:i+1, 2])
    time_text.set_text(f"t = {i*dt:.2f} s")

    # Save frame
    if save_frames and i % save_every == 0 and frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)
        frame_file = os.path.join(frames_dir, f"frame_{i:05d}.pdf")
        plt.savefig(frame_file, format="pdf", bbox_inches="tight")

    return []

# =========================================================
# Main animator with frame saving and lateral view
# =========================================================

def animator(pos, angles, box, params,
             save_frames=True, save_every=10, save_video=True):
    dt = params.dt
    num_steps = pos.shape[0]

    # Frame saving folder
    frames_dir = os.path.join(getattr(params, "plots_dir","."), "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # --- Larger figure (~3x the default 6x4) ---
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(params.xlim)
    ax.set_ylim(params.ylim)
    ax.set_zlim(params.zlim)
    x_range = params.xlim[1]-params.xlim[0]
    y_range = params.ylim[1]-params.ylim[0]
    z_range = params.zlim[1]-params.zlim[0]
    ax.set_box_aspect([x_range, y_range, z_range])

    # Lateral view (X→YZ)
    ax.view_init(elev=0, azim=-90)

    # --- More ticks (one every 0.5 m, adjust to taste) ---
    tick_spacing = 1.0
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # --- Tick labels and axis labels ---
    LABEL_SIZE  = 20   # axis-name font size  (≈2× the previous 6)
    TICK_SIZE   = 18   # tick number font size (≈2× the previous 5)

    ax.tick_params(axis="x", labelsize=TICK_SIZE, pad=2)
    ax.tick_params(axis="z", labelsize=TICK_SIZE, pad=10)   # shift z labels leftward

    ax.set_xlabel("x [m]", fontsize=LABEL_SIZE, labelpad=6)
    ax.set_ylabel("",       fontsize=LABEL_SIZE)
    ax.set_zlabel("z [m]", fontsize=LABEL_SIZE, labelpad=16)

    # Y axis: show grid lines but hide tick numbers
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.tick_params(axis="y", labelsize=0, length=0)         # hide y tick labels
    ax.yaxis.set_ticklabels([])                             # extra safety
    ax.grid(True)                                           # enable grid (uses all axes)

    fig.tight_layout()

    # Global reference frame
    L = 0.4
    ax.plot([0,L],[0,0],[0,0],color="r",linewidth=2)
    ax.plot([0,0],[0,L],[0,0],color="g",linewidth=2)
    ax.plot([0,0],[0,0],[0,L],color="b",linewidth=2)

    # Drone frame
    axx, = ax.plot([],[],[],color="r",linewidth=2)
    axy, = ax.plot([],[],[],color="g",linewidth=2)
    axz, = ax.plot([],[],[],color="b",linewidth=2)
    arms = [ax.plot([],[],[],color="k",linewidth=2)[0] for _ in range(6)]
    rotor_surfs = []
    sphere_surf = [None]
    trail_line, = ax.plot([],[],[],color="cornflowerblue",linewidth=0.8)

    # Reference position marker
    if hasattr(params, "x_ref"):
        xr = params.x_ref[:3]
        ax.plot([xr[0]], [xr[1]], [xr[2]], "rx", markersize=12, markeredgewidth=2.5)

    # --- Smaller time text ---
    time_text = fig.text(0.52, 0.12, "", ha="center", fontsize=16)

    # Bounding box
    box_collection, box_array = None, None
    if box is not None:
        box_np = np.asarray(box)
        box_array = np.tile(box_np,(num_steps,1)) if box_np.ndim==1 else box_np
        faces = build_box_faces(*box_array[0])
        box_collection = Poly3DCollection(faces, facecolor=(1,0,0,0.15),
                                          edgecolor=(1,0,0,0.6), linewidth=0.8)
        ax.add_collection3d(box_collection)

    # Obstacles
    obstacles = getattr(params,"obstacles",[])
    for obs in obstacles:
        if obs["type"]=="sphere":
            u,v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
            c,r = obs["center"], obs["radius"]
            x_sphere = c[0]+r*np.cos(u)*np.sin(v)
            y_sphere = c[1]+r*np.sin(u)*np.sin(v)
            z_sphere = c[2]+r*np.cos(v)
            ax.plot_surface(x_sphere,y_sphere,z_sphere,color="gray",alpha=0.3,linewidth=0)
        elif obs["type"]=="box":
            c,d = obs["center"], obs["dimensions"]
            x_min,y_min,z_min = c[0]-d[0]/2, c[1]-d[1]/2, c[2]-d[2]/2
            x_max,y_max,z_max = c[0]+d[0]/2, c[1]+d[1]/2, c[2]+d[2]/2
            vertices = np.array([[x_min,y_min,z_min],[x_max,y_min,z_min],[x_max,y_max,z_min],[x_min,y_max,z_min],
                                 [x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_max,z_max],[x_min,y_max,z_max]])
            faces = [[vertices[0],vertices[1],vertices[5],vertices[4]],
                     [vertices[2],vertices[3],vertices[7],vertices[6]],
                     [vertices[0],vertices[3],vertices[7],vertices[4]],
                     [vertices[1],vertices[2],vertices[6],vertices[5]],
                     [vertices[0],vertices[1],vertices[2],vertices[3]],
                     [vertices[4],vertices[5],vertices[6],vertices[7]]]
            ax.add_collection3d(Poly3DCollection(faces,facecolor="gray",edgecolor="darkgray",alpha=0.3,linewidths=1))

    # Animation
    ani = FuncAnimation(fig, update, frames=num_steps, interval=dt*1000,
                        fargs=(pos, angles, axx, axy, axz,
                               trail_line, time_text,
                               arms, rotor_surfs, sphere_surf,
                               dt, ax, params,
                               box_collection, box_array,
                               save_frames, frames_dir, save_every))

    # Save video — tight crop via savefig_kwargs
    if save_video:
        video_path = os.path.join(getattr(params,"plots_dir","."),"animation.mp4")
        writer = FFMpegWriter(fps=int(1/dt))
        ani.save(video_path, writer=writer)
        print("Video saved at:", video_path)

    plt.show()