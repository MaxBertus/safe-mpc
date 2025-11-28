import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------- Rotation Matrix --------------------

def rotation_matrix(roll, pitch, yaw):
    """Compute rotation matrix R = Rz * Ry * Rx."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),  np.sin(yaw)

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


# -------------------- Synthetic Trajectory --------------------

def synthetic_trajectory(num_steps):
    """Generate a smooth helical trajectory + oscillatory attitude."""
    t = np.linspace(0, 4*np.pi, num_steps)

    # Helical 3D trajectory
    pos = np.vstack([
        0.5 + 0.3*np.cos(t),
        0.5 + 0.3*np.sin(t),
        0.2 + 0.1*t/(4*np.pi)
    ]).T

    # Smooth roll/pitch oscillations and yaw ramp
    # roll  = 0.2*np.sin(t)
    # pitch = 0.2*np.cos(t)
    # yaw   = t * 0.3
    
    roll = np.zeros(t.shape)
    pitch = np.zeros(t.shape)
    yaw = t 

    angles = np.vstack([roll, pitch, yaw]).T
    return pos, angles


# -------------------- Update Function --------------------

def update(i, pos, angles, axx, axy, axz, trail_line, time_text,
           x0, y0, z0, ellipsoid_container):

    # Extract current position and attitude
    p = pos[i]
    r, pth, y = angles[i]
    R = rotation_matrix(r, pth, y)

    # Compute body axes endpoints in world frame
    scale = 0.1
    ex = p + scale * R[:, 0]
    ey = p + scale * R[:, 1]
    ez = p + scale * R[:, 2]

    # Update body axes lines
    axx.set_data([p[0], ex[0]], [p[1], ex[1]])
    axx.set_3d_properties([p[2], ex[2]])

    axy.set_data([p[0], ey[0]], [p[1], ey[1]])
    axy.set_3d_properties([p[2], ey[2]])

    axz.set_data([p[0], ez[0]], [p[1], ez[1]])
    axz.set_3d_properties([p[2], ez[2]])

    # Remove previous surface
    ellipsoid_container[0].remove()

    # Rotate and translate ellipsoid mesh
    XYZ = np.stack([x0, y0, z0], axis=-1)         # shape (Nu, Nv, 3)
    XYZ_rot = XYZ @ R.T                          # rotation
    XYZ_rot += p                                 # translation

    xw = XYZ_rot[..., 0]
    yw = XYZ_rot[..., 1]
    zw = XYZ_rot[..., 2]

    # Draw new ellipsoid and store reference
    ellipsoid_container[0] = ax.plot_surface(
        xw, yw, zw, color="red", alpha=0.6, linewidth=0
    )

    # Update trail line showing the whole trajectory so far
    trail_line.set_data(pos[:i+1, 0], pos[:i+1, 1])
    trail_line.set_3d_properties(pos[:i+1, 2])

    # Update displayed time stamp
    time_text.set_text(f"t = {i*dt:.2f} s")

    return (axx, axy, axz, trail_line, time_text, ellipsoid_container[0])


# ------------------------------------------------------------
# ------------------------------- Main ------------------------
# ------------------------------------------------------------

if __name__ == "__main__":

    num_steps = 200

    # Generate trajectory and attitude data
    pos, angles = synthetic_trajectory(num_steps)

    # Set up figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    dt = 0.05
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Drone axes
    axx, = ax.plot([], [], [], color="red",   linewidth=2)
    axy, = ax.plot([], [], [], color="green", linewidth=2)
    axz, = ax.plot([], [], [], color="blue",  linewidth=2)

    # Ellipsoid Body (in body frame)
    a, b, c = 0.07, 0.04, 0.02
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)

    x0 = a * np.outer(np.cos(u), np.sin(v))
    y0 = b * np.outer(np.sin(u), np.sin(v))
    z0 = c * np.outer(np.ones_like(u), np.cos(v))

    # Initial drawing — store in a list so that update() can replace it
    ellipsoid_container = [ax.plot_surface(x0, y0, z0, color="red", alpha=0.6)]

    # Trail line
    trail_line, = ax.plot([], [], [], color="red", linewidth=0.8)

    # Animation object
    ani = FuncAnimation(
        fig,
        update,
        frames=num_steps,
        fargs=(pos, angles, axx, axy, axz,
               trail_line, time_text,
               x0, y0, z0, ellipsoid_container),
        interval=dt*1000
    )

    plt.show()
