import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# Rotation Matrix
# =========================================================

def rotation_matrix(roll, pitch, yaw):
    """Compute rotation matrix R = Rz * Ry * Rx."""
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

# =========================================================
# Update Function
# =========================================================

def update(i, pos, angles, axx, axy, axz, trail_line, time_text,
           arm1, arm2, arm3, arm4, arm5, arm6,
           disc1, disc2, disc3, disc4, disc5, disc6, 
           dt, ax, ellipsoid_axes=None, ellipsoid_surf=None):
    '''Update 3D plot to animate it.'''

    # Extract current position and attitude
    p = pos[i]
    r, pth, y = angles[i]
    R = rotation_matrix(r, pth, y)

    # Compute body axes endpoints in world frame
    scale = 0.1
    ex = p + scale * R[:, 0]  # x-axis
    ey = p + scale * R[:, 1]  # y-axis
    ez = p + scale * R[:, 2]  # z-axis

    # Rotor positions in hexagonal pattern (body frame)
    arm_len = 0.385
    angular_offset = np.pi / 3

    p1_body = np.array([ arm_len*np.cos(0*angular_offset), arm_len*np.sin(0*angular_offset), 0])
    p2_body = np.array([ arm_len*np.cos(1*angular_offset), arm_len*np.sin(1*angular_offset), 0])
    p3_body = np.array([ arm_len*np.cos(2*angular_offset), arm_len*np.sin(2*angular_offset), 0])
    p4_body = np.array([ arm_len*np.cos(3*angular_offset), arm_len*np.sin(3*angular_offset), 0])
    p5_body = np.array([ arm_len*np.cos(4*angular_offset), arm_len*np.sin(4*angular_offset), 0])
    p6_body = np.array([ arm_len*np.cos(5*angular_offset), arm_len*np.sin(5*angular_offset), 0])

    # Convert rotor locations to world frame
    p1 = p + R @ p1_body
    p2 = p + R @ p2_body
    p3 = p + R @ p3_body
    p4 = p + R @ p4_body
    p5 = p + R @ p5_body
    p6 = p + R @ p6_body

    # Update body axes lines
    axx.set_data([p[0], ex[0]], [p[1], ex[1]])
    axx.set_3d_properties([p[2], ex[2]])

    axy.set_data([p[0], ey[0]], [p[1], ey[1]])
    axy.set_3d_properties([p[2], ey[2]])

    axz.set_data([p[0], ez[0]], [p[1], ez[1]])
    axz.set_3d_properties([p[2], ez[2]])

    # Update arm segments
    arm1.set_data([p[0], p1[0]], [p[1], p1[1]])
    arm1.set_3d_properties([p[2], p1[2]])

    arm2.set_data([p[0], p2[0]], [p[1], p2[1]])
    arm2.set_3d_properties([p[2], p2[2]])

    arm3.set_data([p[0], p3[0]], [p[1], p3[1]])
    arm3.set_3d_properties([p[2], p3[2]])

    arm4.set_data([p[0], p4[0]], [p[1], p4[1]])
    arm4.set_3d_properties([p[2], p4[2]])

    arm5.set_data([p[0], p5[0]], [p[1], p5[1]])
    arm5.set_3d_properties([p[2], p5[2]])

    arm6.set_data([p[0], p6[0]], [p[1], p6[1]])
    arm6.set_3d_properties([p[2], p6[2]])

    # Update rotor markers via internal 3D scatter attribute
    disc1._offsets3d = ([p1[0]], [p1[1]], [p1[2]])
    disc2._offsets3d = ([p2[0]], [p2[1]], [p2[2]])
    disc3._offsets3d = ([p3[0]], [p3[1]], [p3[2]])
    disc4._offsets3d = ([p4[0]], [p4[1]], [p4[2]])
    disc5._offsets3d = ([p5[0]], [p5[1]], [p5[2]])
    disc6._offsets3d = ([p6[0]], [p6[1]], [p6[2]])

    # Update trail line showing the whole trajectory so far
    trail_line.set_data(pos[:i+1, 0], pos[:i+1, 1])
    trail_line.set_3d_properties(pos[:i+1, 2])

    # Update displayed time stamp
    time_text.set_text(f"t = {i*dt:.2f} s")

    # Udpate ellipsoid
    if ellipsoid_axes is not None and ellipsoid_surf is not None:
            a, b, c = ellipsoid_axes
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            x_ell = p[0] + a * np.cos(u) * np.sin(v)
            y_ell = p[1] + b * np.sin(u) * np.sin(v)
            z_ell = p[2] + c * np.cos(v)
            ellipsoid_surf[0].remove() 
            ellipsoid_surf[0] = ax.plot_surface(
                x_ell, y_ell, z_ell,
                color="cyan", alpha=0.2, linewidth=0
            )

    return (axx, axy, axz,
            arm1, arm2, arm3, arm4, arm5, arm6,
            disc1, disc2, disc3, disc4, disc5, disc6,
            trail_line, time_text)

# =========================================================
# Main function
# =========================================================

def animator(pos, angles, params, ellipsoid_axes=None):
    '''Generate and animated 3D plot for STH simulation.'''
    # *** LOAD DATA ***
    obstacles = params.obstacles
    dt = params.dt
    num_steps = pos.shape[0]
    xlim = params.xlim
    ylim = params.ylim
    zlim = params.zlim
    
    # *** SET UP FIGURE AND 3D AXES ***
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])

    # *** DEFINE DRONE, TRAIL LINE AND OBSTACLES ***
    # Drone axes
    axx, = ax.plot([], [], [], color="red", linewidth=2)
    axy, = ax.plot([], [], [], color="green", linewidth=2)
    axz, = ax.plot([], [], [], color="blue", linewidth=2)

    # Drone Arms
    arm1, = ax.plot([], [], [], color="black", linewidth=2)
    arm2, = ax.plot([], [], [], color="black", linewidth=2)
    arm3, = ax.plot([], [], [], color="black", linewidth=2)
    arm4, = ax.plot([], [], [], color="black", linewidth=2)
    arm5, = ax.plot([], [], [], color="black", linewidth=2)
    arm6, = ax.plot([], [], [], color="black", linewidth=2)

    # Rotors (scatter markers)
    disc1 = ax.scatter([], [], [], color="blue", s=80)
    disc2 = ax.scatter([], [], [], color="red", s=80)
    disc3 = ax.scatter([], [], [], color="red", s=80)
    disc4 = ax.scatter([], [], [], color="blue", s=80)
    disc5 = ax.scatter([], [], [], color="blue", s=80)
    disc6 = ax.scatter([], [], [], color="blue", s=80)

    # Trail line
    trail_line, = ax.plot([], [], [], color="red", linewidth=0.8)

    # Obstacles
    if obstacles != []:

        for obs in obstacles:

            if obs["type"] == "sphere":

                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                
                c = obs["center"]
                r = obs["radius"]

                # Create a sphere
                
                x_sphere = c[0] + r * np.cos(u) * np.sin(v)
                y_sphere = c[1] + r * np.sin(u) * np.sin(v)
                z_sphere = c[2] + r * np.cos(v)

                ax.plot_surface(
                    x_sphere, y_sphere, z_sphere,
                    color='gray', alpha=0.3, linewidth=0
                )

            elif obs["type"] == "box":
                c = obs["center"]
                d = obs["dimensions"]
                
                # Calculate box vertices
                x_min, y_min, z_min = c[0] - d[0]/2, c[1] - d[1]/2, c[2] - d[2]/2
                x_max, y_max, z_max = c[0] + d[0]/2, c[1] + d[1]/2, c[2] + d[2]/2
                
                # Define the 8 vertices of the box
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
                
                # Define the 6 faces of the box (each face is defined by 4 vertices)
                faces = [
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                    [vertices[4], vertices[5], vertices[6], vertices[7]]   # Top face
                ]
                
                # Plot each face
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                face_collection = Poly3DCollection(
                    faces, 
                    alpha=0.3, 
                    facecolor='gray', 
                    edgecolor='darkgray',
                    linewidths=1
                )

                ax.add_collection3d(face_collection)

    # Ellipsoid
    ellipsoid_surf = [None]
    if ellipsoid_axes is not None:
        # inizialmente centrato sul primo punto
        a, b, c = ellipsoid_axes
        p0 = pos[0]
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x_ell = p0[0] + a * np.cos(u) * np.sin(v)
        y_ell = p0[1] + b * np.sin(u) * np.sin(v)
        z_ell = p0[2] + c * np.cos(v)
        ellipsoid_surf[0] = ax.plot_surface(x_ell, y_ell, z_ell, color="cyan", alpha=0.2, linewidth=0)


    # *** ANIMATE PLOT ***
    # Animation object
    ani = FuncAnimation(
        fig,
        update,
        frames=num_steps,
        fargs=(pos, angles, axx, axy, axz, trail_line, time_text,
               arm1, arm2, arm3, arm4, arm5, arm6,
               disc1, disc2, disc3, disc4, disc5, disc6,
               dt, ax, ellipsoid_axes, ellipsoid_surf),
        interval=dt*1000  # conversion to milliseconds
    )

    def on_close(event):
        if ani.event_source is not None: 
            ani.event_source.stop()

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()
