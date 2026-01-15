import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------- Rotation Matrix --------------------

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
           arm1, arm2, arm3, arm4, arm5, arm6,
           disc1, disc2, disc3, disc4, disc5, disc6, 
           dt):

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

    return (axx, axy, axz,
            arm1, arm2, arm3, arm4, arm5, arm6,
            disc1, disc2, disc3, disc4, disc5, disc6,
            trail_line, time_text)

# ------------------------------------------------------------
# ------------------------------- Main ------------------------
# ------------------------------------------------------------

def animator(pos, angles, dt=0.05, num_steps=200):

    # Set up figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

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

    # Animation object
    ani = FuncAnimation(
        fig,
        update,
        frames=num_steps,
        fargs=(pos, angles, axx, axy, axz, trail_line, time_text,
               arm1, arm2, arm3, arm4, arm5, arm6,
               disc1, disc2, disc3, disc4, disc5, disc6,
               dt),
        interval=dt*1000  # conversion to milliseconds
    )

    plt.show()
