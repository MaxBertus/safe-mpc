import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color='red', alpha=0.1, linestyle='--'):
    """
    Draw a 3D box with given boundaries.
    """
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
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [3, 2, 6, 7]],
        [vertices[j] for j in [0, 3, 7, 4]],
        [vertices[j] for j in [1, 2, 6, 5]]
    ]
    box = Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=1.5, linestyles=linestyle, alpha=alpha)
    ax.add_collection3d(box)


def draw_boxes(ax, centers, half_dims, color='blue', alpha=0.3):
    """
    Draw multiple axis-aligned boxes (obstacles) in 3D.
    
    Parameters
    ----------
    centers : ndarray, shape (N,3)
        Centers of boxes.
    half_dims : ndarray, shape (N,3)
        Half-dimensions of each box along x,y,z.
    """
    for i in range(centers.shape[0]):
        cx, cy, cz = centers[i]
        dx, dy, dz = half_dims[i]
        draw_box(
            ax,
            cx - dx, cx + dx,
            cy - dy, cy + dy,
            cz - dz, cz + dz,
            color=color,
            alpha=alpha,
            linestyle='-'
        )


def plot_cube(x_min, x_max, y_min, y_max, z_min, z_max, 
              Q=None, R=None, centers=None, half_dims=None, points=None, plotter=None):
    """
    Plot the optimal 3D box with obstacles (spheres or boxes) and origin.
    
    Parameters
    ----------
    Q, R : obstacles as spheres (optional)
    centers, half_dims : obstacles as boxes (optional)
    """
    if plotter is None:
        fig = plt.figure()
    else:
        fig = plt.figure(plotter)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Plot spherical obstacles
    if Q is not None and R is not None:
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        for i in range(Q.shape[0]):
            x_sphere = Q[i, 0] + R[i] * np.cos(u) * np.sin(v)
            y_sphere = Q[i, 1] + R[i] * np.sin(u) * np.sin(v)
            z_sphere = Q[i, 2] + R[i] * np.cos(v)
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.3, linewidth=0)

        ax.scatter(Q[:,0], Q[:,1], Q[:,2], s=10, c='b')

    # Plot box obstacles
    if centers is not None and half_dims is not None:
        draw_boxes(ax, centers, half_dims, color='blue', alpha=0.3)
        if points is not None:
            ax.scatter(points[:,0], points[:,1], points[:,2], s=10, c='cyan')

    # Plot UAV at origin
    ax.scatter(0, 0, 0, s=20, c='black', edgecolors='white')
    ax.text(0, 0, -0.12, "UAV", color='black', fontsize=8, ha='center', fontweight='bold')

    # Draw UAV as a small sphere
    uav_radius = 0.5  # raggio della sfera UAV
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = uav_radius * np.cos(u) * np.sin(v)
    y_sphere = uav_radius * np.sin(u) * np.sin(v)
    z_sphere = uav_radius * np.cos(v)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.5)

    # Draw optimal box
    draw_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color='red', alpha=0.1, linestyle='--')

    # Compute and display volume
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    ax.text(1.5, 1.5, 1.5, f"Volume: {volume:.3f}", color='red', fontsize=8, fontweight='bold')

    # Axis limits, labels, and grid
    ax.set_xlim([-2,2]); ax.set_ylim([-2,2]); ax.set_zlim([-2,2])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.plot([0,0],[0,0],[-2,2],'k--', linewidth=0.5)
    ax.plot([0,0],[-2,2],[0,0],'k--', linewidth=0.5)
    ax.plot([-2,2],[0,0],[0,0],'k--', linewidth=0.5)
    ax.view_init(elev=20, azim=30)
    ax.grid(True)
    plt.show()
