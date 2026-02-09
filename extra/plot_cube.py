import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_box(ax, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Draw a 3D box with given boundaries.
    """

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

    # Define the 6 faces
    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],  # Bottom
        [vertices[j] for j in [4, 5, 6, 7]],  # Top
        [vertices[j] for j in [0, 1, 5, 4]],  # Front
        [vertices[j] for j in [3, 2, 6, 7]],  # Back
        [vertices[j] for j in [0, 3, 7, 4]],  # Left
        [vertices[j] for j in [1, 2, 6, 5]]   # Right
    ]

    box = Poly3DCollection(
        faces,
        facecolors='red',
        edgecolors='red',
        linewidths=1.5,
        linestyles='--',
        alpha=0.1
    )

    ax.add_collection3d(box)


def plot_cube(x_min, x_max, y_min, y_max, z_min, z_max, Q, R, plotter=None):
    """
    Plot the optimal 3D box with obstacles and goal point.
    """

    if plotter is None:
        fig = plt.figure()
    else:
        fig = plt.figure(plotter)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Plot obstacle points
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=10, c='b')

    # Draw spheres around each obstacle
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]

    for i in range(Q.shape[0]):
        x_sphere = Q[i, 0] + R[i] * np.cos(u) * np.sin(v)
        y_sphere = Q[i, 1] + R[i] * np.sin(u) * np.sin(v)
        z_sphere = Q[i, 2] + R[i] * np.cos(v)

        ax.plot_surface(
            x_sphere, y_sphere, z_sphere,
            color='blue', alpha=0.3, linewidth=0
        )

    # Plot UAV at origin
    ax.scatter(0, 0, 0, s=10, c='black', edgecolors='white')
    ax.text(0, 0, -0.12, "UAV", color='black',
            fontsize=8, ha='center', fontweight='bold')

    # Highlight boundary obstacles
    idx = np.isclose(Q[:, 0], x_min) | np.isclose(Q[:, 0], x_max)
    idy = np.isclose(Q[:, 1], y_min) | np.isclose(Q[:, 1], y_max)
    idz = np.isclose(Q[:, 2], z_min) | np.isclose(Q[:, 2], z_max)
    id_boundary = idx | idy | idz

    ax.scatter(
        Q[id_boundary, 0],
        Q[id_boundary, 1],
        Q[id_boundary, 2],
        s=20, c='red'
    )

    # Draw optimal box
    draw_box(ax, x_min, x_max, y_min, y_max, z_min, z_max)

    # Compute and display volume
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    ax.text(
        1.5, 1.5, 1.5,
        f"Volume: {volume:.3f}",
        color='red',
        fontsize=8,
        fontweight='bold'
    )

    # Axis limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw coordinate axes
    ax.plot([0, 0], [0, 0], [-1, 1], 'k--', linewidth=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k--', linewidth=0.5)
    ax.plot([-1, 1], [0, 0], [0, 0], 'k--', linewidth=0.5)

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    ax.grid(True)

    plt.show()
