import numpy as np
import matplotlib.pyplot as plt
from minCubeSelect import min_cube_select
from plot_cube import plot_cube


def generate_random_spheres_outside_box(N, inner_half=0.5, outer_half=2.0, radius_range=(0.4,0.8), seed=None):
    """
    Generate N random spheres outside a cube of half-side inner_half.
    
    Parameters
    ----------
    N : int
        Number of spheres
    inner_half : float
        Half-side of inner box to avoid
    outer_half : float
        Half-side of bounding cube
    radius_range : tuple
        Min and max radius
    seed : int
        Random seed
    
    Returns
    -------
    centers : ndarray (N,3)
    radii : ndarray (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.zeros((N,3))
    radii = np.random.uniform(radius_range[0], radius_range[1], size=N)
    
    for i in range(N):
        for j in range(3):  # x,y,z
            r = radii[i]
            # Decide randomly whether the coordinate is below or above the box
            if np.random.rand() < 0.5:
                min_c = -outer_half + r
                max_c = -inner_half - r
            else:
                min_c = inner_half + r
                max_c = outer_half - r
            centers[i,j] = np.random.uniform(min_c, max_c)
    
    return centers, radii


# %% Largest empty axis-aligned box containing the origin

if __name__ == "__main__":

    counter = 0
    for r in range(1, 2):

        np.random.seed(r)

        # Fixed parameters
        N = 8

        # Generate spheres outside the minimal box
        Q, R = generate_random_spheres_outside_box(N, seed=r)

        R = np.zeros(Q.shape[0])

        # Add the origin as a fixed "sphere" of radius 0
        Q = np.vstack([Q, np.zeros((1,3))])
        R = np.hstack([R, 0.0])

        x_min, x_max, y_min, y_max, z_min, z_max, exitflag, check, _ = min_cube_select(Q, R)

        if exitflag <= 0:
            print(f"Warning: Optimization failed at iteration {r}, so it fails {counter} times. Check={check}")
            counter += 1
            continue
        else:
            print(f"Iteration {r}: Cube found with volume {(x_max-x_min)*(y_max-y_min)*(z_max-z_min)} and check={check}")

        # Visualize
        plot_cube(
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            Q=Q, R=R,
            plotter=r
        )
