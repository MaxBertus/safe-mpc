import numpy as np
import matplotlib.pyplot as plt
from minCubeSelect import min_cube_select, min_cube_select_boxes
from plot_cube import plot_cube

def generate_random_boxes(n_boxes, inner_half=0.5, outer_half=2.0, max_size=0.8, seed=None):
    """
    Generate random axis-aligned boxes outside a small cube and inside a large cube.
    
    Parameters
    ----------
    n_boxes : int
        Number of boxes to generate.
    inner_half : float
        Half side of inner cube to avoid (centered at 0).
    outer_half : float
        Half side of outer cube (bounding cube).
    max_size : float
        Maximum half-dimension of boxes.
    seed : int
        Random seed.
        
    Returns
    -------
    centers : ndarray (n_boxes, 3)
    half_dims : ndarray (n_boxes, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    centers = np.zeros((n_boxes, 3))
    half_dims = np.zeros((n_boxes, 3))
    
    for i in range(n_boxes):
        for j in range(3):  # x, y, z
            # Half-dimension random
            hd = np.random.uniform(0.05, max_size)
            half_dims[i, j] = hd
            
            # Decide whether to place box below or above inner cube
            if np.random.rand() < 0.5:
                # Below inner cube
                min_c = -outer_half + hd
                max_c = -inner_half - hd
            else:
                # Above inner cube
                min_c = inner_half + hd
                max_c = outer_half - hd
            
            centers[i, j] = np.random.uniform(min_c, max_c)
            
    return centers, half_dims

# %% Largest empty axis-aligned box containing the origin

if __name__ == "__main__":

    for r in range(1, 5):

        Q, D = generate_random_boxes(5, seed=r)

        # Solve optimization problem
        x_min, x_max, y_min, y_max, z_min, z_max, exitflag = min_cube_select_boxes(Q, D)

        if exitflag <= 0:
            print(f"Warning: Optimization failed at iteration {r}")
            continue

        # Visualize
        plot_cube(
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            centers=Q,
            half_dims=D,
            plotter=r
        )

# %%
