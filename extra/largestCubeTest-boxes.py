import numpy as np
import matplotlib.pyplot as plt
from minCubeSelect import min_cube_select
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

import numpy as np

def discretize_box_surface(center, half_dims, step):
    """
    Discretize the surface of an axis-aligned box.
    
    Parameters
    ----------
    center : array-like (3,)
        Box center (cx, cy, cz)
    half_dims : array-like (3,)
        Half dimensions (dx, dy, dz)
    step : float
        Maximum grid spacing
        
    Returns
    -------
    points : ndarray (N, 3)
        Surface points including vertices
    """
    cx, cy, cz = center
    dx, dy, dz = half_dims

    # Bounds
    x = np.linspace(cx - dx, cx + dx,
                    int(np.ceil((2*dx)/step)) + 1)
    y = np.linspace(cy - dy, cy + dy,
                    int(np.ceil((2*dy)/step)) + 1)
    z = np.linspace(cz - dz, cz + dz,
                    int(np.ceil((2*dz)/step)) + 1)

    X, Y = np.meshgrid(x, y, indexing='ij')
    X2, Z = np.meshgrid(x, z, indexing='ij')
    Y2, Z2 = np.meshgrid(y, z, indexing='ij')

    # 6 faces
    faces = [
        np.column_stack([X.ravel(), Y.ravel(),
                         np.full(X.size, cz - dz)]),
        np.column_stack([X.ravel(), Y.ravel(),
                         np.full(X.size, cz + dz)]),
        np.column_stack([X2.ravel(),
                         np.full(X2.size, cy - dy),
                         Z.ravel()]),
        np.column_stack([X2.ravel(),
                         np.full(X2.size, cy + dy),
                         Z.ravel()]),
        np.column_stack([np.full(Y2.size, cx - dx),
                         Y2.ravel(),
                         Z2.ravel()]),
        np.column_stack([np.full(Y2.size, cx + dx),
                         Y2.ravel(),
                         Z2.ravel()])
    ]

    points = np.vstack(faces)

    # Remove duplicates (edges/vertices appear multiple times)
    points = np.unique(points, axis=0)

    return points

def discretize_boxes_surfaces(centers, half_dims, step):
    """
    Discretize surfaces of multiple boxes and return a single
    array containing all surface points.
    
    Parameters
    ----------
    centers : ndarray (n_boxes, 3)
    half_dims : ndarray (n_boxes, 3)
    step : float
    
    Returns
    -------
    points : ndarray (N_total, 3)
        All surface points stacked together
    """
    all_points = [
        discretize_box_surface(c, d, step)
        for c, d in zip(centers, half_dims)
    ]
    
    if len(all_points) == 0:
        return np.empty((0, 3))
    
    return np.vstack(all_points)

if __name__ == "__main__":

    counter = 0
    for r in range(1, 10):

        Q, D = generate_random_boxes(5, seed=r)

        points = discretize_boxes_surfaces(Q, D, 1.0)

        # Solve optimization problem
        x_min, x_max, y_min, y_max, z_min, z_max, exitflag, check, _ = min_cube_select(points)

        # if exitflag <= 0:
        #     print(f"Warning: Optimization failed at iteration {r}, so it fails {counter} times. Check={check}")
        #     counter += 1
        #     continue
        # else:
        #     print(f"Iteration {r}: Cube found with volume {(x_max-x_min)*(y_max-y_min)*(z_max-z_min)} and check={check}")

        if not check:
            print(f"Warning: Optimization failed at iteration {r}, so it fails {counter} times. Check={check}")
            counter += 1

        # Visualize
        plot_cube(
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            centers=Q,
            half_dims=D,
            points=points,
            plotter=r
        )
