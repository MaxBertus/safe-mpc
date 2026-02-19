import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from minCubeSelect import min_cube_select, min_cube_select_fast
from plot_cube import plot_cube

def generate_random_boxes(n_boxes, inner_half=0.6, outer_half=2.0, max_size=0.8,
                          seed=None, goal_point=None, max_attempts=1000):
    """
    Generate random axis-aligned boxes outside a small cube and inside a large cube,
    with no mutual intersections and without containing the goal point.

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
    goal_point : ndarray, shape (3,), optional
        Goal point that must not be inside any box.
    max_attempts : int
        Maximum placement attempts per box before giving up.

    Returns
    -------
    centers : ndarray (n_boxes, 3)
    half_dims : ndarray (n_boxes, 3)

    Raises
    ------
    RuntimeError
        If a box cannot be placed without intersections within max_attempts tries.
    """
    if seed is not None:
        np.random.seed(seed)

    centers = np.zeros((n_boxes, 3))
    half_dims = np.zeros((n_boxes, 3))

    def overlaps_any(c, hd, placed_count):
        """Return True if box (c, hd) intersects any already-placed box."""
        if placed_count == 0:
            return False
        prev_c  = centers[:placed_count]
        prev_hd = half_dims[:placed_count]
        separations = np.abs(c - prev_c) - (hd + prev_hd)  # (placed_count, 3)
        no_overlap = np.any(separations >= 0.0, axis=1)
        return not np.all(no_overlap)

    def contains_goal(c, hd):
        """Return True if goal_point lies inside box (c, hd)."""
        if goal_point is None:
            return False
        return np.all(
            (goal_point >= c - hd) &
            (goal_point <= c + hd)
        )

    def sample_box():
        """Sample a random box that respects the inner/outer cube constraints."""
        hd = np.random.uniform(0.05, max_size, size=3)
        c  = np.empty(3)
        for j in range(3):
            if np.random.rand() < 0.5:
                lo, hi = -outer_half + hd[j], -inner_half - hd[j]
            else:
                lo, hi =  inner_half + hd[j],  outer_half - hd[j]
            c[j] = np.random.uniform(lo, hi)
        return c, hd

    for i in range(n_boxes):
        for attempt in range(max_attempts):
            c, hd = sample_box()
            if not overlaps_any(c, hd, i) and not contains_goal(c, hd):
                centers[i]   = c
                half_dims[i] = hd
                break
        else:
            raise RuntimeError(
                f"Could not place box {i} without intersections after "
                f"{max_attempts} attempts. Try fewer boxes or a larger outer_half."
            )

    return centers, half_dims

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

        goal_point = np.array([0.0, -1.0, -1.0])
        
        # Generate boxes that don't contain the goal point
        Q, D = generate_random_boxes(5, seed=r, goal_point=goal_point)

        points = discretize_boxes_surfaces(Q, D, 1.0)

        # Solve optimization problem
        x_min, x_max, y_min, y_max, z_min, z_max, exitflag, check, goal_incl = min_cube_select_fast(points, R=np.full(points.shape[0],0.01),goal_point=goal_point)

        # if exitflag <= 0:
        #     print(f"Warning: Optimization failed at iteration {r}, so it fails {counter} times. Check={check}")
        #     counter += 1
        #     continue
        # else:
        #     print(f"Iteration {r}: Cube found with volume {(x_max-x_min)*(y_max-y_min)*(z_max-z_min)} and check={check}")

        if not check:
            print(f"Warning: Optimization failed at iteration {r}, so it fails {counter+1} times. Check={check} and goal_incl={goal_incl}")
            counter += 1
        else:
            print(f"Iteration {r}: Cube found with volume {(x_max-x_min)*(y_max-y_min)*(z_max-z_min)} and goal included={goal_incl}")

        # Visualize
        plot_cube(
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            centers=Q,
            half_dims=D,
            points=points,
            plotter=r,
            goal_point=goal_point
        )
