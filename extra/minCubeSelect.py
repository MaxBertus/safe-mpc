import numpy as np
from scipy.optimize import minimize

def min_cube_select(Q, R=None, goal_point=None):
    """
    Python equivalent of the MATLAB function minCubeSelect.
    
    Parameters
    ----------
    Q : ndarray, shape (N, 3)
        Centers of the spheres.
    R : ndarray, shape (N,)
        Radii of the spheres.
    goal_point : ndarray, shape (3,), optional
        Point that must be included in the box.
    
    Returns
    -------
    xMin, xMax, yMin, yMax, zMin, zMax : float
        Bounds of the optimal box.
    exitflag : int
        Optimization success flag (1 = success, 0 = failure).
    all_outside : bool
        Whether all spheres are outside the box.
    goal_included : bool
        Whether the goal point is included in the box (if provided).
    """

    if R is None:
        R = np.zeros(Q.shape[0])

    # Initial guess adjusted to include goal point if provided
    if goal_point is not None:
        # Start with a box that includes the goal point
        margin = 0.1
        x0 = np.array([
            min(-0.5, goal_point[0] - margin),
            max(0.5, goal_point[0] + margin),
            min(-0.5, goal_point[1] - margin),
            max(0.5, goal_point[1] + margin),
            min(-0.5, goal_point[2] - margin),
            max(0.5, goal_point[2] + margin)
        ])
    else:
        x0 = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])

    # Bounds (origin must be inside)
    bounds = [
        (-2.0, 0.0),   # xMin
        (0.0, 2.0),    # xMax
        (-2.0, 0.0),   # yMin
        (0.0, 2.0),    # yMax
        (-2.0, 0.0),   # zMin
        (0.0, 1.0)     # zMax
    ]

    # Objective: maximize volume -> minimize negative volume
    def objective(x):
        return -box_volume(x)
    
    # Constraints list
    constraints_list = []

    # Nonlinear inequality constraints (c(x) <= 0)
    constraints_list.append({
        "type": "ineq",
        "fun": lambda x: -sphere_box_constraints(x, Q, R)
    })

    # Goal point constraint (if provided)
    if goal_point is not None:
        eps = 1e-6
        constraints_list.append({
            "type": "ineq",
            "fun": lambda x: np.array([
                goal_point[0] - x[0] - eps,  # goal_x >= xMin
                x[1] - goal_point[0] - eps,  # goal_x <= xMax
                goal_point[1] - x[2] - eps,  # goal_y >= yMin
                x[3] - goal_point[1] - eps,  # goal_y <= yMax
                goal_point[2] - x[4] - eps,  # goal_z >= zMin
                x[5] - goal_point[2] - eps   # goal_z <= zMax
            ])
        })

    # Solve optimization
    result = minimize(
        objective,
        x0,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints_list,
        options={"verbose": 0}
    )

    xOpt = result.x
    exitflag = int(result.success)

    eps = 1e-4
    inside = (
        (Q[:,0] - R >= xOpt[0]) &
        (Q[:,0] + R <= xOpt[1]) &
        (Q[:,1] - R >= xOpt[2]) &
        (Q[:,1] + R <= xOpt[3]) &
        (Q[:,2] - R >= xOpt[4]) &
        (Q[:,2] + R <= xOpt[5])
    )

    all_outside = not np.any(inside)

    if goal_point is not None:
        goal_included = (
            (xOpt[0] <= goal_point[0] <= xOpt[1]) and
            (xOpt[2] <= goal_point[1] <= xOpt[3]) and
            (xOpt[4] <= goal_point[2] <= xOpt[5])
        )
    else:
        goal_included = True

    return (
        xOpt[0], xOpt[1],
        xOpt[2], xOpt[3],
        xOpt[4], xOpt[5],
        exitflag, all_outside,
        goal_included
    )

def min_cube_select_boxes(Q, D):
    """
    Maximal axis-aligned box centered at origin avoiding given boxes.
    
    Parameters
    ----------
    Q : ndarray, shape (N,3)
        Centers of the obstacles.
    D : ndarray, shape (N,3)
        Half-dimensions along each axis (dx, dy, dz).
        
    Returns
    -------
    xMin, xMax, yMin, yMax, zMin, zMax : float
        Box bounds.
    exitflag : int
        Success flag.
    """

    # Initial guess (small cube around origin)
    x0 = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])

    # Bounds (we assume [-1,1] for all directions)
    bounds = [
        (-2, 0), (0, 2),  # xMin, xMax
        (-2, 0), (0, 2),  # yMin, yMax
        (-2, 0), (0, 2)   # zMin, zMax
    ]

    # Objective: maximize volume
    def objective(x):
        return -box_volume(x)

    constraints = {"type": "ineq", "fun": lambda x: box_box_constraints(x, Q, D)}

    result = minimize(
        objective,
        x0,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False}
    )

    xOpt = result.x
    exitflag = int(result.success)

    return (
        xOpt[0], xOpt[1],
        xOpt[2], xOpt[3],
        xOpt[4], xOpt[5],
        exitflag
    )

def box_volume(x):
    """
    Compute volume of the box.
    """
    dx = x[1] - x[0]
    dy = x[3] - x[2]
    dz = x[5] - x[4]
    return dx * dy * dz

def sphere_box_constraints(x, Q, R):
    """
    Inequality constraints enforcing that each sphere
    does not intersect the box.
    
    Returns c such that c <= 0.
    """

    xMin, xMax, yMin, yMax, zMin, zMax = x

    N = Q.shape[0]
    c = np.zeros(N)

    for i in range(N):
        cx, cy, cz = Q[i]
        r = R[i]

        dx = max(xMin - cx, 0.0, cx - xMax)
        dy = max(yMin - cy, 0.0, cy - yMax)
        dz = max(zMin - cz, 0.0, cz - zMax)

        dist2 = dx**2 + dy**2 + dz**2

        # inequality: r^2 - dist^2 <= 0
        c[i] = r**2 - dist2 + 1e-3

    return c

def box_box_constraints(x, centers, half_dims, eps=1e-4):
    xMin, xMax, yMin, yMax, zMin, zMax = x
    N = centers.shape[0]
    c = np.zeros(N)

    for i in range(N):
        cx, cy, cz = centers[i]
        dx, dy, dz = half_dims[i]

        dist_x = max(0, (cx - dx) - xMax, xMin - (cx + dx))
        dist_y = max(0, (cy - dy) - yMax, yMin - (cy + dy))
        dist_z = max(0, (cz - dz) - zMax, zMin - (cz + dz))

        c[i] = dist_x**2 + dist_y**2 + dist_z**2 - eps

    return c

