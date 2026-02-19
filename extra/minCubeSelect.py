import numpy as np
from scipy.optimize import minimize

def min_cube_select(Q=None, R=None, goal_point=None):
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
        (0.0, 2.0)     # zMax
    ]

    # Objective: maximize volume -> minimize negative volume
    def objective(x):
        return -box_volume(x)
    
    # Constraints list
    constraints_list = []

    # Nonlinear inequality constraints (c(x) >= 0)
    constraints_list.append({
        "type": "ineq",
        "fun": lambda x: sphere_box_constraints(x, Q, R)
    })

    # Nonlinear inequality constraints (c(x) >= 0)
    constraints_list.append({
        "type": "ineq",
        "fun": lambda x: drone_occupancy(x)
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

def smooth_max(a, b, c, alpha=5.0):
    """Log-sum-exp smooth approximation of max(a, b, c)."""
    vals = np.array([a, b, c]) * alpha
    return np.log(np.sum(np.exp(vals - np.max(vals)))) / alpha + np.max(vals) / alpha

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
        c[i] = -r**2 + dist2 - 1e-3

    return c

def drone_occupancy(x):
    """
    Inequality constraints ensuring the box can contain a sphere
    of radius 0.5 centered at the origin.
    
    Since xMin, yMin, zMin are always negative (from bounds),
    we ensure they are <= -0.5, and xMax, yMax, zMax are >= 0.5.
    
    Returns c such that c >= 0 (for scipy 'ineq' constraint).
    """
    xMin, xMax, yMin, yMax, zMin, zMax = x
    
    drone_radius = 0.5
    eps = 1e-6  # small tolerance for numerical stability
    
    # Since mins are negative: xMin <= -drone_radius means -xMin >= drone_radius
    # Since maxs are positive: xMax >= drone_radius
    c = np.array([
        -xMin - drone_radius + eps,  # -xMin >= 0.5 (since xMin < 0)
        xMax - drone_radius + eps,   # xMax >= 0.5
        -yMin - drone_radius + eps,  # -yMin >= 0.5 (since yMin < 0)
        yMax - drone_radius + eps,   # yMax >= 0.5
        -zMin - drone_radius + eps,  # -zMin >= 0.5 (since zMin < 0)
        zMax - drone_radius + eps    # zMax >= 0.5
    ])
    
    return c

def min_cube_select_fast(Q=None, R=None, goal_point=None, drone_radius=0.5):
    """
    Fast closed-form version of min_cube_select.
    Builds the largest axis-aligned box centered at the origin
    that avoids all spheres, using a greedy half-space intersection approach.

    Strategy:
        - Start from the maximum allowed box [-2,2]^3
        - For each sphere that intersects the current box,
          shrink the nearest face to push the sphere outside
        - Iterate until no sphere intersects or no improvement is possible
    """
    if R is None:
        R = np.zeros(Q.shape[0])

    # Hard limits from bounds
    LIMIT = 2.0

    # Initialize box to maximum extent
    box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)

    # Enforce drone occupancy from the start
    box[0] = min(box[0], -drone_radius)
    box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius)
    box[3] = max(box[3],  drone_radius)
    box[4] = min(box[4], -drone_radius)
    box[5] = max(box[5],  drone_radius)

    # Enforce goal point inclusion
    if goal_point is not None:
        box[0] = min(box[0], goal_point[0])
        box[1] = max(box[1], goal_point[0])
        box[2] = min(box[2], goal_point[1])
        box[3] = max(box[3], goal_point[1])
        box[4] = min(box[4], goal_point[2])
        box[5] = max(box[5], goal_point[2])

    max_iter = 50
    tol = 1e-5

    for _ in range(max_iter):
        moved = False

        # Vectorized: find spheres intersecting the box
        intersecting = _spheres_intersect_box(Q, R, box, tol)
        if not np.any(intersecting):
            break

        Qi = Q[intersecting]
        Ri = R[intersecting]

        # For each intersecting sphere, find which face to push
        # and by how much — pick the face that maximizes volume retention
        box, moved = _push_faces(box, Qi, Ri, drone_radius, goal_point)

        if not moved:
            break

    xOpt = box
    exitflag = 1 if not np.any(_spheres_intersect_box(Q, R, box, tol)) else 0

    inside = (
        (Q[:, 0] - R >= xOpt[0]) & (Q[:, 0] + R <= xOpt[1]) &
        (Q[:, 1] - R >= xOpt[2]) & (Q[:, 1] + R <= xOpt[3]) &
        (Q[:, 2] - R >= xOpt[4]) & (Q[:, 2] + R <= xOpt[5])
    )
    all_outside = not np.any(inside)

    if goal_point is not None:
        goal_included = (
            xOpt[0] <= goal_point[0] <= xOpt[1] and
            xOpt[2] <= goal_point[1] <= xOpt[3] and
            xOpt[4] <= goal_point[2] <= xOpt[5]
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


def _spheres_intersect_box(Q, R, box, tol=1e-5):
    """
    Vectorized check: returns boolean mask of spheres intersecting the box.
    A sphere intersects if its closest point to the box is within radius R.
    """
    xMin, xMax, yMin, yMax, zMin, zMax = box

    # Closest point on box to each sphere center (vectorized)
    cx = np.clip(Q[:, 0], xMin, xMax)
    cy = np.clip(Q[:, 1], yMin, yMax)
    cz = np.clip(Q[:, 2], zMin, zMax)

    dist2 = (Q[:, 0] - cx)**2 + (Q[:, 1] - cy)**2 + (Q[:, 2] - cz)**2
    return dist2 < (R**2 - tol)


def _push_faces(box, Qi, Ri, drone_radius, goal_point):
    """
    For each intersecting sphere, compute the volume-preserving best face push.
    Returns updated box and whether any face was actually moved.
    """
    xMin, xMax, yMin, yMax, zMin, zMax = box
    moved = False

    for i in range(len(Qi)):
        cx, cy, cz = Qi[i]
        r = Ri[i]

        # Volume of box before push
        vol_before = (xMax - xMin) * (yMax - yMin) * (zMax - zMin)

        # Candidate new face positions to exclude this sphere
        # For each face, compute the new position and resulting volume
        candidates = []

        # Push xMin right (shrink from left): new xMin = cx + r + eps
        new_xMin = cx + r + 1e-4
        if new_xMin <= 0 and new_xMin >= -2.0:  # must stay in bounds
            vol = (xMax - new_xMin) * (yMax - yMin) * (zMax - zMin)
            candidates.append(('xMin', new_xMin, vol))

        # Push xMax left (shrink from right): new xMax = cx - r - eps
        new_xMax = cx - r - 1e-4
        if new_xMax >= 0 and new_xMax <= 2.0:
            vol = (new_xMax - xMin) * (yMax - yMin) * (zMax - zMin)
            candidates.append(('xMax', new_xMax, vol))

        # Push yMin up
        new_yMin = cy + r + 1e-4
        if new_yMin <= 0 and new_yMin >= -2.0:
            vol = (xMax - xMin) * (yMax - new_yMin) * (zMax - zMin)
            candidates.append(('yMin', new_yMin, vol))

        # Push yMax down
        new_yMax = cy - r - 1e-4
        if new_yMax >= 0 and new_yMax <= 2.0:
            vol = (xMax - xMin) * (new_yMax - yMin) * (zMax - zMin)
            candidates.append(('yMax', new_yMax, vol))

        # Push zMin up
        new_zMin = cz + r + 1e-4
        if new_zMin <= 0 and new_zMin >= -2.0:
            vol = (xMax - xMin) * (yMax - yMin) * (zMax - new_zMin)
            candidates.append(('zMin', new_zMin, vol))

        # Push zMax down
        new_zMax = cz - r - 1e-4
        if new_zMax >= 0 and new_zMax <= 2.0:
            vol = (xMax - xMin) * (yMax - yMin) * (new_zMax - zMin)
            candidates.append(('zMax', new_zMax, vol))

        if not candidates:
            continue

        # Pick the face move that retains the most volume
        best = max(candidates, key=lambda c: c[2])

        # Apply only if it doesn't violate drone occupancy or goal constraints
        face, val, vol = best
        new_box = np.array([xMin, xMax, yMin, yMax, zMin, zMax])

        face_idx = {'xMin': 0, 'xMax': 1, 'yMin': 2, 'yMax': 3, 'zMin': 4, 'zMax': 5}
        new_box[face_idx[face]] = val

        if not _violates_constraints(new_box, drone_radius, goal_point):
            xMin, xMax, yMin, yMax, zMin, zMax = new_box
            box = new_box
            moved = True

    return box, moved


def _violates_constraints(box, drone_radius, goal_point):
    """Check drone occupancy and goal point constraints."""
    xMin, xMax, yMin, yMax, zMin, zMax = box

    # Drone occupancy: box must contain sphere of radius drone_radius at origin
    if (-xMin < drone_radius or xMax < drone_radius or
            -yMin < drone_radius or yMax < drone_radius or
            -zMin < drone_radius or zMax < drone_radius):
        return True

    # Goal point must be inside box
    if goal_point is not None:
        if not (xMin <= goal_point[0] <= xMax and
                yMin <= goal_point[1] <= yMax and
                zMin <= goal_point[2] <= zMax):
            return True

    return False