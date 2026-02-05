import numpy as np
from scipy.optimize import minimize

def min_cube_select(Q, R):
    """
    Python equivalent of the MATLAB function minCubeSelect.
    
    Parameters
    ----------
    Q : ndarray, shape (N, 3)
        Centers of the spheres.
    R : ndarray, shape (N,)
        Radii of the spheres.
    
    Returns
    -------
    xMin, xMax, yMin, yMax, zMin, zMax : float
        Bounds of the optimal box.
    exitflag : int
        Optimization success flag (1 = success, 0 = failure).
    """

    # Initial guess (small box)
    x0 = np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1])

    # Bounds (origin must be inside)
    bounds = [
        (-1.0, 0.0),   # xMin
        (0.0, 1.0),    # xMax
        (-1.0, 0.0),   # yMin
        (0.0, 1.0),    # yMax
        (-1.0, 0.0),   # zMin
        (0.0, 1.0)     # zMax
    ]

    # Objective: maximize volume -> minimize negative volume
    def objective(x):
        return -box_volume(x)

    # Nonlinear inequality constraints (c(x) <= 0)
    constraints = {
        "type": "ineq",
        "fun": lambda x: -sphere_box_constraints(x, Q, R)
    }

    # Solve optimization
    result = minimize(
        objective,
        x0,
        method="SLSQP",
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
        c[i] = r**2 - dist2

    return c
