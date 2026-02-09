import numpy as np
import matplotlib.pyplot as plt
from min_cube_select import min_cube_select
from plot_cube import plot_cube


# %% Largest empty axis-aligned box containing the origin

if __name__ == "__main__":

    for r in range(2, 3):

        np.random.seed(r)

        # Generate random spheres
        N = 8
        P = np.random.rand(N, 3)
        R = np.random.uniform(0.1, 0.2, size=(N,))

        # Fixed point (becomes origin)
        id_start = 3          
        start = P[id_start, :]
        R[id_start] = 0.0

        # Translate coordinate system
        Q = P - start
        R = np.hstack([R, np.zeros(6)])

        # Solve optimization problem
        x_min, x_max, y_min, y_max, z_min, z_max, exitflag = min_cube_select(Q, R)

        if exitflag <= 0:
            print(f"Warning: Optimization failed at iteration {r}")
            continue

        # Visualize
        plot_cube(
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            Q, R,
            plotter=r
        )
