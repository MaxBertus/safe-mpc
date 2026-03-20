import numpy as np
import casadi as ca
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross, fmin, fmax
from acados_template import (
    AcadosModel, AcadosOcp, AcadosOcpSolver,
    AcadosSim, AcadosSimSolver,
)
from utils.plotter import plotter
import os
import pickle
from tqdm import tqdm


# =============================================================================
# PARAMETERS
# =============================================================================

class Params:
    """Hard-coded configuration for the STH MPC simulation.

    Groups all physical, control, simulation, and environment parameters
    into a single object for easy access throughout the pipeline.
    """

    def __init__(self) -> None:

        # --- Robot physical parameters ---
        self.mass = 3.5                          # Total mass [kg]
        self.J = np.diag([0.155, 0.147, 0.251])  # Inertia tensor [kg·m²]
        self.l = 0.385                           # Arm length [m]
        self.cf = 1.5e-3                         # Thrust coefficient
        self.ct = 4.590e-5                       # Torque coefficient
        self.u_bar = 108 * 108                   # Max squared rotor speed [rad²/s²]
        self.alpha_tilt = np.deg2rad(20)         # Rotor tilt angle [rad]
        self.g = 9.81                            # Gravitational acceleration [m/s²]
        self.robot_name = "aSTedH"
        self.propRad = 0.172                     # Propeller radius [m]
        self.maxRad = self.l + self.propRad      # Effective safety radius [m]

        # --- MPC parameters ---
        self.nx = 12   # State dimension
        self.nu = 6    # Input dimension
        self.Q = np.diag([
            1e2, 1e2, 1e2,   # Position weights
            1e2, 1e2, 1e2,   # Orientation weights
            1e1, 1e1, 1e1,   # Linear velocity weights
            5e1, 5e1, 5e1,   # Angular velocity weights
        ])
        self.R = 1e0 * np.eye(self.nu)   # Input weight matrix
        self.N = 50                       # Prediction horizon length

        # --- Simulation parameters ---
        self.SimDuration = 5.0                          # Total simulation time [s]
        self.dt = 0.02                                  # Control time step [s]
        self.dtSim = 0.0001                             # Integration time step [s]
        self.nsub = int(self.dt / self.dtSim)           # Sub-steps per control step
        self.time = np.arange(0, self.SimDuration, self.dt)

        # --- Reference state (hover at [0, 0.5, 3] with zero velocity) ---
        self.x_ref = np.array([
            0.0, 0.5, 3.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ])
        self.use_u_ref_hovering = True   # If True, compute hovering input as reference

        # --- Environment: obstacle list ---
        # Each entry is a dict with keys 'center', 'dimensions' (or 'radius'), and 'type'
        self.obstacles = [
            {
                "center": np.array([0.0, 0.0, 1.5]),
                "dimensions": np.array([1.0, 1.0, 0.2]),
                "type": "box",
            },
        ]

        # --- Room extents and state constraints ---
        self.state_constraint_active = True
        self.xlim = [-3.0, 3.0]
        self.ylim = [-3.0, 3.0]
        self.zlim = [-2.0, 4.0]


# =============================================================================
# MODEL
# =============================================================================

class SthModel:
    """CasADi symbolic model for the Star-shaped Tilted Hexarotor (STH).

    Builds symbolic expressions for the rotation matrix, thrust/torque
    allocation matrices, explicit rigid-body dynamics, and obstacle-avoidance
    constraints.  The resulting ``amodel`` attribute is passed directly to
    acados OCP and simulation objects.

    Parameters
    ----------
    params : Params
        Configuration object containing all physical parameters.
    """

    def __init__(self, params: object) -> None:

        self.nx = params.nx
        self.nu = params.nu

        # --- Symbolic state and input variables ---
        x = MX.sym("x", self.nx)
        u = MX.sym("u", self.nu)

        # State decomposition: position, Euler angles, linear and angular velocity
        p = x[0:3]
        rpy = x[3:6]
        v = x[6:9]
        omega = x[9:12]
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

        # --- Rotation matrix (ZYX Euler convention): world ← body ---
        R_x = vertcat(
            horzcat(1,          0,           0        ),
            horzcat(0,  cos(roll), -sin(roll)          ),
            horzcat(0,  sin(roll),  cos(roll)          ),
        )
        R_y = vertcat(
            horzcat( cos(pitch), 0, sin(pitch)),
            horzcat( 0,          1, 0         ),
            horzcat(-sin(pitch), 0, cos(pitch)),
        )
        R_z = vertcat(
            horzcat(cos(yaw), -sin(yaw), 0),
            horzcat(sin(yaw),  cos(yaw), 0),
            horzcat(0,         0,         1),
        )
        R = R_z @ R_y @ R_x
        self.R = ca.Function('R', [x], [R])

        # --- Thrust and torque allocation matrices ---
        sin_a = np.sin(params.alpha_tilt)
        cos_a = np.cos(params.alpha_tilt)
        tan_a = np.tan(params.alpha_tilt)
        self.r = params.cf / params.ct * params.l   # Effective moment-arm ratio

        # F maps normalised rotor inputs to body-frame force [3 × nu]
        self.F = params.u_bar * params.cf * np.array([
            [0,      np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a,
             0,      np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
            [sin_a, -1/2 * sin_a,           -1/2 * sin_a,
             sin_a, -1/2 * sin_a,           -1/2 * sin_a],
            [cos_a,  cos_a,                  cos_a,
             cos_a,  cos_a,                  cos_a],
        ])

        # M maps normalised rotor inputs to body-frame torque [3 × nu]
        self.M = params.u_bar * params.ct * np.array([
            [0,
              np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a,
              np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a,
             0,
             -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a,
             -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a],
            [-self.r * cos_a + sin_a,
             -1/2 * self.r * cos_a + 1/2 * sin_a,
              1/2 * self.r * cos_a - 1/2 * sin_a,
              self.r * cos_a - sin_a,
              1/2 * self.r * cos_a - 1/2 * sin_a,
             -1/2 * self.r * cos_a + 1/2 * sin_a],
            [self.r * sin_a + cos_a,
             -self.r * sin_a - cos_a,
              self.r * sin_a + cos_a,
             -self.r * sin_a - cos_a,
              self.r * sin_a + cos_a,
             -self.r * sin_a - cos_a],
        ])

        # Net control force (world frame) and torque (body frame)
        fc = R @ (self.F @ u)
        tc = self.M @ u

        # --- Euler-rate to angular-velocity transformation (inverse) ---
        Tinv = vertcat(
            horzcat(1, sin(roll) * tan(pitch), cos(roll) * tan(pitch)),
            horzcat(0, cos(roll),              -sin(roll)             ),
            horzcat(0, sin(roll) / cos(pitch),  cos(roll) / cos(pitch)),
        )

        # --- Explicit continuous-time dynamics: x_dot = f(x, u) ---
        f_expl = vertcat(
            v,
            Tinv @ omega,
            -params.g * ca.DM([0, 0, 1]) + fc / params.mass,
            ca.solve(params.J, -cross(omega, params.J @ omega) + tc),
        )
        self.f_expl_func = ca.Function("f_expl", [x, u], [f_expl])

        # --- Per-obstacle nonlinear constraints ---
        # For spheres: squared distance >= (r + maxRad)²
        # For boxes:   signed distance function (SDF) >= 0
        h_expr_list = []
        h_min_list = []
        h_max_list = []

        for obs in params.obstacles:
            if obs["type"] == "sphere":
                c = obs["center"]
                r = obs["radius"]

                # Squared Euclidean distance from drone centre to sphere centre
                h = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2
                h_expr_list.append(h)
                h_min_list.append((r + params.maxRad) ** 2)
                h_max_list.append(1e10)

            elif obs["type"] == "box":
                c = obs["center"]
                d = obs["dimensions"]
                b = d / 2.0   # Half-dimensions (semi-axes)

                # Signed distance function (SDF) for an axis-aligned box
                q = ca.fabs(p - c) - b

                # Outside component: L2 norm of penetration per axis
                outside_dist = ca.norm_2(ca.fmax(q, 0))

                # Inside component: depth of the deepest axis inside the box
                inside_dist = ca.fmin(ca.fmax(q[0], ca.fmax(q[1], q[2])), 0)

                # Full SDF: positive outside, negative inside
                dist = outside_dist + inside_dist

                # Constraint: SDF must exceed the drone safety radius
                h_expr_list.append(dist - params.maxRad)
                h_min_list.append(0.0)
                h_max_list.append(1e10)

        self.h_expr = vertcat(*h_expr_list)
        self.h_min = np.array(h_min_list)
        self.h_max = np.array(h_max_list)
        self.h_func = ca.Function("h_func", [x], [self.h_expr])

        # --- Input bounds (normalised rotor speed in [0, 1]) ---
        self.u_min = np.zeros(self.nu)
        self.u_max = np.ones(self.nu)

        # --- Acados model registration ---
        model = AcadosModel()
        model.name = params.robot_name
        model.x = x
        model.u = u
        model.f_expl_expr = f_expl
        self.amodel = model


# =============================================================================
# INITIAL GUESS
# =============================================================================

def initialize_guess(
    solver: AcadosOcpSolver,
    model: SthModel,
    params: Params,
    x0: np.ndarray,
    u_guess: np.ndarray | None = None,
    x_guess: np.ndarray | None = None,
) -> None:
    """Initialise the OCP solver with state and input guesses.

    Supports both constant (1-D) and trajectory (2-D) guesses for states
    and inputs.  If a guess is not provided, sensible defaults are used.

    Parameters
    ----------
    solver : AcadosOcpSolver
        The acados solver instance to initialise.
    model : SthModel
        Robot model providing state and input dimensions.
    params : Params
        Configuration object providing the horizon length ``N``.
    x0 : np.ndarray
        Initial state, shape (nx,).
    u_guess : np.ndarray or None, optional
        Control guess of shape (nu,) for constant or (N, nu) for trajectory.
        Defaults to zero input.
    x_guess : np.ndarray or None, optional
        State guess of shape (nx,) for constant or (N+1, nx) for trajectory.
        Defaults to ``x0`` at every stage.
    """
    N = params.N
    nu = model.nu

    # --- Default guesses ---
    if u_guess is None:
        u_guess = np.zeros(nu)
    if x_guess is None:
        x_guess = x0.copy()

    # --- Set interior stages ---
    for k in range(N):
        solver.set(k, "u", u_guess if u_guess.ndim == 1 else u_guess[k])
        solver.set(k, "x", x_guess if x_guess.ndim == 1 else x_guess[k])

    # --- Set terminal stage ---
    solver.set(N, "x", x_guess if x_guess.ndim == 1 else x_guess[N])


# =============================================================================
# SIMULATOR
# =============================================================================

def create_acados_sim(
    model: SthModel,
    params: Params,
) -> AcadosSimSolver:
    """Create and return an acados simulation solver.

    Parameters
    ----------
    model : SthModel
        Robot model whose ``amodel`` is passed to the simulator.
    params : Params
        Configuration object providing the integration time step ``dtSim``.

    Returns
    -------
    sim_solver : AcadosSimSolver
        Compiled acados simulation solver ready to integrate one step.
    """
    sim = AcadosSim()
    sim.model = model.amodel
    sim.dims.nx = model.nx
    sim.dims.nu = model.nu
    sim.solver_options.T = params.dtSim
    return AcadosSimSolver(sim, json_file="acados_sim.json")


# =============================================================================
# ROLLBACK GUESS
# =============================================================================

def rollback_guess(
    solver: AcadosOcpSolver,
    model: SthModel,
    params: Params,
    x_current: np.ndarray,
) -> None:
    """Shift the current MPC solution by one step and reinitialise the solver.

    Retrieves the optimal trajectory from the solver, shifts it forward
    by one time step (repeating the last stage), and calls
    ``initialize_guess`` with the updated warm-start.

    Parameters
    ----------
    solver : AcadosOcpSolver
        The acados solver holding the previous solution.
    model : SthModel
        Robot model providing dimensions.
    params : Params
        Configuration object providing the horizon length ``N``.
    x_current : np.ndarray
        State at the next time step used as the new initial constraint.
    """
    N = params.N

    # Retrieve the full solution trajectory
    x_sol = np.array([solver.get(k, "x") for k in range(N + 1)])
    u_sol = np.array([solver.get(k, "u") for k in range(N)])

    # Shift by one step, repeating the terminal stage
    x_guess = np.vstack([x_sol[1:], x_sol[-1]])
    u_guess = np.vstack([u_sol[1:], u_sol[-1]])

    initialize_guess(solver, model, params, x_current,
                     u_guess=u_guess, x_guess=x_guess)


# =============================================================================
# DYNAMICS SIMULATION
# =============================================================================

def dynamicsSim(
    sim_solver: AcadosSimSolver,
    x: np.ndarray,
    u: np.ndarray,
    nsub: int,
) -> np.ndarray:
    """Integrate the system dynamics over ``nsub`` sub-steps.

    Parameters
    ----------
    sim_solver : AcadosSimSolver
        Acados simulation solver performing one integration step at a time.
    x : np.ndarray
        Current state, shape (nx,).
    u : np.ndarray
        Control input held constant during integration, shape (nu,).
    nsub : int
        Number of integration sub-steps to perform.

    Returns
    -------
    x : np.ndarray
        Updated state after all sub-steps, shape (nx,).
    """
    for _ in range(nsub):
        sim_solver.set("x", x)
        sim_solver.set("u", u)
        sim_solver.solve()
        x = sim_solver.get("x")
    return x


# =============================================================================
# MPC + SIMULATION
# =============================================================================

def run_mpc(
    model: SthModel,
    params: Params,
) -> None:
    """Build the tracking OCP, run the MPC loop, and save the trajectory to disk.

    Constructs and compiles the acados OCP with a linear least-squares cost,
    room-boundary state constraints (optional), and per-obstacle nonlinear
    constraints.  Then steps through the simulation horizon, applying the
    optimal input at each step, rolling the warm-start forward, and
    persisting the resulting trajectories.

    Parameters
    ----------
    model : SthModel
        Robot model providing dynamics, constraints, and dimensions.
    params : Params
        Configuration object providing all simulation and OCP settings.
    """
    nx, nu = model.nx, model.nu
    ny = nx + nu      # Stage output dimension
    ny_e = nx         # Terminal output dimension

    # --- OCP object ---
    ocp = AcadosOcp()
    ocp.model = model.amodel
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.N * params.dt

    # --- Cost: linear least-squares on [x; u] deviations ---
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)      # Select full state in stage output

    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)        # Select full input in stage output

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.W = np.block([
        [params.Q,              np.zeros((nx, nu))],
        [np.zeros((nu, nx)),    params.R           ],
    ])
    ocp.cost.W_e = params.Q

    # Compute hovering reference input via pseudo-inverse of the thrust matrix
    x_ref = params.x_ref
    if params.use_u_ref_hovering:
        u_ref = (
            np.linalg.pinv(model.R(x_ref).full() @ model.F)
            @ np.array([0, 0, params.mass * params.g])
        )
    else:
        u_ref = np.zeros(nu)

    ocp.cost.yref = np.hstack((x_ref, u_ref))
    ocp.cost.yref_e = x_ref

    # --- Constraints ---
    ocp.constraints.constr_type = "BGH"

    # Input bounds
    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)

    # Initial state fixed to x0 (set at solve time)
    ocp.constraints.x0 = np.zeros(nx)

    # Room-boundary state constraints (position only, inset by maxRad)
    if params.state_constraint_active:
        ocp.constraints.lbx = np.array([
            params.xlim[0] + params.maxRad,
            params.ylim[0] + params.maxRad,
            params.zlim[0] + params.maxRad,
        ])
        ocp.constraints.ubx = np.array([
            params.xlim[1] - params.maxRad,
            params.ylim[1] - params.maxRad,
            params.zlim[1] - params.maxRad,
        ])
        ocp.constraints.idxbx = np.arange(3)

        ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
        ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
        ocp.constraints.idxbx_e = np.arange(3)

    # Obstacle constraints at initial, path, and terminal stages
    ocp.model.con_h_expr_0 = model.h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max

    ocp.model.con_h_expr = model.h_expr
    ocp.constraints.lh = model.h_min
    ocp.constraints.uh = model.h_max

    ocp.model.con_h_expr_e = model.h_expr
    ocp.constraints.lh_e = model.h_min
    ocp.constraints.uh_e = model.h_max

    # --- Solver options ---
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 500

    # --- Compile solvers ---
    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    sim_solver = create_acados_sim(model, params)

    # --- Initialise simulation state and warm-start ---
    x = np.zeros(nx)
    xg = [x.copy()]
    ug = []

    u_prev = (
        np.linalg.pinv(model.R(x).full() @ model.F)
        @ np.array([0, 0, params.mass * params.g])
        / params.u_bar
    )

    initialize_guess(solver, model, params, x, u_guess=u_prev, x_guess=x)

    # --- MPC loop ---
    for ist in tqdm(range(params.time.shape[0]), desc="MPC Simulation Progress"):
        solver.solve_for_x0(x, False, False)
        u = solver.get(0, "u")

        x = dynamicsSim(sim_solver, x, u, params.nsub)
        rollback_guess(solver, model, params, x)

        ug.append(u)
        xg.append(x.copy())

    # --- Persist trajectory data ---
    traj_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl"
    )
    with open(traj_path, "wb") as f:
        pickle.dump(
            {"xg": np.asarray(xg), "ug": np.asarray(ug)},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    params = Params()
    model = SthModel(params)

    run_mpc(model, params)
    print("MPC simulation completed. Trajectory saved to 'trajectory.pkl'.")

    traj_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl"
    )
    plotter(traj_path, model, params, animate=True)