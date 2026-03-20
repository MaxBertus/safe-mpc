import os
import copy
import ctypes
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from tqdm import tqdm
import casadi as ca
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross, fmin, fmax
import l4casadi as l4c
from acados_template import (
    AcadosModel, AcadosOcp, AcadosOcpSolver,
    AcadosSim, AcadosSimSolver,
)

from safe_mpc import ocp
from utils.plotter import plotter


# =============================================================================
# PARAMETERS
# =============================================================================

class Params:
    """Hard-coded configuration for the STH MPC simulation.

    Groups all physical, control, simulation, environment, and neural
    network parameters into a single object for easy access throughout
    the pipeline.
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
        self.nx = 12    # State dimension
        self.nu = 6     # Input dimension
        self.Q = np.diag([
            1e2, 1e2, 1e2,   # Position weights
            1e2, 1e2, 1e2,   # Orientation weights
            1e1, 1e1, 1e1,   # Linear velocity weights
            5e1, 5e1, 5e1,   # Angular velocity weights
        ])
        self.Qv = np.diag([
            1e2, 1e2, 1e2,   # Linear velocity weights
            1e2, 1e2, 1e2,   # Angular velocity weights
        ])
        self.R = 1e0 * np.eye(self.nu)   # Input weight matrix
        self.N = 60                       # Tracking MPC horizon
        self.Nvboc = 20                   # Safe-abort MPC horizon
        self.nlp_solver_max_iter = 100    # Max NLP iterations

        # --- Simulation parameters ---
        self.SimDuration = 5.0                          # Total simulation time [s]
        self.dt = 0.02                                  # Control time step [s]
        self.dtSim = 1e-4                               # Integration time step [s]
        self.nsub = int(self.dt / self.dtSim)           # Sub-steps per control step
        self.T = int(self.SimDuration / self.dt)        # Total number of control steps
        self.time = np.arange(0, self.SimDuration, self.dt)

        # --- Reference state (hover at [2, 0, 4] with zero velocity) ---
        self.x_ref = np.array([
            2.0, 0.0, 4.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ])
        self.use_u_ref_hovering = True   # If True, compute hovering input as reference

        # --- Environment: obstacle list ---
        # Each entry is a dict with keys 'center', 'dimensions' (or 'radius'), and 'type'
        self.obstacles = [
            {"center": np.array([2.0,  0.0, 0.5]), "dimensions": np.array([0.5, 2.0, 2.0]), "type": "box"},
            {"center": np.array([-1.5, 0.0, 1.0]), "dimensions": np.array([0.5, 3.0, 3.0]), "type": "box"},
            {"center": np.array([0.0,  0.0, 3.0]), "dimensions": np.array([2.0, 2.0, 0.5]), "type": "box"},
        ]

        # --- Room extents [m] ---
        self.xlim = [-3.0, 3.0]
        self.ylim = [-3.0, 3.0]
        self.zlim = [-3.0, 5.0]

        # --- Neural network parameters ---
        # Input: 6 box dims + 3 orientations + 3 linear vels + 3 angular vels = 15
        self.input_size = 15
        self.hidden_size = 1024
        self.output_size = 1
        self.number_hidden = 2
        self.act_fun = torch.nn.GELU(approximate='tanh')
        self.net_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "nn/sth_gelu.pt"
        )
        self.build_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "nn"
        )
        self.eps = 1e-6    # Smoothing term for velocity norm
        self.alpha = 5     # Safety margin percentage for NN constraint scaling

        self.device = (
            torch.device('cuda:0') if torch.cuda.is_available()
            else torch.device('cpu')
        )

        # --- Output directories ---
        self.plots_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "plots"
        )


# =============================================================================
# MODEL
# =============================================================================

class SthModel:
    """CasADi symbolic model for the Star-shaped Tilted Hexarotor (STH).

    Builds symbolic expressions for the rotation matrix, thrust/torque
    allocation matrices, explicit rigid-body dynamics, obstacle-avoidance
    constraints, and input bounds.  The resulting ``amodel`` attribute is
    passed directly to acados OCP and simulation objects.

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
            horzcat(1,           0,            0         ),
            horzcat(0,  cos(roll),  -sin(roll)            ),
            horzcat(0,  sin(roll),   cos(roll)            ),
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

        # --- Axis-aligned box obstacle-avoidance constraint ---
        # box = [x_min, x_max, y_min, y_max, z_min, z_max] in world frame
        # Constraint: drone centre must stay at least maxRad away from each face
        box = ca.MX.sym("b", 6)
        h_box = ca.vertcat(
            p[0] - (box[0] + params.maxRad),    # x >= x_min + maxRad
            (box[1] - params.maxRad) - p[0],    # x <= x_max - maxRad
            p[1] - (box[2] + params.maxRad),    # y >= y_min + maxRad
            (box[3] - params.maxRad) - p[1],    # y <= y_max - maxRad
            p[2] - (box[4] + params.maxRad),    # z >= z_min + maxRad
            (box[5] - params.maxRad) - p[2],    # z <= z_max - maxRad
        )
        self.h_func = ca.Function('h_func', [x, box], [h_box])
        self.h_min = np.zeros(6)
        self.h_max = np.full(6, 1e6)

        # --- Input bounds (normalised rotor speed in [0, 1]) ---
        self.u_min = np.zeros(self.nu)
        self.u_max = np.ones(self.nu)

        # --- Acados model registration ---
        model = AcadosModel()
        model.name = params.robot_name
        model.x = x
        model.u = u
        model.f_expl_expr = f_expl
        model.p = box
        self.amodel = model


# =============================================================================
# INITIAL GUESS
# =============================================================================

def initialize_guess(
    solver: AcadosOcpSolver,
    N: int,
    model: SthModel,
    params: Params,
    x0: np.ndarray,
    u_guess: np.ndarray | None = None,
    x_guess: np.ndarray | None = None,
    p_guess: np.ndarray | None = None,
) -> None:
    """Initialise the OCP solver with state, input, and parameter guesses.

    Supports both constant (1-D) and trajectory (2-D) guesses for states
    and inputs.  If a guess is not provided, sensible defaults are used.

    Parameters
    ----------
    solver : AcadosOcpSolver
        The acados solver instance to initialise.
    N : int
        Prediction horizon length.
    model : SthModel
        Robot model providing state and input dimensions.
    params : Params
        Configuration object (unused directly but kept for API consistency).
    x0 : np.ndarray
        Initial state, shape (nx,).
    u_guess : np.ndarray or None, optional
        Control guess of shape (nu,) for constant or (N, nu) for trajectory.
        Defaults to zero input.
    x_guess : np.ndarray or None, optional
        State guess of shape (nx,) for constant or (N+1, nx) for trajectory.
        Defaults to ``x0`` repeated at every stage.
    p_guess : np.ndarray or None, optional
        Parameter (box) guess, shape (n_p,).  Defaults to zero vector.
    """
    nx = model.nx
    nu = model.nu

    # --- Default guesses ---
    if u_guess is None:
        u_guess = np.zeros(nu)
    if x_guess is None:
        x_guess = x0.copy()
    if p_guess is None:
        p_guess = np.zeros(model.amodel.p.shape[0])

    # --- Set interior stages ---
    for k in range(N):
        solver.set(k, "u", u_guess if u_guess.ndim == 1 else u_guess[k])
        solver.set(k, "x", x_guess if x_guess.ndim == 1 else x_guess[k])

    # --- Set terminal stage ---
    solver.set(N, "x", x_guess if x_guess.ndim == 1 else x_guess[N])

    for k in range(N + 1):
        solver.set(k, "p", p_guess)


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
    sim.parameter_values = np.zeros(6)
    return AcadosSimSolver(sim, json_file="acados_sim.json")


# =============================================================================
# ROLLBACK GUESS
# =============================================================================

def rollback_guess(
    solver: AcadosOcpSolver,
    model: SthModel,
    params: Params,
    x_current: np.ndarray,
    p_current: np.ndarray | None = None,
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
    p_current : np.ndarray or None, optional
        Updated box parameter for the new solve.  Defaults to None.
    """
    N = params.N

    # Retrieve the full solution trajectory
    x_sol = np.array([solver.get(k, "x") for k in range(N + 1)])
    u_sol = np.array([solver.get(k, "u") for k in range(N)])

    # Shift by one step, repeating the terminal stage
    x_guess = np.vstack([x_sol[1:], x_sol[-1]])
    u_guess = np.vstack([u_sol[1:], u_sol[-1]])

    initialize_guess(
        solver, params.N, model, params, x_current,
        u_guess=u_guess,
        x_guess=x_guess,
        p_guess=p_current,
    )


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
        Control input to hold constant during integration, shape (nu,).
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
# OCP DEFINITION
# =============================================================================

def define_ocp(
    model: SthModel,
    params: Params,
    safe_set: object,
) -> AcadosOcp:
    """Build the tracking MPC OCP with a neural-network terminal safe-set constraint.

    Uses a linear least-squares cost on state and input deviations from the
    reference, obstacle-avoidance nonlinear constraints at every stage, and
    the learned viability kernel as an additional terminal constraint.

    Parameters
    ----------
    model : SthModel
        Robot model providing dynamics, constraints, and dimensions.
    params : Params
        Configuration object providing cost weights, horizon, and references.
    safe_set : NetSafeSet
        Object exposing ``nn_func`` — a CasADi function for the NN constraint.

    Returns
    -------
    ocp : AcadosOcp
        Fully configured acados OCP object ready for solver compilation.
    """
    nx, nu = model.nx, model.nu
    ny = nx + nu      # Stage output dimension
    ny_e = nx         # Terminal output dimension

    ocp = AcadosOcp()
    ocp.model = copy.deepcopy(model.amodel)
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.N * params.dt
    ocp.model.name = params.robot_name + "_tracking"

    # --- Cost: linear least-squares on [x; u] deviations ---
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.parameter_values = np.zeros(6)

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

    # Obstacle constraint at initial, path, and terminal stages
    h_expr = model.h_func(model.amodel.x, model.amodel.p)

    ocp.model.con_h_expr_0 = h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max

    ocp.model.con_h_expr = h_expr
    ocp.constraints.lh = model.h_min
    ocp.constraints.uh = model.h_max

    # Terminal constraint: obstacle + NN safe-set stacked
    nn_expr = safe_set.nn_func(model.amodel.x, model.amodel.p)
    ocp.model.con_h_expr_e = vertcat(h_expr, nn_expr)
    ocp.constraints.lh_e = np.zeros((7, 1))
    ocp.constraints.uh_e = np.full((7, 1), 1e6)

    # --- Solver options ---
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = params.nlp_solver_max_iter

    # Link the l4casadi shared library so acados can resolve the model symbol
    lib_name = f'{params.robot_name}_model'
    ocp.solver_options.custom_templates = []
    ocp.solver_options.ext_fun_compile_flags = f'-I{params.build_dir}'
    ocp.solver_options.link_libs = (
        f'-L{params.build_dir} -l{lib_name} -Wl,-rpath,{params.build_dir}'
    )

    return ocp


def define_ocpSafeAbort(
    model: SthModel,
    params: Params,
) -> AcadosOcp:
    """Build the safe-abort OCP that drives velocities to zero.

    Uses a linear least-squares cost penalising deviations of velocities
    and inputs from hovering values, with a shorter horizon ``Nvboc``.
    Obstacle constraints are enforced at all stages; no terminal NN
    constraint is applied.

    Parameters
    ----------
    model : SthModel
        Robot model providing dynamics, constraints, and dimensions.
    params : Params
        Configuration object providing cost weights, horizon, and references.

    Returns
    -------
    ocp : AcadosOcp
        Fully configured acados OCP object ready for solver compilation.
    """
    nx, nu = model.nx, model.nu
    nv = 6              # Number of velocity states (linear + angular)
    ny = nv + nu        # Stage output: velocities + inputs

    ocp = AcadosOcp()
    ocp.model = copy.deepcopy(model.amodel)
    ocp.dims.N = params.Nvboc
    ocp.solver_options.tf = params.Nvboc * params.dt
    ocp.model.name = params.robot_name + "_safe_abort"

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.parameter_values = np.zeros(6)

    # --- Cost matrices: select velocity states and inputs ---
    Vx = np.zeros((ny, nx))
    Vx[0:nv, nv:nx] = np.eye(nv)   # Rows 0:nv select states 6:12 (velocities)

    Vu = np.zeros((ny, nu))
    Vu[nv:, :] = np.eye(nu)         # Rows nv: select full input

    Vx_e = np.zeros((nv, nx))
    Vx_e[:, nv:nx] = np.eye(nv)    # Terminal: select velocity states only

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e

    # Stage weight matrix: block-diagonal [Qv, R]
    W = np.zeros((ny, ny))
    W[:nv, :nv] = params.Qv
    W[nv:, nv:] = params.R
    ocp.cost.W = W
    ocp.cost.W_e = params.Qv

    # Compute hovering reference input
    if params.use_u_ref_hovering:
        u_hover = (
            np.linalg.pinv(model.R(np.zeros(nx)).full() @ model.F)
            @ np.array([0, 0, params.mass * params.g])
            / params.u_bar
        )
    else:
        u_hover = np.zeros(nu)

    # Reference: zero velocities + hovering input
    yref = np.zeros(ny)
    yref[nv:] = u_hover
    ocp.cost.yref = yref
    ocp.cost.yref_e = np.zeros(nv)

    # --- Constraints ---
    ocp.constraints.constr_type = "BGH"

    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.x0 = np.zeros(nx)

    # Obstacle constraint at initial, path, and terminal stages
    h_expr = model.h_func(model.amodel.x, model.amodel.p)

    ocp.model.con_h_expr_0 = h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max

    ocp.model.con_h_expr = h_expr
    ocp.constraints.lh = model.h_min
    ocp.constraints.uh = model.h_max

    ocp.model.con_h_expr_e = h_expr
    ocp.constraints.lh_e = model.h_min
    ocp.constraints.uh_e = model.h_max

    # --- Solver options ---
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = params.nlp_solver_max_iter

    return ocp


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class NeuralNetwork(nn.Module):
    """Fully-connected feedforward neural network with configurable depth.

    Parameters
    ----------
    input_size : int
        Dimension of the input vector.
    hidden_size : int
        Number of neurons in each hidden layer.
    output_size : int
        Dimension of the output vector.
    number_hidden : int
        Number of hidden layers between the input and output layers.
    activation : nn.Module, optional
        Activation function applied after each linear layer.
        Defaults to ``nn.ReLU()``.
    ub : float or None, optional
        Output upper-bound scaling factor.  Defaults to 1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        number_hidden: int,
        activation: nn.Module = nn.ReLU(),
        ub: float | None = None,
    ) -> None:
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)

        # Hidden layers
        for _ in range(number_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(activation)

        self.linear_stack = nn.Sequential(*layers)
        self.ub = ub if ub is not None else 1
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and scale the output by ``ub``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_size).

        Returns
        -------
        out : torch.Tensor
            Scaled network output, shape (batch, output_size).
        """
        return self.linear_stack(x) * self.ub

    def initialize_weights(self) -> None:
        """Initialise all linear layers with Xavier normal weights and zero biases."""
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


# =============================================================================
# SAFE SET
# =============================================================================

class NetSafeSet:
    """Viability kernel approximation via a trained neural network.

    Loads a pre-trained ``NeuralNetwork`` checkpoint, wraps it with
    ``l4casadi`` to obtain a CasADi-compatible function, and exposes
    ``nn_func`` for use inside acados OCP terminal constraints.

    The input to the network is:
        [box (standardised), orientation (standardised), velocity direction]

    Parameters
    ----------
    model : SthModel
        Robot model providing symbolic state and parameter variables.
    params : Params
        Configuration object providing network architecture, paths, and
        normalisation statistics loaded from the checkpoint.
    """

    def __init__(self, model: object, params: object) -> None:
        self.constraints = []
        self.constraints_fun = []
        self.bounds = []

        npos = 3
        nori = 3

        # --- Load network weights ---
        nn_data = torch.load(params.net_path, weights_only=False)
        model_net = NeuralNetwork(
            params.input_size, params.hidden_size, params.output_size,
            params.number_hidden, params.act_fun, ub=1,
        ).to(params.device)
        model_net.load_state_dict(nn_data['model'])

        x_cp = model.amodel.x
        p_cp = model.amodel.p

        # --- Translate box to robot frame and reorder to [mins, maxs] ---
        # Original box p_cp = [x_min, x_max, y_min, y_max, z_min, z_max]
        # Reordered:          [x_min, y_min, z_min, x_max, y_max, z_max]
        box_in_robot_frame = (
            p_cp[[0, 2, 4, 1, 3, 5]]
            - ca.vertcat(x_cp[0], x_cp[1], x_cp[2],
                         x_cp[0], x_cp[1], x_cp[2])
        )

        # Clip lower/upper bounds to room extents to avoid out-of-distribution inputs
        room_lower = ca.DM([-2.0, -2.0, -2.0])
        room_upper = ca.DM([2.0,  2.0,  2.0])

        box_lower = box_in_robot_frame[:3]
        box_upper = box_in_robot_frame[3:]

        box_in_robot_frame[:3] = -ca.fmax(box_lower, room_lower)
        box_in_robot_frame[3:] =  ca.fmin(box_upper, room_upper)

        # Standardise box and orientation using training statistics
        box = (box_in_robot_frame - nn_data['mean']) / nn_data['std']
        orient = (x_cp[npos:npos + nori] - nn_data['mean']) / nn_data['std']

        # Compute unit velocity direction (smoothed norm to avoid division by zero)
        vel_norm = ca.sqrt(
            ca.dot(x_cp[npos + nori:], x_cp[npos + nori:]) + params.eps ** 2
        )
        vel_dir = x_cp[npos + nori:] / vel_norm

        # Assemble network input: [box, orientation, velocity direction]
        state = ca.vertcat(box, orient, vel_dir)

        # Wrap PyTorch model with l4casadi for CasADi compatibility
        self.l4c_model = l4c.L4CasADi(
            model_net,
            device=params.device,
            name=f'{params.robot_name}_model',
            build_dir=params.build_dir,
        )

        # NN constraint: output scaled by (1 - alpha/100) must exceed vel_norm
        nn_model = self.l4c_model(state) * (100 - params.alpha) / 100 - vel_norm
        self.nn_func = ca.Function(
            'nn_func', [model.amodel.x, model.amodel.p], [nn_model]
        )


# =============================================================================
# BOX SELECTION
# =============================================================================

def min_cube_select(
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    goal_point: np.ndarray | None = None,
    drone_radius: float = 0.5,
) -> tuple[float, float, float, float, float, float, int, bool]:
    """Find the largest axis-aligned free-space box avoiding all spheres.

    Uses a greedy, closed-form half-space intersection strategy:

    1. Start from the maximum allowed box ``[-2, 2]³``.
    2. For each sphere intersecting the current box, shrink the face that
       retains the greatest volume while keeping the drone and goal inside.
    3. Repeat until no sphere intersects or no improvement is possible.

    Parameters
    ----------
    Q : np.ndarray or None
        Sphere centres, shape (n_spheres, 3).
    R : np.ndarray or None
        Sphere radii, shape (n_spheres,).  Defaults to zero vector.
    goal_point : np.ndarray or None, optional
        3-D point that must remain inside the box.  Ignored if None.
    drone_radius : float, optional
        Minimum half-extent the box must contain around the origin.
        Default is 0.5.

    Returns
    -------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Box extents along each axis.
    exitflag : int
        1 if no sphere intersects the final box, 0 otherwise.
    goal_included : bool
        True if the goal point lies inside the final box (or no goal given).
    """
    if R is None:
        R = np.zeros(Q.shape[0])

    LIMIT = 2.0

    # Initialise to maximum extent and enforce drone occupancy
    box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    box[0] = min(box[0], -drone_radius)
    box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius)
    box[3] = max(box[3],  drone_radius)
    box[4] = min(box[4], -drone_radius)
    box[5] = max(box[5],  drone_radius)

    # Enforce goal inclusion
    if goal_point is not None:
        box[0] = min(box[0], goal_point[0])
        box[1] = max(box[1], goal_point[0])
        box[2] = min(box[2], goal_point[1])
        box[3] = max(box[3], goal_point[1])
        box[4] = min(box[4], goal_point[2])
        box[5] = max(box[5], goal_point[2])

    max_iter = 100
    tol = 1e-10

    for _ in range(max_iter):
        intersecting = _spheres_intersect_box(Q, R, box, tol)
        if not np.any(intersecting):
            break

        box, moved = _push_faces(
            box, Q[intersecting], R[intersecting], drone_radius, goal_point
        )
        if not moved:
            break

    xOpt = box
    exitflag = 1 if not np.any(_spheres_intersect_box(Q, R, box, tol)) else 0

    if goal_point is not None:
        goal_included = (
            xOpt[0] <= goal_point[0] <= xOpt[1]
            and xOpt[2] <= goal_point[1] <= xOpt[3]
            and xOpt[4] <= goal_point[2] <= xOpt[5]
        )
    else:
        goal_included = True

    return (
        xOpt[0], xOpt[1],
        xOpt[2], xOpt[3],
        xOpt[4], xOpt[5],
        exitflag,
        goal_included,
    )


def _spheres_intersect_box(
    Q: np.ndarray,
    R: np.ndarray,
    box: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return a boolean mask of spheres whose closest point to the box is within their radius.

    Parameters
    ----------
    Q : np.ndarray
        Sphere centres, shape (n, 3).
    R : np.ndarray
        Sphere radii, shape (n,).
    box : np.ndarray
        Box extents ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    tol : float, optional
        Numerical tolerance subtracted from ``R²``. Default is 1e-9.

    Returns
    -------
    mask : np.ndarray of bool
        True for each sphere that intersects the box.
    """
    xMin, xMax, yMin, yMax, zMin, zMax = box

    # Closest point on box to each sphere centre (vectorised)
    cx = np.clip(Q[:, 0], xMin, xMax)
    cy = np.clip(Q[:, 1], yMin, yMax)
    cz = np.clip(Q[:, 2], zMin, zMax)

    dist2 = (Q[:, 0] - cx) ** 2 + (Q[:, 1] - cy) ** 2 + (Q[:, 2] - cz) ** 2
    return dist2 < (R ** 2 - tol)


def _push_faces(
    box: np.ndarray,
    Qi: np.ndarray,
    Ri: np.ndarray,
    drone_radius: float,
    goal_point: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Shrink one box face per intersecting sphere to maximise retained volume.

    For each sphere, six candidate face positions are evaluated (one per face).
    The candidate that retains the most volume without violating drone-occupancy
    or goal-inclusion constraints is applied.

    Parameters
    ----------
    box : np.ndarray
        Current box extents ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    Qi : np.ndarray
        Centres of intersecting spheres, shape (k, 3).
    Ri : np.ndarray
        Radii of intersecting spheres, shape (k,).
    drone_radius : float
        Minimum half-extent the box must contain around the origin.
    goal_point : np.ndarray or None
        3-D goal point that must remain inside the box.

    Returns
    -------
    box : np.ndarray
        Updated box extents after face pushes.
    moved : bool
        True if at least one face was successfully moved.
    """
    xMin, xMax, yMin, yMax, zMin, zMax = box
    moved = False
    face_idx = {'xMin': 0, 'xMax': 1, 'yMin': 2, 'yMax': 3, 'zMin': 4, 'zMax': 5}

    for i in range(len(Qi)):
        cx, cy, cz = Qi[i]
        r = Ri[i]

        # Build candidate face moves and their resulting box volumes
        candidates = []

        new_xMin = cx + r + 1e-4
        if -2.0 <= new_xMin <= 0:
            vol = (xMax - new_xMin) * (yMax - yMin) * (zMax - zMin)
            candidates.append(('xMin', new_xMin, vol))

        new_xMax = cx - r - 1e-4
        if 0 <= new_xMax <= 2.0:
            vol = (new_xMax - xMin) * (yMax - yMin) * (zMax - zMin)
            candidates.append(('xMax', new_xMax, vol))

        new_yMin = cy + r + 1e-4
        if -2.0 <= new_yMin <= 0:
            vol = (xMax - xMin) * (yMax - new_yMin) * (zMax - zMin)
            candidates.append(('yMin', new_yMin, vol))

        new_yMax = cy - r - 1e-4
        if 0 <= new_yMax <= 2.0:
            vol = (xMax - xMin) * (new_yMax - yMin) * (zMax - zMin)
            candidates.append(('yMax', new_yMax, vol))

        new_zMin = cz + r + 1e-4
        if -2.0 <= new_zMin <= 0:
            vol = (xMax - xMin) * (yMax - yMin) * (zMax - new_zMin)
            candidates.append(('zMin', new_zMin, vol))

        new_zMax = cz - r - 1e-4
        if 0 <= new_zMax <= 2.0:
            vol = (xMax - xMin) * (yMax - yMin) * (new_zMax - zMin)
            candidates.append(('zMax', new_zMax, vol))

        if not candidates:
            continue

        # Apply the volume-maximising candidate that satisfies constraints
        face, val, _ = max(candidates, key=lambda c: c[2])
        new_box = np.array([xMin, xMax, yMin, yMax, zMin, zMax])
        new_box[face_idx[face]] = val

        if not _violates_constraints(new_box, drone_radius, goal_point):
            xMin, xMax, yMin, yMax, zMin, zMax = new_box
            box = new_box
            moved = True

    return box, moved


def _violates_constraints(
    box: np.ndarray,
    drone_radius: float,
    goal_point: np.ndarray | None,
) -> bool:
    """Check whether the box violates drone-occupancy or goal-inclusion constraints.

    Parameters
    ----------
    box : np.ndarray
        Box extents ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    drone_radius : float
        Minimum half-extent around the origin that the box must contain.
    goal_point : np.ndarray or None
        3-D goal point that must lie inside the box.  Ignored if None.

    Returns
    -------
    violated : bool
        True if any constraint is violated.
    """
    xMin, xMax, yMin, yMax, zMin, zMax = box

    # Drone occupancy: box must contain a sphere of radius drone_radius at origin
    if (-xMin < drone_radius or xMax < drone_radius
            or -yMin < drone_radius or yMax < drone_radius
            or -zMin < drone_radius or zMax < drone_radius):
        return True

    # Goal point must be inside box
    if goal_point is not None:
        if not (xMin <= goal_point[0] <= xMax
                and yMin <= goal_point[1] <= yMax
                and zMin <= goal_point[2] <= zMax):
            return True

    return False


def discretize_box_surface(
    center: np.ndarray,
    dims: np.ndarray,
    step: float,
) -> np.ndarray:
    """Sample points on the surface of an axis-aligned box at a given resolution.

    Parameters
    ----------
    center : np.ndarray
        Box centre ``(cx, cy, cz)``, shape (3,).
    dims : np.ndarray
        Box full dimensions ``(dx, dy, dz)``, shape (3,).
    step : float
        Maximum grid spacing between adjacent sample points [m].

    Returns
    -------
    points : np.ndarray
        Deduplicated surface points (vertices, edges, face interiors included),
        shape (N, 3).
    """
    cx, cy, cz = center
    dx, dy, dz = dims / 2.0

    x = np.linspace(cx - dx, cx + dx, int(np.ceil((2 * dx) / step)) + 1)
    y = np.linspace(cy - dy, cy + dy, int(np.ceil((2 * dy) / step)) + 1)
    z = np.linspace(cz - dz, cz + dz, int(np.ceil((2 * dz) / step)) + 1)

    X,  Y  = np.meshgrid(x, y, indexing='ij')
    X2, Z  = np.meshgrid(x, z, indexing='ij')
    Y2, Z2 = np.meshgrid(y, z, indexing='ij')

    # One array per face (bottom, top, front, back, left, right)
    faces = [
        np.column_stack([X.ravel(),  Y.ravel(),  np.full(X.size,  cz - dz)]),
        np.column_stack([X.ravel(),  Y.ravel(),  np.full(X.size,  cz + dz)]),
        np.column_stack([X2.ravel(), np.full(X2.size, cy - dy), Z.ravel()]),
        np.column_stack([X2.ravel(), np.full(X2.size, cy + dy), Z.ravel()]),
        np.column_stack([np.full(Y2.size, cx - dx), Y2.ravel(), Z2.ravel()]),
        np.column_stack([np.full(Y2.size, cx + dx), Y2.ravel(), Z2.ravel()]),
    ]

    # Remove duplicates introduced at shared edges and vertices
    return np.unique(np.vstack(faces), axis=0)


# =============================================================================
# MPC + SIMULATION
# =============================================================================

def run_mpc(
    model: SthModel,
    params: Params,
) -> None:
    """Run the full MPC simulation loop and save the trajectory to disk.

    For each initial state in ``x0_list``:

    1. Discretise obstacles into sphere clouds for the box solver.
    2. Compile the tracking and safe-abort OCP solvers.
    3. Step through the simulation horizon, applying the tracking MPC
       control or falling back to stored inputs on infeasibility.
    4. If consecutive failures exceed the horizon, execute the pre-computed
       safe-abort trajectory.
    5. Persist state, input, and box trajectories to ``data/trajectory.pkl``.

    Parameters
    ----------
    model : SthModel
        Robot model providing dynamics and constraint functions.
    params : Params
        Configuration object providing all simulation settings.
    """
    # --- Build neural-network safe set ---
    safe_set = NetSafeSet(model, params)

    # --- Discretise obstacles into sphere point clouds ---
    obsCenters = []
    obsRadii = []

    for obs in params.obstacles:
        if obs['type'] == 'sphere':
            obsCenters.append(obs['center'])
            obsRadii.append(obs['radius'])
        elif obs['type'] == 'box':
            disc = discretize_box_surface(obs['center'], obs['dimensions'], params.maxRad)
            obsCenters.append(disc)
            obsRadii.append(np.full(disc.shape[0], 0.01))

    if obsCenters:
        obsCenters = np.vstack(obsCenters)
        obsRadii = (
            np.array(obsRadii) if obs['type'] == 'sphere'
            else np.concatenate(obsRadii)
        )
    else:
        obsCenters = np.empty((0, 3))
        obsRadii = np.empty((0,))

    # --- Compile OCP solvers ---
    ocp_tracking = define_ocp(model, params, safe_set)
    ocpSafeAbort = define_ocpSafeAbort(model, params)

    # Load l4casadi shared library globally so acados can resolve the model symbol
    ctypes.CDLL(
        os.path.join(params.build_dir, f'lib{params.robot_name}_model.so'),
        mode=ctypes.RTLD_GLOBAL,
    )

    solver = AcadosOcpSolver(ocp_tracking, json_file="acados_ocp.json")
    solverSafeAbort = AcadosOcpSolver(ocpSafeAbort, json_file="acados_ocp_safe_abort.json")
    sim_solver = create_acados_sim(model, params)

    # Placeholder: list of initial states (extend for multi-IC experiments)
    x0_list = np.zeros((1, model.nx))

    xg_all, ug_all, bg_all = [], [], []

    for x0 in x0_list:
        x = x0.copy()
        xg = [x.copy()]
        ug = []
        bg = []

        x_prev = np.full((params.N + 1, model.nx), x)
        u0 = (
            np.linalg.pinv(model.R(x).full() @ model.F)
            @ np.array([0, 0, params.mass * params.g])
            / params.u_bar
        )
        u_prev = np.full((params.N, model.nu), u0)

        # Compute initial free-space box
        if len(obsCenters) > 0:
            discObsPoints = obsCenters - x[:3]
            x_min, x_max, y_min, y_max, z_min, z_max, _, _ = min_cube_select(
                discObsPoints, obsRadii, drone_radius=params.maxRad
            )
            box = np.array([
                x_min + x[0], x_max + x[0],
                y_min + x[1], y_max + x[1],
                z_min + x[2], z_max + x[2],
            ])
        else:
            box = np.array([
                params.xlim[0], params.xlim[1],
                params.ylim[0], params.ylim[1],
                params.zlim[0], params.zlim[1],
            ])

        initialize_guess(
            solver, params.N, model, params, x,
            u_guess=u_prev, x_guess=x_prev, p_guess=box,
        )

        # --- MPC loop ---
        fails = 0
        follow_safe_abort = False
        u_safe_abort = None
        ttot = 0.0
        tmax = 0.0
        tmin = params.dt * 10

        for i in tqdm(range(params.time.shape[0]), desc="MPC Simulation Progress"):
            t0 = time.perf_counter()
            solver.solve_for_x0(x, False, False)
            t1 = (time.perf_counter() - t0) * 10

            tmax = max(tmax, t1)
            tmin = min(tmin, t1)
            ttot += 1

            x_sol = np.array([solver.get(k, "x") for k in range(params.N + 1)])
            u_sol = np.array([solver.get(k, "u") for k in range(params.N)])
            feas = solver.get_status()

            if feas == 0:
                # Nominal: apply first optimal input and roll back guess
                fails = 0
                u_to_apply = u_sol[0]
                x_prev = x_sol.copy()
                u_prev = u_sol.copy()
            else:
                if fails == 0:
                    # First failure: pre-compute safe-abort trajectory
                    print("Alert: MPC infeasibility detected.")
                    x_safe_start = x_prev[params.N, :]
                    u_hover = (
                        np.linalg.pinv(model.R(x_safe_start).full() @ model.F)
                        @ np.array([0, 0, params.mass * params.g])
                        / params.u_bar
                    )
                    x_guess_abort = np.tile(x_safe_start, (params.Nvboc + 1, 1))
                    u_guess_abort = np.tile(u_hover, (params.Nvboc, 1))

                    initialize_guess(
                        solverSafeAbort, params.Nvboc, model, params,
                        x_safe_start,
                        u_guess=u_guess_abort,
                        x_guess=x_guess_abort,
                        p_guess=box,
                    )
                    solverSafeAbort.solve_for_x0(x_safe_start, False, False)
                    u_safe_abort = np.array([
                        solverSafeAbort.get(k, "u") for k in range(params.Nvboc)
                    ])

                if fails == params.N:
                    # Too many consecutive failures: switch to safe abort
                    print(f"Switching to safe abort trajectory at t={i * params.dt:.2f}s")
                    follow_safe_abort = True
                    break

                # Fall back to stored input from the previous feasible solution
                u_to_apply = u_prev[fails]
                fails += 1

            x_next = dynamicsSim(sim_solver, x, u_to_apply, params.nsub)

            # Update free-space box for the next step
            if len(obsCenters) > 0:
                discObsPoints = obsCenters - x_next[:3]
                x_min, x_max, y_min, y_max, z_min, z_max, boxFeasible, _ = min_cube_select(
                    discObsPoints, obsRadii, drone_radius=params.maxRad
                )
                if boxFeasible:
                    box = np.array([
                        x_min + x_next[0], x_max + x_next[0],
                        y_min + x_next[1], y_max + x_next[1],
                        z_min + x_next[2], z_max + x_next[2],
                    ])

            if feas == 0:
                rollback_guess(solver, model, params, x_next, p_current=box)
            else:
                # Manual shift without reading from the infeasible solver
                u_shifted = np.vstack([u_prev[1:], u_prev[-1]])
                x_shifted = np.vstack([x_prev[1:], x_prev[-1]])
                initialize_guess(
                    solver, params.N, model, params, x_next,
                    u_guess=u_shifted, x_guess=x_shifted, p_guess=box,
                )

            x = x_next.copy()
            ug.append(u_to_apply)
            xg.append(x_next.copy())
            bg.append(box.copy())

        print(
            f"Average time per iteration: {ttot / i:.2f} ms, "
            f"max: {tmax:.2f} ms, min: {tmin:.2f} ms"
        )

        # --- Execute safe-abort trajectory if triggered ---
        if follow_safe_abort:
            print(f"x_safe_start: {x_prev[params.N, :]}")
            print(f"x after {fails} fallback steps: {x}")
            print(f"Difference: {x - x_prev[params.N, :]}")
            input("Press Enter to continue...")

            for j in range(params.Nvboc):
                x_next = dynamicsSim(sim_solver, x, u_safe_abort[j], params.nsub)
                x = x_next.copy()
                ug.append(u_safe_abort[j])
                xg.append(x_next.copy())
                bg.append(box.copy())

        xg_all.append(np.asarray(xg))
        ug_all.append(np.asarray(ug))
        bg_all.append(np.asarray(bg))

    # --- Persist trajectory data ---
    traj_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl"
    )
    with open(traj_path, "wb") as f:
        pickle.dump({"xg": xg_all, "ug": ug_all, "bg": bg_all}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)


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