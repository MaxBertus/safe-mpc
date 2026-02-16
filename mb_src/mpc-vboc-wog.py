import numpy as np
from scipy.optimize import minimize
import casadi as ca
import torch
import torch.nn as nn
import copy
import l4casadi as l4c
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross, fmin, fmax
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from safe_mpc import ocp
from utils.plotter import plotter
import os
import pickle
from tqdm import tqdm

# =========================================================
# PARAMETERS
# =========================================================
class Params:
    def __init__(self):

        # *** MODEL PARAMETERS  ***
        self.mass = 3.5
        self.J = np.diag([0.155, 0.147, 0.251])
        self.l = 0.385
        self.cf = 1.5e-3
        self.ct = 4.590e-5  
        self.u_bar = 108*108
        self.alpha_tilt = np.deg2rad(20)
        self.g = 9.81
        self.robot_name = "aSTedH"
        self.propRad = 0.172
        self.maxRad = self.l + self.propRad
        

        # *** MPC PARAMETERS ***
        self.nx = 12
        self.nu = 6
        self.Q = np.diag([1e2, 1e2, 1e2,    # position
                           1e2, 1e2, 1e2,   # orientation
                           1e1, 1e1, 1e1,   # linear velocities 
                           5e1, 5e1, 5e1])  # angular velocities
        self.R = 1e0 * np.eye(self.nu)
        self.N = 100

        # *** SIMULATION PARAMETERS ***
        self.SimDuration = 5.0  
        self.dt = 0.02
        self.dtSim = 0.0001
        self.nsub = int( self.dt / self.dtSim )  # Number of simulation steps for each control steps
        self.T = int(self.SimDuration / self.dt)
        self.time = np.arange(0, self.SimDuration, self.dt)

        # *** REFERENCE STATE ***
        self.x_ref = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.use_u_ref_hovering = True

        # *** ENVIRONMENT PARAMETERS ***
        # Obstacles
        self.obstacles = [
            # {"center": np.array([0.0, 0.0, 2.0]), "dimensions": np.array([0.5, 0.5, 0.5]), "type": "box"},
            # {"center": np.array([0.0, 0.0, 1.5]), "radius": 0.2, "type": "sphere"},    
        ]
        
        # Room dimensions
        self.xlim = [-3.0, 3.0]
        self.ylim = [-3.0, 3.0] 
        self.zlim = [-2.0, 4.0]

        # *** NEURAL NETWORK PARAMETERS ***
        self.input_size = 15 # 6 box dimensions + 3 orientations + 3 linear velocities + 3 angular velocities = 15
        self.hidden_size = 512
        self.output_size = 1
        self.number_hidden = 3
        self.act_fun = torch.nn.GELU(approximate='tanh')
        self.net_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"nn/sth_{self.act_fun}.pt")
        self.build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn")
        self.eps = 1e-6
        self.alpha = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# MODEL GENERATION
# =========================================================
class SthModel:
    def __init__(self, params):
    
        # *** PROBLEM VARIABLES DEFINITION ***
        self.nx = params.nx
        self.nu = params.nu

        x = MX.sym("x", self.nx)
        u = MX.sym("u", self.nu)

        p = x[0:3]
        rpy = x[3:6]
        v = x[6:9]
        omega = x[9:12]

        roll, pitch, yaw = rpy[0], rpy[1], rpy[2] 

        # *** ROTATION MATRIX DEFINITION ***
        R_x = vertcat(
            horzcat(1, 0, 0),
            horzcat(0, cos(roll), -sin(roll)),
            horzcat(0, sin(roll), cos(roll))
        )
        R_y = vertcat(
            horzcat(cos(pitch), 0, sin(pitch)),
            horzcat(0, 1, 0),
            horzcat(-sin(pitch), 0, cos(pitch))
        )
        R_z = vertcat(
            horzcat(cos(yaw), -sin(yaw), 0),
            horzcat(sin(yaw), cos(yaw), 0),
            horzcat(0, 0, 1)
        )

        R = R_z @ R_y @ R_x

        self.R = ca.Function('R', [x], [R])

        # *** WRENCH MATRICES DEFINITION ***
        sin_a = np.sin(params.alpha_tilt)
        cos_a = np.cos(params.alpha_tilt)
        tan_a = np.tan(params.alpha_tilt)

        self.r = params.cf / params.ct * params.l

        self.F = params.u_bar * params.cf * np.array([
        [0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a, 0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
        [sin_a, -1/2 * sin_a, -1/2 * sin_a, sin_a, -1/2 * sin_a, -1/2 * sin_a],
        [cos_a, cos_a, cos_a, cos_a, cos_a, cos_a]
        ])

        self.M = params.u_bar * params.ct * np.array([
            [0, np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 0, -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a],
            [-self.r * cos_a + sin_a, -1/2 * self.r * cos_a + 1/2 * sin_a, 1/2 * self.r * cos_a - 1/2 * sin_a, self.r * cos_a - sin_a, 1/2 * self.r * cos_a - 1/2 * sin_a, -1/2 * self.r * cos_a + 1/2 * sin_a],
            [self.r * sin_a + cos_a, -self.r * sin_a - cos_a, self.r * sin_a + cos_a, -self.r * sin_a - cos_a, self.r * sin_a + cos_a, -self.r * sin_a - cos_a]
        ])

        # *** CONTROL WRENCH DEFINITION ***   
        fc = R @ (self.F @  u)
        tc = self.M @ u

        # *** DYNAMICS DEFINITION ***
        Tinv = vertcat(
            horzcat(1, sin(roll)*tan(pitch), cos(roll)*tan(pitch)),
            horzcat(0, cos(roll), -sin(roll)),
            horzcat(0, sin(roll)/cos(pitch), cos(roll)/cos(pitch))
        )

        f_expl = vertcat(
            v,
            Tinv @ omega,
            -params.g * ca.DM([0, 0, 1]) + fc / params.mass,
            ca.solve(params.J, -cross(omega, params.J @ omega) + tc)
        )

        self.f_expl_func = ca.Function("f_expl", [x, u], [f_expl])

        # *** OBSTACLE CONSTRAINT DEFINITION ***
        box = ca.MX.sym("b", 6)

        h_box = ca.vertcat(
            p[0] - (box[0] + params.maxRad),  # x >= x_min + maxRad
            (box[1] - params.maxRad) - p[0],  # x <= x_max - maxRad
            p[1] - (box[2] + params.maxRad),  # y >= y_min + maxRad
            (box[3] - params.maxRad) - p[1],  # y <= y_max - maxRad
            p[2] - (box[4] + params.maxRad),  # z >= z_min + maxRad
            (box[5] - params.maxRad) - p[2]   # z <= z_max - maxRad
        )
        
        self.h_expr = h_box
        self.h_min = np.zeros(6)
        self.h_max = np.full(6, 1e6)

        # *** INPUT CONSTRAINTS DEFINITION ***
        self.u_min = np.zeros(self.nu)
        self.u_max = np.ones(self.nu)


        # *** ACADOS MODEL CREATION ***
        model = AcadosModel()
        model.name = params.robot_name
        model.x = x
        model.u = u
        model.f_expl_expr = f_expl
        model.p = box

        self.amodel = model

# =========================================================
# INITIAL GUESS
# =========================================================
def initialize_guess(solver, model, params, x0, u_guess=None, x_guess=None, p_guess=None):
    """
    Initialize the OCP with a state and control guess.

    x0      : initial state (nx,)
    u_guess : control guess (nu,) or (N, nu)
    x_guess : state guess (nx,) or (N+1, nx)
    p_guess : parameter guess (np.ndarray of shape (n_p,))
    """

    # *** DIMENSIONS DEFINITION ***
    N = params.N
    nx = model.nx
    nu = model.nu

    # *** DEFAULT GUESSES DEFINITION ***
    if u_guess is None:
        u_guess = np.zeros(nu)

    if x_guess is None:
        x_guess = x0.copy()

    if p_guess is None:
        p_guess = np.zeros(model.p.shape[0])

    # *** CUSTOM GUESSES ASSIGNMENT ***
    # Intermidiate steps
    for k in range(N):
        if u_guess.ndim == 1:
            solver.set(k, "u", u_guess)
        else:
            solver.set(k, "u", u_guess[k])

        if x_guess.ndim == 1:
            solver.set(k, "x", x_guess)
        else:
            solver.set(k, "x", x_guess[k])

    # Terminal step
    if x_guess.ndim == 1:
        solver.set(N, "x", x_guess)
    else:
        solver.set(N, "x", x_guess[N])

    for k in range(N+1):
        solver.set(k, "p", p_guess)

# =========================================================
# CREATE SIMULATOR
# =========================================================
def create_acados_sim(model, params):
    '''Create acados simulation object.'''
    sim = AcadosSim()
    sim.model = model.amodel

    sim.dims.nx = model.nx
    sim.dims.nu = model.nu

    sim.solver_options.T = params.dtSim

    return AcadosSimSolver(sim, json_file="acados_sim.json")

# =========================================================
# ROLLBACK GUESS
# =========================================================
def rollback_guess(solver, model, params, x_current, p_current=None):
    """
    Shift MPC solution and reinitialize the solver with the new guess.

    Returns
    -------
    u0 : np.ndarray
        Control input to apply at the current step.
    """

    N = params.N

    # *** RETRIEVE SOLUTION ***
    x_sol = np.array([solver.get(k, "x") for k in range(N + 1)])
    u_sol = np.array([solver.get(k, "u") for k in range(N)])

    # *** SHIFT SOLUTION ***
    x_guess = np.vstack([x_sol[1:], x_sol[-1]])
    u_guess = np.vstack([u_sol[1:], u_sol[-1]])

    # *** SET GUESS ***
    initialize_guess(
        solver,
        model,
        params,
        x_current,
        u_guess=u_guess,
        x_guess=x_guess,
        p_guess=p_current
    )

# =========================================================
# DYNAMICS SIMULATION
# =========================================================
def dynamicsSim(sim_solver, x, u, nsub):
    """
    Simulate the system using multiple sub-steps.

    Returns
    -------
    x : np.ndarray
        Updated state after simulation.
    """

    for _ in range(nsub):
        sim_solver.set("x", x)
        sim_solver.set("u", u)
        sim_solver.solve()
        x = sim_solver.get("x")

    return x

# =========================================================
# OCP DEFINITION
# =========================================================

def define_ocp(model, params, safe_set):
    # *** PROBLEM DIMENSIONS DEFINITION ***
    nx, nu = model.nx, model.nu
    ny = nx + nu
    ny_e = nx

    # *** OCP OBJECT CREATION ***
    ocp = AcadosOcp()
    ocp.model = model.amodel
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.N * params.dt

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # *** COST FUNCTION DEFINITION ***
    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)

    Vx_e = np.eye((nx))

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e

    Q = params.Q
    R = params.R

    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))],
                            [np.zeros((nu, nx)), R]])
    ocp.cost.W_e = Q

    x_ref = params.x_ref
    if params.use_u_ref_hovering:
        u_ref = np.linalg.pinv(model.R(x_ref).full() @ model.F) @ np.array([0, 0, params.mass * params.g])
    else:
        u_ref = np.zeros(nu)

    ocp.cost.yref = np.hstack((x_ref, u_ref))
    ocp.cost.yref_e = x_ref

    # *** CONSTRAINTS DEFINITION ***
    # Input
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)

    # Initial state constraint
    ocp.constraints.x0 = x0
    
    # Obstacles 
    ocp.model.con_h_expr_0 = model.h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max

    ocp.model.con_h_expr = model.h_expr
    ocp.constraints.lh = model.h_min
    ocp.constraints.uh = model.h_max

    ocp.model.con_h_expr_e = model.h_expr
    ocp.constraints.lh_e = model.h_min
    ocp.constraints.uh_e = model.h_max

    # Neural network safe set constraint
    nn_expr = safe_set.nn_func(model.amodel.x, model.amodel.p)

    ocp.model.con_h_expr_e = vertcat(model.h_expr, nn_expr)
    ocp.constraints.lh_e = np.concatenate([model.h_min, [0.]])
    ocp.constraints.uh_e = np.concatenate([model.h_max, [1e6]])

    # *** SOLVER PARAMETERS DEFINITION ***
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP" 

    ocp.solver_options.nlp_solver_max_iter = 500

    return ocp

def define_ocpSafeAbort(model, params):
    # *** PROBLEM DIMENSIONS DEFINITION ***
    nx, nu = model.nx, model.nu
    ny = nu

    # *** OCP OBJECT CREATION ***
    ocp = AcadosOcp()
    ocp.model = model.amodel
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.N * params.dt

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # *** COST FUNCTION DEFINITION ***
    Vx = np.zeros((ny, nx))
    Vu = np.eye(nu)

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.zeros((nx, nx))

    R = params.R
    ocp.cost.W = R
    ocp.cost.W_e = np.zeros((nx, nx)) 

    if params.use_u_ref_hovering:
        u_ref = np.linalg.pinv(model.R(params.x_ref).full() @ model.F) @ np.array([0, 0, params.mass * params.g])
    else:
        u_ref = np.zeros(nu)

    ocp.cost.yref = u_ref
    ocp.cost.yref_e = np.zeros(nx)

    # *** CONSTRAINTS DEFINITION ***
    # Input
    ocp.constraints.constr_type = "BGH"

    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)
    
    # Initial state constraint
    ocp.constraints.x0 = np.zeros(nx)  
    
    # Velocity constraint
    idx_vel = np.arange(6, 12)    # velocities

    ocp.constraints.idxbx_e = idx_vel
    ocp.constraints.lbx_e = np.zeros(6)
    ocp.constraints.ubx_e = np.zeros(6)

    # Obstacles 
    ocp.model.con_h_expr_0 = model.h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max

    ocp.model.con_h_expr = model.h_expr
    ocp.constraints.lh = model.h_min
    ocp.constraints.uh = model.h_max

    ocp.model.con_h_expr_e = model.h_expr
    ocp.constraints.lh_e = model.h_min
    ocp.constraints.uh_e = model.h_max

    # *** SOLVER PARAMETERS DEFINITION ***
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP" 

    ocp.solver_options.nlp_solver_max_iter = 500

    return ocp

# =========================================================
# NEURAL NETWORK DEFINITION
# =========================================================

class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, number_hidden, activation=nn.ReLU(), ub=None):
        super().__init__()
        layers=[]

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

        #self.input_size = input_size

    def forward(self, x):
        #out = self.linear_stack(x[:,:self.input_size])* self.ub 
        out = self.linear_stack(x) * self.ub 

        return out #(out + 1) * self.ub / 2
    
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

# =========================================================
# SAFE SET DEFINITION
# =========================================================

class NetSafeSet():
    def __init__(self,model,params):
        self.constraints = []
        self.constraints_fun = []
        self.bounds = []

        npos = 3
        nori = 3
        nvel = 3
        nang_vel = 3
        nbox = 6
        nbori = nbox + nori

        nn_data = torch.load(params.net_path,
                             map_location=params.device)
        model_net = NeuralNetwork(params.input_size, params.hidden_size, params.output_size, params.number_hidden, params.act_fun, 1).to(params.device)
        model_net.load_state_dict(nn_data['model'])

        x_cp = model.amodel.x
        p_cp = model.amodel.p

        # Standardize box dimensions
        box = p_cp - x_cp[:npos]
        box = (p_cp - nn_data['mean'][:nbox]) / nn_data['std'][:nbox]

        # Standardize initial orientation
        orient = (x_cp[npos:npos+nori] - nn_data['mean'][nbox:nbori]) / nn_data['std'][nbox:nbori]

        # Normalize velocities            
        vel_norm = ca.norm_2(x_cp[npos+nori:])
        vel_dir = x_cp[npos+nori:] / (vel_norm + params.eps) 

        state = ca.vertcat(box, orient, vel_dir)

        self.l4c_model = l4c.L4CasADi(model_net,
                                      device=params.device,
                                      name=f'{params.robot_name}_model',
                                      build_dir=params.build_dir)
    
        nn_model = self.l4c_model(state) * (100 - params.alpha) / 100 - vel_norm
       
        self.nn_func = ca.Function('nn_func', [model.x, model.p], [nn_model])

# =========================================================
# BOX CONSTRAINTS DEFINITION
# =========================================================

def min_cube_select(params, Q, R=None, goal_point=None):
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
            min(-params.maxRad, goal_point[0] - margin),
            max(params.maxRad, goal_point[0] + margin),
            min(-params.maxRad, goal_point[1] - margin),
            max(params.maxRad, goal_point[1] + margin),
            min(-params.maxRad, goal_point[2] - margin),
            max(params.maxRad, goal_point[2] + margin)
        ])
    else:
        x0 = np.array([-params.maxRad, params.maxRad, -params.maxRad, params.maxRad, -params.maxRad, params.maxRad])

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
        "fun": lambda x: drone_occupancy(x, params.maxRad)
    })

    # Goal point constraint (if provided)
    if goal_point is not None:
        eps = 1e-3
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
        exitflag,
        goal_included
    )

def box_volume(x):
    """
    Compute volume of the box.
    """
    dx = x[1] - x[0]
    dy = x[3] - x[2]
    dz = x[5] - x[4]
    return dx * dy * dz

def drone_occupancy(x, drone_radius):
    """
    Inequality constraints ensuring the box can contain a sphere
    of radius 0.5 centered at the origin.
    
    Since xMin, yMin, zMin are always negative (from bounds),
    we ensure they are <= -0.5, and xMax, yMax, zMax are >= 0.5.
    
    Returns c such that c >= 0 (for scipy 'ineq' constraint).
    """
    xMin, xMax, yMin, yMax, zMin, zMax = x
    
    eps = 1e-6  # small tolerance for numerical stability
    
    # Since mins are negative: xMin <= -drone_radius means -xMin >= drone_radius
    # Since maxs are positive: xMax >= drone_radius
    c = np.array([
        -xMin - drone_radius + eps,  # -xMin >= drone_radius (since xMin < 0)
        xMax - drone_radius + eps,   # xMax >= drone_radius
        -yMin - drone_radius + eps,  # -yMin >= drone_radius (since yMin < 0)
        yMax - drone_radius + eps,   # yMax >= drone_radius
        -zMin - drone_radius + eps,  # -zMin >= drone_radius (since zMin < 0)
        zMax - drone_radius + eps    # zMax >= drone_radius
    ])
    
    return c

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

        # inequality: dist^2 >= 0 
        c[i] = -r**2 + dist2 - 1e-3

    return c

def discretize_box_surface(center, dims, step):
    """
    Discretize the surface of an axis-aligned box.
    
    Parameters
    ----------
    center : array-like (3,)
        Box center (cx, cy, cz)
    dims : array-like (3,)
        Box dimensions (dx, dy, dz)
    step : float
        Maximum grid spacing
        
    Returns
    -------
    points : ndarray (N, 3)
        Surface points including vertices
    """
    cx, cy, cz = center
    dx, dy, dz = dims / 2.0 

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

# =========================================================
# MPC + SIMULATION
# =========================================================

def run_mpc(model, params):
    '''Compute, apply and simulate MPC control.'''
    
    # *** DEFINE SAFE SET ***
    safe_set = NetSafeSet(model, params)

    # *** DISCRETIZE OBSTACLES FOR CONSTRAINTS ***
    obsCenters = []
    obsRadii = []

    for obs in params.obstacles:
        if obs['type'] == 'sphere':
            obsCenters.append(obs['center'])
            obsRadii.append(obs['radius'])
        elif obs['type'] == 'box':
            discObs = discretize_box_surface(obs['center'], obs['dimensions'], params.maxRad)
            obsCenters.append(discObs)
            obsRadii.append(np.full(discObs.shape[0], 0.0))
    
    if len(obsCenters) > 0:
        obsCenters = np.vstack(obsCenters)
        obsRadii = np.concatenate(obsRadii)
    else:
        obsCenters = np.empty((0, 3))
        obsRadii = np.empty((0,))
    
    # *** OCPs DEFINITION ***
    ocp = define_ocp(model, params, safe_set)
    ocpSafeAbort = define_ocpSafeAbort(model, params)

    # *** SOLVERS CREATION ***
    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    solverSafeAbort = AcadosOcpSolver(ocpSafeAbort, json_file="acados_ocp_safe_abort.json")
    sim_solver = create_acados_sim(model, params)

    # *** VARIABLES DEFINITION ***
    x0_list = np.zeros((1, model.nx))  # ♟️ PLACEHOLDER: List of initial states (can be modified to include multiple initial conditions)

    xg_all = []
    ug_all = []

    for x0 in x0_list:
        x = x0
        xg = [x.copy()]
        ug = []

        x_prev = x
        u_prev = np.linalg.pinv(model.R(x).full() @ model.F) @ np.array([0, 0, params.mass * params.g]) / params.u_bar
        
        # Calculate initial box
        if len(obsCenters) > 0:
            Q = obsCenters - x[:3]
            x_min, x_max, y_min, y_max, z_min, z_max, _ , _ = min_cube_select(params, Q, obsRadii)
            box = np.array([x_min + x[0], x_max + x[0], 
                            y_min + x[1], y_max + x[1], 
                            z_min + x[2], z_max + x[2]])
        else:
            box = np.array([params.xlim[0], params.xlim[1],
                            params.ylim[0], params.ylim[1],
                            params.zlim[0], params.zlim[1]])

        # *** FIRST INITIALIZATION ***
        initialize_guess(
            solver,
            model,
            params,
            x,
            u_guess=u_prev,
            x_guess=x_prev,
            p_guess=box
        )

        # *** MPC LOOP ***

        fails = 0 
        follow_safe_abort = False

        for i in tqdm(range(params.time.shape[0]), desc="MPC Simulation Progress"): 
        
            solver.solve_for_x0(x, False, False)
            x_sol = np.array([solver.get(k, "x") for k in range(params.N + 1)])
            u_sol = np.array([solver.get(k, "u") for k in range(params.N)])

            feas = solver.get_status()

            if feas == 0:
                fails = 0
            else:
                if fails == 0:
                    print("MPC infeasibility detected.")
                    solverSafeAbort.solve_for_x0( x_prev[params.N-1] , False, False) # Why N-1?
                elif fails == params.N-1:
                    print("MPC infeasibility persists. Following safe abort strategy.")
                    follow_safe_abort = True
                    break
                    
                fails = fails + 1
                x_sol = x_prev.copy()
                u_sol = u_prev.copy()
                    
            x_next = dynamicsSim(sim_solver, x, u_sol[0], params.nsub)

            if len(obsCenters) > 0:
                Q = obsCenters - x_next[:3]
                x_min, x_max, y_min, y_max, z_min, z_max, _ , _ = min_cube_select(params, Q, obsRadii)
                box = np.array([x_min + x_next[0], x_max + x_next[0], 
                                y_min + x_next[1], y_max + x_next[1], 
                                z_min + x_next[2], z_max + x_next[2]])
            rollback_guess(solver, model, params, x_next, p_guess=box)

            x = x_next.copy()
            x_prev = x_sol.copy()
            u_prev = u_sol.copy()

            ug.append(u_sol[0])
            xg.append(x_next.copy())

        if follow_safe_abort:

            u_sol = np.array([solverSafeAbort.get(k, "u") for k in range(params.N)])
            
            for j in range(0, params.N):
                x_next = dynamicsSim(sim_solver, x, u_sol[j], params.nsub)

                x = x_next.copy()
                ug.append(u_sol[j])
                xg.append(x_next.copy())
        
        xg_all.append(np.asarray(xg))
        ug_all.append(np.asarray(ug))

    # *** DATA SAVING ***
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl")
    data = {
        "xg": xg_all,
        "ug": ug_all
    }

    with open(traj_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # *** RETRIEVE PARAMETERS AND MODEL ***
    params = Params()
    model = SthModel(params)

    # *** RUN MPC ***
    run_mpc(model, params)
    print("MPC simulation completed. Trajectory saved to 'trajectory.pkl'.")

    # *** PLOT DATA ***
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl")
    plotter(traj_path, model, params, animate=True)