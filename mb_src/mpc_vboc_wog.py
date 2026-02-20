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
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver

from safe_mpc import ocp
from utils.plotter import plotter

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
        self.Qv = np.diag([1e2, 1e2, 1e2,   # linear velocities 
                           1e2, 1e2, 1e2])  # angular velocities
        self.R = 1e0 * np.eye(self.nu)
        self.N = 60
        self.Nvboc = 50
        self.nlp_solver_max_iter = 100

        # *** SIMULATION PARAMETERS ***
        self.SimDuration = 5.0  
        self.dt = 0.02
        self.dtSim = 1e-4
        self.nsub = int( self.dt / self.dtSim )  # Number of simulation steps for each control steps
        self.T = int(self.SimDuration / self.dt)
        self.time = np.arange(0, self.SimDuration, self.dt)

        # *** REFERENCE STATE ***
        self.x_ref = np.array([2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.use_u_ref_hovering = True

        # *** ENVIRONMENT PARAMETERS ***
        # Obstacles
        self.obstacles = [
            {"center": np.array([2.0, 0.0, 0.5]), "dimensions": np.array([0.5, 2.0, 2.0]), "type": "box"},
            {"center": np.array([-1.5, 0.0, 1.0]), "dimensions": np.array([0.5, 3.0, 3.0]), "type": "box"},
            {"center": np.array([0.0, 0.0, 3.0]), "dimensions": np.array([2.0, 2.0, 0.5]), "type": "box"},
            # {"center": np.array([1.5, 0.0, 0.5]), "radius": 0.5, "type": "sphere"},    
            # {"center": np.array([-1.5, 0.0, 0.5]), "radius": 0.5, "type": "sphere"}, 
            # {"center": np.array([0.0, 0.0, 3.5]), "radius": 0.5, "type": "sphere"},                         
        ]
        
        # Room dimensions
        self.xlim = [-3.0, 3.0]
        self.ylim = [-3.0, 3.0] 
        self.zlim = [-3.0, 5.0]

        # *** NEURAL NETWORK PARAMETERS ***
        self.input_size = 15 # 6 box dimensions + 3 orientations + 3 linear velocities + 3 angular velocities = 15
        self.hidden_size = 512
        self.output_size = 1
        self.number_hidden = 2
        self.act_fun = torch.nn.GELU(approximate='tanh')
        self.net_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"nn/sth_gelu.pt")
        self.build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn")
        self.eps = 1e-6
        self.alpha = 5
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0') 
        else:
            self.device = torch.device('cpu')

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
        
        self.h_func = ca.Function('h_func', [x, box], [h_box])
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
def initialize_guess(solver, N, model, params, x0, u_guess=None, x_guess=None, p_guess=None):
    """
    Initialize the OCP with a state and control guess.

    x0      : initial state (nx,)
    u_guess : control guess (nu,) or (N, nu)
    x_guess : state guess (nx,) or (N+1, nx)
    p_guess : parameter guess (np.ndarray of shape (n_p,))
    """

    # *** DIMENSIONS DEFINITION ***
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

    sim.parameter_values = np.zeros(6)

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
        params.N,
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
    ocp.model = copy.deepcopy(model.amodel)
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.N * params.dt
    ocp.model.name = params.robot_name + "_tracking"

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.parameter_values = np.zeros(6)

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
    ocp.constraints.x0 = np.zeros(nx)
    
    # Obstacles 
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

    # Neural network safe set constraint
    nn_expr = safe_set.nn_func(model.amodel.x, model.amodel.p)

    ocp.model.con_h_expr_e = vertcat(h_expr, nn_expr)
    ocp.constraints.lh_e = np.zeros((7,1))
    ocp.constraints.uh_e = np.full((7,1), 1e6)


    # *** SOLVER PARAMETERS DEFINITION ***
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP" 

    ocp.solver_options.nlp_solver_max_iter = params.nlp_solver_max_iter


    # Link the l4casadi shared library so acados can resolve "aSTedH_model"
    ocp.solver_options.custom_templates = []

    # Add the l4casadi lib path to the linker
    lib_name = f'{params.robot_name}_model'
    ocp.solver_options.ext_fun_compile_flags = f'-I{params.build_dir}'
    ocp.solver_options.link_libs = f'-L{params.build_dir} -l{lib_name} -Wl,-rpath,{params.build_dir}'


    return ocp

def define_ocpSafeAbort(model, params):
    # *** PROBLEM DIMENSIONS DEFINITION ***
    nx, nu = model.nx, model.nu
    nv = 6                     # number of velocity states
    ny = nv + nu               # velocities + inputs

    # *** OCP OBJECT CREATION ***
    ocp = AcadosOcp()
    ocp.model = copy.deepcopy(model.amodel)
    ocp.dims.N = params.Nvboc
    ocp.solver_options.tf = params.Nvboc * params.dt
    ocp.model.name = params.robot_name + "_safe_abort"

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.parameter_values = np.zeros(6)

    # *** COST FUNCTION DEFINITION ***
    Vx = np.zeros((ny, nx))
    Vu = np.zeros((ny, nu))

    # Select only velocities (states 6:12)
    Vx[0:nv, nv:nx] = np.eye(nv)

    # Select inputs
    Vu[nv:, :] = np.eye(nu)

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu

    # Terminal cost: only velocities
    Vx_e = np.zeros((nv, nx))
    Vx_e[:, nv:nx] = np.eye(nv)
    ocp.cost.Vx_e = Vx_e

    # Weights
    Qv = params.Qv              # 6x6 velocity weight matrix
    R = params.R                # nu x nu input weight matrix

    W = np.zeros((ny, ny))
    W[:nv, :nv] = Qv
    W[nv:, nv:] = R

    ocp.cost.W = W
    ocp.cost.W_e = Qv

    # Hovering reference input
    if params.use_u_ref_hovering:
        u_hover = np.linalg.pinv(
            model.R(np.zeros(nx)).full() @ model.F
        ) @ np.array([0, 0, params.mass * params.g]) / params.u_bar
    else:
        u_hover = np.zeros(nu)

    # Reference: zero velocities, hovering input
    yref = np.zeros(ny)
    yref[nv:] = u_hover

    ocp.cost.yref = yref
    ocp.cost.yref_e = np.zeros(nv)

    # *** CONSTRAINTS DEFINITION ***
    ocp.constraints.constr_type = "BGH"

    # Input bounds
    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)
    
    # Initial state constraint
    ocp.constraints.x0 = np.zeros(nx)  
    
    # # Terminal velocity constraint (optional but kept as in original)
    # idx_vel = np.arange(6, 12)
    # ocp.constraints.idxbx_e = idx_vel
    # ocp.constraints.lbx_e = np.zeros(6)
    # ocp.constraints.ubx_e = np.zeros(6)

    # Obstacles 
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

    # *** SOLVER PARAMETERS DEFINITION ***
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP" 

    ocp.solver_options.nlp_solver_max_iter = params.nlp_solver_max_iter

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

        nn_data = torch.load(params.net_path, weights_only=False)

        model_net = NeuralNetwork(params.input_size, params.hidden_size, params.output_size, params.number_hidden, params.act_fun, 1).to(params.device)
        model_net.load_state_dict(nn_data['model'])

        x_cp = model.amodel.x
        p_cp = model.amodel.p

        # Translate box to robot frame, reorder to [mins, maxs] and standardize (to follow VBOC input format)
        box_in_robot_frame = p_cp[[0, 2, 4, 1, 3, 5]] - ca.vertcat(x_cp[0], x_cp[1], x_cp[2], x_cp[0], x_cp[1], x_cp[2])
        # Clip lower bounds: box_min distances cannot exceed room limits (negated because robot-relative)
        room_lower = ca.DM([-2.0, -2.0, -2.0])
        room_upper = ca.DM([2.0, 2.0, 2.0])

        box_lower = box_in_robot_frame[:3]
        box_upper = box_in_robot_frame[3:]

        # Apply element-wise clipping using CasADi symbolic ops
        box_in_robot_frame[:3] = -ca.fmax(box_lower, room_lower)  
        box_in_robot_frame[3:] =  ca.fmin( box_upper,  room_upper)  
        box = (box_in_robot_frame - nn_data['mean']) / nn_data['std']

        # Standardize initial orientation
        orient = (x_cp[npos:npos+nori] - nn_data['mean']) / nn_data['std']

        # Normalize velocities            
        # vel_norm = ca.norm_2(x_cp[npos+nori:])
        vel_norm = ca.sqrt(ca.dot(x_cp[npos+nori:], x_cp[npos+nori:]) + params.eps**2)
        vel_dir = x_cp[npos+nori:] / (vel_norm) 

        state = ca.vertcat(box, orient, -vel_dir)

        self.l4c_model = l4c.L4CasADi(model_net,
                                      device=params.device,
                                      name=f'{params.robot_name}_model',
                                      build_dir=params.build_dir)
    
        nn_model = self.l4c_model(state) * (100 - params.alpha) / 100 - vel_norm
       
        self.nn_func = ca.Function('nn_func', [model.amodel.x, model.amodel.p], [nn_model])

# =========================================================
# BOX CONSTRAINTS DEFINITION
# =========================================================

def min_cube_select(Q=None, R=None, goal_point=None, drone_radius=0.5):
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

    max_iter = 100
    tol = 1e-10

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

    # inside = (
    #     (Q[:, 0] - R >= xOpt[0]) & (Q[:, 0] + R <= xOpt[1]) &
    #     (Q[:, 1] - R >= xOpt[2]) & (Q[:, 1] + R <= xOpt[3]) &
    #     (Q[:, 2] - R >= xOpt[4]) & (Q[:, 2] + R <= xOpt[5])
    # )
    # all_outside = not np.any(inside)

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
        exitflag,
        goal_included
    )

def _spheres_intersect_box(Q, R, box, tol=1e-9):
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

    # result = safe_set.nn_func(np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0, 0.0,0.0,0.0]), np.array([-2, 2, -2, 2, -2, 2]))
    # print("NN output at origin with box [-2,2] in all dimensions:", result)

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
            obsRadii.append(np.full(discObs.shape[0], 0.01))
    
    if len(obsCenters) > 0:
        obsCenters = np.vstack(obsCenters)
        if obs['type'] == 'sphere':
            obsRadii = np.array(obsRadii)
        elif obs['type'] == 'box':
            obsRadii = np.concatenate(obsRadii)
    else:
        obsCenters = np.empty((0, 3))
        obsRadii = np.empty((0,))

    # *** OCPs DEFINITION ***
    ocp = define_ocp(model, params, safe_set)
    ocpSafeAbort = define_ocpSafeAbort(model, params)

    # *** SOLVERS CREATION ***
    ctypes.CDLL(os.path.join(params.build_dir, f'lib{params.robot_name}_model.so'), mode=ctypes.RTLD_GLOBAL)

    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    solverSafeAbort = AcadosOcpSolver(ocpSafeAbort, json_file="acados_ocp_safe_abort.json")
    sim_solver = create_acados_sim(model, params)

    # *** VARIABLES DEFINITION ***
    x0_list = np.zeros((1, model.nx))  # ♟️ PLACEHOLDER: List of initial states (can be modified to include multiple initial conditions)

    xg_all = []
    ug_all = []
    bg_all = []

    for x0 in x0_list:
        x = x0
        xg = [x.copy()]
        ug = []
        bg = []

        x_prev = np.full((params.N + 1, model.nx), x)
        u0 = np.linalg.pinv(model.R(x).full() @ model.F) @ np.array([0, 0, params.mass * params.g]) / params.u_bar
        u_prev = np.full((params.N, model.nu), u0)

        # Calculate initial box
        if len(obsCenters) > 0:
            discObsPoints = obsCenters - x[:3]
            x_min, x_max, y_min, y_max, z_min, z_max, _ , _ = min_cube_select(discObsPoints, obsRadii,drone_radius=params.maxRad)

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
            params.N,
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
        u_safe_abort = None
        ttot = 0.0
        tmax = 0.0
        tmin = params.dt * 10

        for i in tqdm(range(params.time.shape[0]), desc="MPC Simulation Progress"): 
        
            # Print actual x and box for debugging
            # print(f"Current state x: {x}")
            # print(f"Current box: {box}")

            # Test NN output at current state and box for debugging
            # result = safe_set.nn_func(x, box)
            # print("NN output at current state and box:", result)

            t0 = time.perf_counter()
            
            solver.solve_for_x0(x, False, False)
            
            t1 = (time.perf_counter()-t0)*10
            if(t1 > tmax):
                tmax = t1
            if(t1 < tmin ):
                tmin = t1
            
            ttot = 1 + ttot

            x_sol = np.array([solver.get(k, "x") for k in range(params.N + 1)])
            u_sol = np.array([solver.get(k, "u") for k in range(params.N)])

            # Do previous debug with last predicted state and box for comparison
            # print(f"Predicted next state x_sol[1]: {x_sol[params.N]}") 
            # result = safe_set.nn_func(x_sol[params.N], box)
            # print("NN output at predicted next state and box:", result)

            feas = solver.get_status()

            if feas == 0:
                fails = 0
                u_to_apply = u_sol[0]
                x_prev = x_sol.copy()
                u_prev = u_sol.copy()
            else:
                if fails == 0:
                    print("Alert: MPC infeasibility detected.")

                    x_safe_start = x_prev[params.N, :]
                    u_hover = np.linalg.pinv(
                                model.R(x_safe_start).full() @ model.F
                            ) @ np.array([0, 0, params.mass * params.g]) / params.u_bar

                    # Initialize guess for safe abort
                    x_guess_abort = np.tile(x_safe_start, (params.Nvboc + 1, 1))
                    u_guess_abort = np.tile(u_hover, (params.Nvboc, 1))

                    initialize_guess(
                        solverSafeAbort,
                        params.Nvboc,
                        model,
                        params,
                        x_safe_start,
                        u_guess=u_guess_abort,
                        x_guess=x_guess_abort,
                        p_guess=box
                    )

                    solverSafeAbort.solve_for_x0( x_safe_start , False, False)
                    u_safe_abort = np.array([
                        solverSafeAbort.get(k, "u") for k in range(params.Nvboc)
                    ])
                
                if fails == params.N:
                    print(f"Switching to safe abort trajectory at t={i*params.dt:.2f}s")
                    follow_safe_abort = True
                    break
                    
                u_to_apply = u_prev[fails]
                fails += 1
                    
            x_next = dynamicsSim(sim_solver, x, u_to_apply, params.nsub)

            if len(obsCenters) > 0:
                discObsPoints = obsCenters - x_next[:3]
                x_min, x_max, y_min, y_max, z_min, z_max, boxFeasible , _ = min_cube_select(discObsPoints, obsRadii, drone_radius=params.maxRad)

                if boxFeasible:
                    box = np.array([x_min + x_next[0], x_max + x_next[0], 
                                    y_min + x_next[1], y_max + x_next[1], 
                                    z_min + x_next[2], z_max + x_next[2]])

            if feas == 0:
                rollback_guess(solver, model, params, x_next, p_current=box)
            else:
                # Shift manuale di u_prev/x_prev senza leggere dal solver infeasible
                u_shifted = np.vstack([u_prev[1:], u_prev[-1]])
                x_shifted = np.vstack([x_prev[1:], x_prev[-1]])
                initialize_guess(
                    solver, params.N, model, params,
                    x_next,
                    u_guess=u_shifted,
                    x_guess=x_shifted,
                    p_guess=box
                )

            x = x_next.copy()
            ug.append(u_to_apply)
            xg.append(x_next.copy())
            bg.append(box.copy())

        print(f"Avarage time for each iteration: {ttot/i:.2f} ms, max: {tmax:.2f} ms, min: {tmin:.2f}")

        if follow_safe_abort:

            print(f"x_safe_start was: {x_prev[params.N, :]}")
            print(f"x actual after {fails} fallback steps: {x}")
            print(f"Difference: {x - x_prev[params.N, :]}")

            input("press to continue")

            for j in range(0, params.Nvboc):
                x_next = dynamicsSim(sim_solver, x, u_safe_abort[j], params.nsub)
                x = x_next.copy()
                ug.append(u_safe_abort[j])
                xg.append(x_next.copy())
                bg.append(box.copy())
                
        xg_all.append(np.asarray(xg))
        ug_all.append(np.asarray(ug))
        bg_all.append(np.asarray(bg))

    # *** DATA SAVING ***
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl")
    data = {
        "xg": xg_all,
        "ug": ug_all,
        "bg": bg_all,
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