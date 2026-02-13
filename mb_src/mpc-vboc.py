import numpy as np
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
            #{"center": np.array([0.0, 0.0, 2.0]), "dimensions": np.array([0.5, 0.5, 0.5]), "type": "box"},
            {"center": np.array([0.0, 0.0, 1.5]), "radius": 0.2, "type": "sphere"},    
        ]
        
        # Room dimensions
        self.state_constraint_active = True
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

        # *** OBSTACLE CONSTRAINTS DEFINITION ***
        h_expr = []
        h_min = []
        h_max = []

        for obs in params.obstacles:

            if obs["type"] == "sphere":

                c = obs["center"]
                r = obs["radius"]

                h = (p[0] - c[0])**2 + (p[1] - c[1])**2 + (p[2] - c[2])**2
                h_expr.append(h)

                h_min.append((r+params.maxRad)**2)
                h_max.append(1e10)

            elif obs["type"] == "box":

                c = obs["center"]
                d = obs["dimensions"]

                # semi-dimensions
                b = d / 2.0

                # signed distance function (SDF) for axis-aligned box
                q = ca.fabs(p - c) - b

                # outside distance
                outside = ca.fmax(q, 0)
                outside_dist = ca.norm_2(outside)

                # inside distance
                inside_dist = ca.fmin(ca.fmax(q[0], ca.fmax(q[1], q[2])), 0)

                # signed distance
                dist = outside_dist + inside_dist

                # safety margin for drone radius
                h = dist - params.maxRad

                h_expr.append(h)

                # enforce h >= 0
                h_min.append(0.0)
                h_max.append(1e10)

        self.h_expr = vertcat(*h_expr)
        self.h_min = np.array(h_min)
        self.h_max = np.array(h_max)

        self.h_func = ca.Function("h_func", [x], [self.h_expr])

        # *** INPUT CONSTRAINTS DEFINITION ***
        self.u_min = np.zeros(self.nu)
        self.u_max = np.ones(self.nu)

        # *** PARAMETERS ***
        box = ca.MX.sym("b", 6)

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
    ocp.constraints.x0 = np.zeros(nx)  
    
    # State
    if params.state_constraint_active:
        ocp.constraints.lbx = np.array([params.xlim[0]+params.maxRad, params.ylim[0]+params.maxRad, params.zlim[0]+params.maxRad])  
        ocp.constraints.ubx = np.array([params.xlim[1]-params.maxRad, params.ylim[1]-params.maxRad, params.zlim[1]-params.maxRad])  
        ocp.constraints.idxbx = np.arange(3)

        ocp.constraints.lbx_e = np.array([params.xlim[0]+params.maxRad, params.ylim[0]+params.maxRad, params.zlim[0]+params.maxRad])  
        ocp.constraints.ubx_e = np.array([params.xlim[1]-params.maxRad, params.ylim[1]-params.maxRad, params.zlim[1]-params.maxRad])  
        ocp.constraints.idxbx_e = np.arange(3)

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
    
    # State
    idx_pos = np.arange(0, 3)     # positions
    idx_vel = np.arange(6, 12)    # velocities

    # position bounds (only if active)
    if params.state_constraint_active:

        lb_pos = np.array([
            params.xlim[0] + params.maxRad,
            params.ylim[0] + params.maxRad,
            params.zlim[0] + params.maxRad
        ])

        ub_pos = np.array([
            params.xlim[1] - params.maxRad,
            params.ylim[1] - params.maxRad,
            params.zlim[1] - params.maxRad
        ])

        ocp.constraints.lbx = lb_pos
        ocp.constraints.ubx = ub_pos
        ocp.constraints.idxbx = idx_pos

        # velocities must be zero
        lb_vel = np.zeros(6)
        ub_vel = np.zeros(6)

        # concatenate
        ocp.constraints.idxbx_e = np.concatenate((idx_pos, idx_vel))
        ocp.constraints.lbx_e = np.concatenate((lb_pos, lb_vel))
        ocp.constraints.ubx_e = np.concatenate((ub_pos, ub_vel))

    else:
        # only velocity constraint
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

        x_cp = model.x
        p_cp = model.p

        # Standardize box dimensions
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
# MPC + SIMULATION
# =========================================================
def run_mpc(model, params):
    '''Compute, apply and simulate MPC control.'''
    
    # Define safe set
    safe_set = NetSafeSet(model, params)

    # *** OCPs DEFINITION ***
    ocp = define_ocp(model, params, safe_set)
    ocpSafeAbort = define_ocpSafeAbort(model, params)

    # *** SOLVERS CREATION ***
    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    solverSafeAbort = AcadosOcpSolver(ocpSafeAbort, json_file="acados_ocp_safe_abort.json")
    sim_solver = create_acados_sim(model, params)

    # *** VARIABLES DEFINITION ***
    x = np.zeros(model.nx)
    xg = [x.copy()]
    ug = []

    x_prev = x
    u_prev = np.linalg.pinv(model.R(x).full() @ model.F) @ np.array([0, 0, params.mass * params.g]) / params.u_bar
    
    boxes = np.zeros(6) # Placeholder for box dimensions, to be updated if needed

    # *** FIRST INITIALIZATION ***
    initialize_guess(
        solver,
        model,
        params,
        x,
        u_guess=u_prev,
        x_guess=x_prev,
        p_guess=boxes
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
        rollback_guess(solver, model, params, x_next, p_guess=boxes)

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
        
    # *** DATA SAVING ***
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl")
    data = {
        "xg": np.asarray(xg),
        "ug": np.asarray(ug)
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