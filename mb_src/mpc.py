import numpy as np
import casadi as ca
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross, fmin, fmax
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
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

        # *** ACADOS MODEL CREATION ***
        model = AcadosModel()
        model.name = params.robot_name
        model.x = x
        model.u = u
        model.f_expl_expr = f_expl

        self.amodel = model

# =========================================================
# INITIAL GUESS
# =========================================================
def initialize_guess(solver, model, params, x0, u_guess=None, x_guess=None):
    """
    Initialize the OCP with a state and control guess.

    x0      : initial state (nx,)
    u_guess : control guess (nu,) or (N, nu)
    x_guess : state guess (nx,) or (N+1, nx)
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
def rollback_guess(solver, model, params, x_current):
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
        x_guess=x_guess
    )

    # *** RETURN FIRST CONTROL INPUT ***
    return u_sol[0]

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
# MPC + SIMULATION
# =========================================================
def run_mpc(model, params):
    '''Compute, apply and simulate MPC control.'''
    
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

    # *** SOLVER PARAMETERS DEFINITION ***
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP" 

    ocp.solver_options.nlp_solver_max_iter = 500

    # *** SOLVERS CREATION ***
    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    sim_solver = create_acados_sim(model, params)

    # *** VARIABLES DEFINITION ***
    x = np.zeros(nx)
    xg = [x.copy()]
    ug = []

    x_prev = x
    u_prev = np.linalg.pinv(model.R(x).full() @ model.F) @ np.array([0, 0, params.mass * params.g]) / params.u_bar
    
    # *** FIRST INITIALIZATION ***
    initialize_guess(
        solver,
        model,
        params,
        x,
        u_guess=u_prev,
        x_guess=x_prev
    )

    # *** MPC LOOP ***
    for ist in tqdm(range(params.time.shape[0]), desc="MPC Simulation Progress"): 

        solver.solve_for_x0(x, False, False)

        u = rollback_guess(solver, model, params, x)
        x = dynamicsSim(sim_solver, x, u, params.nsub)

        ug.append(u)
        xg.append(x.copy())

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