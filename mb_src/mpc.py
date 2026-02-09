import numpy as np
import casadi as ca
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from utils.plotter import plotter
import os
import pickle
from tqdm import tqdm

# =========================================================
# PARAMETERS
# =========================================================

class Params:
    def __init__(self):

        # *** MODEL PARAMETERS ***
        self.mass = 3.5
        self.J = np.diag([0.155, 0.147, 0.251])
        self.l = 0.385
        self.cf = 1.5e-3
        self.ct = 4.590e-5  
        self.u_bar = 108*108
        self.alpha_tilt = np.deg2rad(20)
        self.g = 9.81
        self.robot_name = "aSTedH"

        # *** MPC PARAMETERS ***
        self.Q = 5e6
        self.R = 1e-2
        self.N = 50

        # *** SIMULATION PARAMETERS ***
        self.SimDuration = 5.0
        self.dt = 0.01

        # *** REFERENCE STATE ***
        self.x_ref = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # *** ENVIRONMENT PARAMETERS ***
        self.obstacles = [
            {"center": np.array([0.0, 0.0, 0.5]), "radius": 0.05},
            # {"center": np.array([0.5, 0.5, 1.2]), "radius": 0.25},            
        ]

        self.state_constraint_active = True
        self.use_explicit_euler = False
        self.zlim = 2.0

# =========================================================
# MODEL GENERATION
# =========================================================

class SthModel:
    def __init__(self, params):
        self.r = params.cf / params.ct * params.l

        self.nx = 12
        self.nx_half = 6
        self.nu = 6

        x = MX.sym("x", self.nx)
        u = MX.sym("u", self.nu)

        p = x[0:3]
        rpy = x[3:6]
        v = x[6:9]
        omega = x[9:12]

        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

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

        sin_a = np.sin(params.alpha_tilt)
        cos_a = np.cos(params.alpha_tilt)
        tan_a = np.tan(params.alpha_tilt)

        self.F = params.cf * np.array([
        [0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a, 0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
        [sin_a, -1/2 * sin_a, -1/2 * sin_a, sin_a, -1/2 * sin_a, -1/2 * sin_a],
        [cos_a, cos_a, cos_a, cos_a, cos_a, cos_a]
        ])

        self.M = params.ct * np.array([
            [0, np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 0, -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a],
            [-self.r * cos_a + sin_a, -1/2 * self.r * cos_a + 1/2 * sin_a, 1/2 * self.r * cos_a - 1/2 * sin_a, self.r * cos_a - sin_a, 1/2 * self.r * cos_a - 1/2 * sin_a, -1/2 * self.r * cos_a + 1/2 * sin_a],
            [self.r * sin_a + cos_a, -self.r * sin_a - cos_a, self.r * sin_a + cos_a, -self.r * sin_a - cos_a, self.r * sin_a + cos_a, -self.r * sin_a - cos_a]
        ])

        fc = R @ (self.F @  u)
        tc = self.M @ u

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

        h_expr = []
        h_min = []
        h_max = []

        for obs in params.obstacles:
            c = obs["center"]
            r = obs["radius"]

            h = (p[0] - c[0])**2 + (p[1] - c[1])**2 + (p[2] - c[2])**2
            h_expr.append(h)

            h_min.append(r**2)
            h_max.append(1e10)

        self.h_expr = vertcat(*h_expr)
        self.h_min = np.array(h_min)
        self.h_max = np.array(h_max)

        model = AcadosModel()
        model.name = params.robot_name
        model.x = x
        model.u = u
        model.f_expl_expr = f_expl

        self.amodel = model

        self.u_min = np.zeros(self.nu)
        self.u_max = params.u_bar * np.ones(self.nu)

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

    N = params.N
    nx = model.nx
    nu = model.nu

    # Default guesses
    if u_guess is None:
        u_guess = np.zeros(nu)

    if x_guess is None:
        x_guess = x0.copy()

    for k in range(N):
        if u_guess.ndim == 1:
            solver.set(k, "u", u_guess)
        else:
            solver.set(k, "u", u_guess[k])

        if x_guess.ndim == 1:
            solver.set(k, "x", x_guess)
        else:
            solver.set(k, "x", x_guess[k])

    # Terminal state
    if x_guess.ndim == 1:
        solver.set(N, "x", x_guess)
    else:
        solver.set(N, "x", x_guess[N])

# =========================================================
# MPC + SIMULATION
# =========================================================

def run_mpc(model, params):

    nx, nx_half, nu = model.nx, model.nx_half, model.nu
    ny = nx_half + nu
    ny_e = nx_half
    time = np.arange(0, params.SimDuration, params.dt)

    ocp = AcadosOcp()
    ocp.model = model.amodel
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.N * params.dt

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Vx = np.zeros((ny, nx))
    Vx[:nx_half, :nx_half] = np.eye(nx_half)

    Vu = np.zeros((ny, nu))
    Vu[nx_half:, :] = np.eye(nu)

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:, :nx_half] = np.eye(nx_half)

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e

    Q = params.Q* np.eye(nx_half)
    R = params.R * np.eye(nu)

    ocp.cost.W = np.block([[Q, np.zeros((nx_half, nu))],
                            [np.zeros((nu, nx_half)), R]])
    ocp.cost.W_e = Q

    x_ref = params.x_ref
    ocp.cost.yref = np.hstack((x_ref, np.zeros(nu)))
    ocp.cost.yref_e = x_ref

    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = np.zeros(nx)  

    ocp.model.con_h_expr_0 = model.h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max
    
    if params.state_constraint_active:
        ocp.constraints.ubx = np.array([params.zlim])  
        ocp.constraints.lbx = np.array([0.0]) 
        ocp.constraints.idxbx = np.array([2])

        ocp.constraints.ubx_e = np.array([params.zlim])  
        ocp.constraints.lbx_e = np.array([0.0]) 
        ocp.constraints.idxbx_e = np.array([2])

    ocp.model.con_h_expr_0 = model.h_expr
    ocp.constraints.lh_0 = model.h_min
    ocp.constraints.uh_0 = model.h_max

    ocp.model.con_h_expr = model.h_expr
    ocp.constraints.lh = model.h_min
    ocp.constraints.uh = model.h_max

    ocp.model.con_h_expr_e = model.h_expr
    ocp.constraints.lh_e = model.h_min
    ocp.constraints.uh_e = model.h_max

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP" 
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.nlp_solver_max_iter = 200

    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    x = np.zeros(nx)
    xg = [x.copy()]
    ug = []

    u_prev = np.linalg.pinv(model.R(x).full() @ model.F) @ np.array([0, 0, params.mass * params.g])
    x_prev = x

    for ist in tqdm(range(time.shape[0]), desc="MPC Simulation Progress"):
    # for _ in range(1,2):    

        initialize_guess(
            solver,
            model,
            params,
            x,
            u_guess=u_prev,
            x_guess=x_prev
        )


        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)

        # solver.set(0, "x", x)

        solver.solve()

        # for k in range(0, params.N+1):
        #     print(f"{ist*params.dt:.3f}: z at node {k} {solver.get(k, 'x')[2]}")

        # print("----")

        status = solver.get_status()

        # # *** DEBUGGING INFO ***
        # print(f"Solver status: {status}")

        # if status == 0:
        #     # Verifica se i vincoli sono attivi
        #     print(f"Solver status - again: {status}")
        #     for k in range(params.N):
        #         x_k = solver.get(k, "x")
        #         p_k = x_k[0:3]
                
        #         for i, obs in enumerate(params.obstacles):
        #             dist = np.linalg.norm(p_k - obs["center"])
        #             constraint_value = dist**2
        #             print(f"Step {k}, Obs {i}: dist={dist:.4f}, h={constraint_value:.4f}, h_min={obs['radius']**2:.4f}, violated={constraint_value < obs['radius']**2}")

        #     # Verifica i moltiplicatori di Lagrange
        #     # Se sono zero, i vincoli non sono attivi
        #     residuals = solver.get_residuals()
        #     print(f"Residuals: {residuals}")
        #     # *** FINE DEBUGGING INFO ***

        u_prev = np.array([solver.get(k, "u") for k in range(params.N)])
        x_prev = np.array([solver.get(k, "x") for k in range(params.N + 1)])

        u = u_prev[0]
        ug.append(u)

        if params.use_explicit_euler:
            xdot = model.f_expl_func(x, u).full().squeeze()
            x = x + params.dt * xdot
        else:
            x = solver.get(1, "x")
        xg.append(x.copy())

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

    params = Params()
    model = SthModel(params)

    run_mpc(model, params)

    print("MPC simulation completed. Trajectory saved to 'trajectory.pkl'.")
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl")
    plotter(traj_path, model, params, animate=False)