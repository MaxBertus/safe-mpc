import numpy as np
import casadi as ca
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from safe_mpc.plotter import plotter
import os
import pickle


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
        self.robot_name = "STH"

        # *** MPC PARAMETERS ***
        self.Q = 1e8 
        self.R = 1e-2
        self.N = 15

        # *** SIMULATION PARAMETERS ***
        self.Tf = 10.0
        self.dt = 0.005


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

        sin_a = np.sin(params.alpha_tilt)
        cos_a = np.cos(params.alpha_tilt)

        F = params.cf * np.array([
            [0,  np.sqrt(3)/2*sin_a, -np.sqrt(3)/2*sin_a, 0,  np.sqrt(3)/2*sin_a, -np.sqrt(3)/2*sin_a],
            [sin_a, -0.5*sin_a, -0.5*sin_a, sin_a, -0.5*sin_a, -0.5*sin_a],
            [cos_a]*6
        ])

        M = params.ct * np.array([
            [0,  np.sqrt(3)/2*self.r*cos_a,  np.sqrt(3)/2*self.r*cos_a, 0,
             -np.sqrt(3)/2*self.r*cos_a, -np.sqrt(3)/2*self.r*cos_a],
            [-self.r*cos_a, -0.5*self.r*cos_a, 0.5*self.r*cos_a,
              self.r*cos_a,  0.5*self.r*cos_a, -0.5*self.r*cos_a],
            [1, -1, 1, -1, 1, -1]
        ])

        fc = R @ (F @ u)
        tc = M @ u

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

        model = AcadosModel()
        model.name = params.robot_name
        model.x = x
        model.u = u
        model.f_expl_expr = f_expl

        self.amodel = model

        self.u_min = np.zeros(self.nu)
        self.u_max = params.u_bar * np.ones(self.nu)


# =========================================================
# MPC + SIMULATION
# =========================================================

def run_mpc(model, params):

    nx, nx_half, nu = model.nx, model.nx_half, model.nu
    ny = nx_half + nu
    ny_e = nx_half
    time = np.arange(0, params.Tf, params.dt)
    N = 15

    ocp = AcadosOcp()
    ocp.model = model.amodel
    ocp.dims.N = params.N
    ocp.solver_options.tf = params.Tf


    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Vx = np.zeros((ny, nx))
    Vx[:nx_half, :nx_half] = np.eye(nx_half)

    Vu = np.zeros((nx, nu))
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

    x_ref = np.array([0, 0, 1, 0, 0, 0])
    ocp.cost.yref = np.hstack((x_ref, np.zeros(nu)))
    ocp.cost.yref_e = x_ref

    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = model.u_min
    ocp.constraints.ubu = model.u_max
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    x = np.zeros(nx)
    xg = [x.copy()]
    ug = []

    for _ in range(time.shape[0]):
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)

        solver.solve()
        u = solver.get(0, "u")
        ug.append(u)

        xdot = model.f_expl_func(x, u).full().squeeze()
        x = x + params.dt * xdot
        xg.append(x.copy())

    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory.pkl")
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
    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory.pkl")
    plotter(traj_path, animate=False, Duration=params.Tf, dt=params.dt)