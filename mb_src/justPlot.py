import numpy as np
import casadi as ca
from casadi import MX, vertcat, horzcat, sin, cos, tan, cross
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from utils.plotter import plotter
from mpc import Params, SthModel
import os
import pickle

if __name__ == "__main__":

    params = Params()
    model = SthModel(params)

    traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/trajectory.pkl")
    plotter(traj_path, model, params, animate=False)