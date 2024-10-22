import time
import pickle
import adam.numpy
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from urdf_parser_py.urdf import URDF
import adam
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel, TriplePendulumModel
from safe_mpc.utils import obstacles, ee_ref, get_ocp


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=False)
params.act = args['activation']
if model_name == 'triple_pendulum':
    model = TriplePendulumModel(params)
else:
    model = AdamModel(params, n_dofs=4)
    model.ee_ref = ee_ref

ocp_name = args['controller']
ocp = get_ocp(ocp_name, model, obstacles)
opti = ocp.instantiateProblem()

num_ics = params.test_num
succ, fails, skip_ics = 0, 0, 0
sampler = qmc.Halton(model.nq, scramble=False)
x_guess, u_guess = [], []
progress_bar = tqdm(total=num_ics, desc='Generating initial conditions')

start_time = time.time()
while succ < num_ics:
    q0 = qmc.scale(sampler.random(), model.x_min[:model.nq], model.x_max[:model.nq])[0]
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = q0

    if ocp.checkCollision(x0):
        u0 = np.zeros((model.nu,)) if model_name == 'triple_pendulum' else model.gravity(np.eye(4), q0)[6:]

        opti.set_value(ocp.x_init, x0)
        for k in range(params.N):
            opti.set_initial(ocp.X[k], x0)
            opti.set_initial(ocp.U[k], u0)
        opti.set_initial(ocp.X[-1], x0)

        try:
            sol = opti.solve()
            xg = np.array([sol.value(ocp.X[k]) for k in range(params.N + 1)])
            ug = np.array([sol.value(ocp.U[k]) for k in range(params.N)])
            x_guess.append(xg), u_guess.append(ug)
            succ += 1
            progress_bar.update(1)
        except:
            sol = opti.debug
            fails += 1
    else:
        skip_ics += 1

progress_bar.close()
print(f'Number of failed initializations: {fails}')
print(f'Number of skipped initial conditions: {skip_ics}')

x_guess = np.asarray(x_guess)
u_guess = np.asarray(u_guess)

with open(f'{params.DATA_DIR}{model_name}_{ocp_name}_guess.pkl', 'wb') as f:
    pickle.dump({'xg': x_guess, 'ug': u_guess}, f)

elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')
