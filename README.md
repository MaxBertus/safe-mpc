# STH-MPC

Nonlinear MPC for trajectory tracking and safe abort on the **aSTedH** (Star-shaped Tilted Hexarotor), with axis-aligned box obstacle avoidance and a neural-network viability kernel as terminal safe-set constraint.

OCP formulated with [CasADi](https://web.casadi.org/) and solved with [acados](https://docs.acados.org/). The NN terminal constraint is embedded via [l4casadi](https://github.com/Tim-Salzmann/l4casadi).

## Usage

```bash
python3 main.py
```

Set `mc_mode = True` in `Params` to run Monte Carlo simulations with randomised obstacles. All parameters are in the `Params` class at the top of `main.py`.

## Requirements

`acados`, `casadi`, `l4casadi`, `torch`, `numpy`, `scipy`, `tqdm`

---

Copyright © 2025. All rights reserved.