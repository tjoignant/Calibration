import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

import cev
import black_scholes

# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 300)
pd.set_option('display.expand_frame_repr', False)

# Set Numerical Differential Step
step = 0.1

# Set Market Options DataFrame
df_mkt = pd.DataFrame()
df_mkt["Strike"] = np.arange(95, 105, 1)
df_mkt["Price (3M)"] = [8.67, 7.14, 5.98, 4.93, 4.09, 3.99, 3.43, 3.01, 2.72, 2.53]
df_mkt["Price (6M)"] = [10.71, 8.28, 6.91, 6.36, 5.29, 5.07, 4.76, 4.47, 4.35, 4.14]
df_mkt["Price (9M)"] = [11.79, 8.95, 8.07, 7.03, 6.18, 6.04, 5.76, 5.50, 5.50, 5.39]
df_mkt["Price (12M)"] = [12.40, 9.59, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]

# Compute Option's Implied Volatility (IV)
for maturity in [3, 6, 9, 12]:
    df_mkt[f"IV ({maturity}M)"] = df_mkt.apply(
        lambda x: black_scholes.BS_IV_Newton_Raphson(
            MktPrice=x[f"Price ({maturity}M)"], df=1, f=100, k=x["Strike"], t=maturity / 12, OptType="C")[0], axis=1)

# ----------------------------------------- PART 3 : OPTIMISATION ALGORITHMS ------------------------------------------
# Set Estimation Inputs
mktPrice_list = list(df_mkt["Price (3M)"]) + list(df_mkt["Price (6M)"]) + list(df_mkt["Price (9M)"]) \
                + list(df_mkt["Price (12M)"])
strike_list = list(df_mkt["Strike"]) + list(df_mkt["Strike"]) + list(df_mkt["Strike"]) + list(df_mkt["Strike"])
maturity_list = [3 / 12 for _ in list(df_mkt["Price (3M)"])] + [6 / 12 for _ in list(df_mkt["Price (6M)"])] + \
                [9 / 12 for _ in list(df_mkt["Price (9M)"])] + [12 / 12 for _ in list(df_mkt["Price (12M)"])]
inputs_list = []
for i in range(0, len(mktPrice_list)):
    inputs_list.append((strike_list[i], maturity_list[i], 100, 1, "C"))

# Set PSO Inputs
iters = 30
particles = 10

# Estimate CEV Params (Scipy)
params, error, nit = cev.CEV_calibration_scipy(inputs_list=inputs_list, mktPrice_list=mktPrice_list)
print("\nScipy (Nelder-Mead)")
print(f" - [gamma, sigma0]: {params}")
print(f" - MSE: {error}")
print(f" - nb iterations: {nit}")

# Estimate CEV Params (Nelder-Mead)
params, error, nit = cev.CEV_calibration_nelder_mead(inputs_list=inputs_list, mktPrice_list=mktPrice_list)
print("\nNelder-Mead")
print(f" - [gamma, sigma0]: {params}")
print(f" - MSE: {error}")
print(f" - nb iterations: {nit}")

# Estimate CEV Params (PSO)
params, error, nit = cev.CEV_calibration_pso(max_iter=iters, n=particles, dim=2, lowx=0.01, uppx=0.99, inputs_list=inputs_list, mktPrice_list=mktPrice_list)
print("\nParticle Swarm Optimization")
print(f" - [gamma, sigma0]: {params}")
print(f" - MSE: {error}")
print(f" - nb iterations: {nit}")

# Display Graphs
plt.show()
