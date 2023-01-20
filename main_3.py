import numpy as np
import pandas as pd
import time

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

# Estimate CEV Params (Scipy)
start = time.perf_counter()
params, error, nit = cev.CEV_calibration_scipy(inputs_list=inputs_list, mktPrice_list=mktPrice_list)
end = time.perf_counter()
print("\nScipy (Nelder-Mead)")
print(f" - [gamma, sigma0]: {params}")
print(f" - MSE: {error}")
print(f" - nb iterations: {nit}")
print(f" - optimum reached ({round(end - start, 1)}s)")


# Estimate CEV Params (Nelder-Mead)
start = time.perf_counter()
params_nm, error_nm, nit_nm, hist_error_nm = cev.CEV_calibration_nelder_mead(inputs_list=inputs_list, mktPrice_list=mktPrice_list)
end = time.perf_counter()
print("\nNelder-Mead")
print(f" - [gamma, sigma0]: {params_nm}")
print(f" - MSE: {error_nm}")
print(f" - nb iterations: {nit_nm}")
print(f" - optimum reached ({round(end - start, 1)}s)")

# Set PSO Inputs
particles = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
iteration_list = []
error_list = []
# Estimate CEV Params (PSO)
for nb_particle in particles:
    start = time.perf_counter()
    params_pso, error_pso, nit_pso, hist_error_pso = cev.CEV_calibration_pso(n=nb_particle, dim=2, lowx=0.01, uppx=0.99,
                                                             inputs_list=inputs_list, mktPrice_list=mktPrice_list)
    end = time.perf_counter()
    print("\nParticle Swarm Optimization")
    print(f" - [gamma, sigma0]: {params_pso}")
    print(f" - MSE: {error_pso}")
    print(f" - nb iterations: {nit_pso}")
    print(f" - optimum reached ({round(end - start, 1)}s)")
    print(f" - nb particles: {nb_particle}")
    iteration_list.append(nit_pso)
    error_list.append(hist_error_pso)

# Plot & Save Graph: Iterations required PSO
x = np.arange(len(iteration_list))
fig8, axs8 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs8.bar(x, height=iteration_list)
axs8.set_xticks(x, labels=map(str, particles))
axs8.grid()
axs8.set_xlabel("Iteration")
axs8.set_ylabel("Particles")
fig8.savefig('results/3_PSO_iterations.png')

# We choose the PSO model with 6 particles because we can see on the above graph that it has good performances
fig9, ax9 = plt.subplots(figsize=(15, 7.5))
ax9.title.set_text('Error Evolution')
ax9.plot(np.linspace(0, iteration_list[4], iteration_list[4]+1), hist_error_nm[:iteration_list[4]+1], color="blue", linewidth=2, label="Nelder Mead")
ax9.set_xlabel("Iteration")
ax9.set_ylabel("Nelder Mead Error", color="blue")
ax9.tick_params(axis='y', labelcolor="blue")
ax9bis = ax9.twinx()  # instantiate a second axes that shares the same x-axis
ax9bis.set_ylabel('PSO Error', color="red")  # we already handled the x-label with ax1
ax9bis.plot(np.linspace(0, iteration_list[4], iteration_list[4]+1), error_list[4], color="red", linewidth=2, label="PSO")
ax9bis.tick_params(axis='y', labelcolor="red")
fig9.tight_layout()  # otherwise the right y-label is slightly clipped
ax9.grid()
fig9.savefig('results/3_error_iterations.png')

# Display Graphs
plt.show()
