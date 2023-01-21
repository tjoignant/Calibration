import warnings
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

import cev
import interpolation
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

# ---------------------------------------- PART 2.1 : BLACK-SCHOLES PRICE ----------------------------------------
# Plot & Save Graph: Volatility Surface
fig6 = plt.figure(figsize=(15, 7.5))
axs6 = fig6.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(df_mkt["Strike"], [3, 6, 9, 12])
Z = np.array([np.array(df_mkt["IV (3M)"]), np.array(df_mkt["IV (6M)"]),
              np.array(df_mkt["IV (9M)"]), np.array(df_mkt["IV (12M)"])])
surf = axs6.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs6.set_xlabel("Strike")
axs6.set_ylabel("Maturity")
axs6.set_zlabel("IV")
fig6.savefig('results/2.1_Volatility_Surface.png')

# Compute New Option BS Price (K=99.5, T=8M)
strike_interp_6M = interpolation.Interp3D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (6M)"])
strike_interp_9M = interpolation.Interp3D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (9M)"])
maturity_interp = interpolation.Interp1D(x_list=[6, 9], y_list=[pow(strike_interp_6M.get_image(99.5), 2)*6/12,
                                                                pow(strike_interp_9M.get_image(99.5), 2)*9/12])
iv = np.sqrt(maturity_interp.get_image(8)/(8/12))
BS_price = black_scholes.BS_Price(f=100, k=99.5, t=8 / 12, v=iv, df=1, OptType="C")
print(f"\nBlack-Scholes")
print(f" - Implied Vol: {iv}")
print(f" - Price: {BS_price}")

# Create Interpolated Local volatilities Dataframe
df_local_vol = pd.DataFrame()
df_local_vol["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"]+step), step)
df_local_vol["IV (6M)"] = df_local_vol.apply(lambda x: strike_interp_6M.get_image(x=x["Strike"]), axis=1)
df_local_vol["IV (9M)"] = df_local_vol.apply(lambda x: strike_interp_9M.get_image(x=x["Strike"]), axis=1)
df_local_vol["IV (8M)"] = df_local_vol.apply(lambda x: np.sqrt(interpolation.Interp1D(x_list=[6, 9],
                                                                              y_list=[pow(strike_interp_6M.get_image(x["Strike"]), 2) * 6/12, pow(strike_interp_9M.get_image(x["Strike"]), 2) * 9/12]).get_image(8) / (8/12)), axis=1)

# Plot & Save Graph: Interpolated Volatilities (8M)
fig7, axs7 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs7.plot(df_local_vol["Strike"], df_local_vol["IV (6M)"], label="6M")
axs7.plot(df_local_vol["Strike"], df_local_vol["IV (9M)"], label="9M")
axs7.plot(df_local_vol["Strike"], df_local_vol["IV (8M)"], "--", label="8M")
axs7.grid()
axs7.set_xlabel("Strike")
axs7.set_ylabel("IV")
axs7.legend()
fig7.savefig('results/2.2_Interpolated_Volatilities_8M.png')


# -------------------------------------------- PART 2.2 : CEV PRICE ---------------------------------------------
"""
# CEV Calibration (fixed gamma=1)
df_cev_fixed_gamma = df_mkt

# Compute Option's Implied Volatility (IV)
for maturity in [3, 6, 9, 12]:
    df_cev_fixed_gamma[f"sigma0 ({maturity}M)"] = df_cev_fixed_gamma.apply(
        lambda x: cev.CEV_Sigma_Nelder_Mead_1D(
            MktPrice=x[f"Price ({maturity}M)"], df=1, f=100, k=x["Strike"], t=maturity / 12, gamma=1, OptType="C")[0], axis=1)

# Plot & Save Graph: Sigma0 Surface
fig8 = plt.figure(figsize=(15, 7.5))
axs8 = fig8.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(df_cev_fixed_gamma["Strike"], [3, 6, 9, 12])
Z = np.array([np.array(df_cev_fixed_gamma["sigma0 (3M)"]), np.array(df_cev_fixed_gamma["sigma0 (6M)"]),
              np.array(df_cev_fixed_gamma["sigma0 (9M)"]), np.array(df_cev_fixed_gamma["sigma0 (12M)"])])
surf = axs8.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs8.set_xlabel("Strike")
axs8.set_ylabel("Maturity")
axs8.set_zlabel("Sigma0")
fig8.savefig('results/2.3_Sigma0_Surface.png')
"""


# CEV Calibration
df_cev_gamma = df_mkt
mktPrice_list = list(df_mkt["Price (3M)"]) + list(df_mkt["Price (6M)"]) + list(df_mkt["Price (9M)"]) \
                + list(df_mkt["Price (12M)"])
strike_list = list(df_mkt["Strike"]) + list(df_mkt["Strike"]) + list(df_mkt["Strike"]) + list(df_mkt["Strike"])
maturity_list = [3 / 12 for _ in list(df_mkt["Price (3M)"])] + [6 / 12 for _ in list(df_mkt["Price (6M)"])] + \
                [9 / 12 for _ in list(df_mkt["Price (9M)"])] + [12 / 12 for _ in list(df_mkt["Price (12M)"])]
inputs_list = []
for i in range(0, len(mktPrice_list)):
    inputs_list.append((strike_list[i], maturity_list[i], 100, 1, "C"))

gamma, nb_iter = cev.CEV_Gamma_Calibration_Nelder_Mead_1D(inputs_list, mktPrice_list)
print("\nCEV Calibration:")
print(f" - gamma: {gamma}")
print(f" - nb_iter: {nb_iter}")

# Compute Option's Implied Volatility (IV)
for maturity in [3, 6, 9, 12]:
    df_cev_gamma[f"sigma0 ({maturity}M)"] = df_cev_gamma.apply(
        lambda x: cev.CEV_Sigma_Nelder_Mead_1D(
            MktPrice=x[f"Price ({maturity}M)"], df=1, f=100, k=x["Strike"], t=maturity / 12, gamma=1, OptType="C")[0], axis=1)

# Plot & Save Graph: Sigma0 Surface (with calibrated gamma)
fig9 = plt.figure(figsize=(15, 7.5))
axs9 = fig8.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(df_cev_fixed_gamma["Strike"], [3, 6, 9, 12])
Z = np.array([np.array(df_cev_fixed_gamma["sigma0 (3M)"]), np.array(df_cev_fixed_gamma["sigma0 (6M)"]),
              np.array(df_cev_fixed_gamma["sigma0 (9M)"]), np.array(df_cev_fixed_gamma["sigma0 (12M)"])])
surf = axs9.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs9.set_xlabel("Strike")
axs9.set_ylabel("Maturity")
axs9.set_zlabel("Sigma0")
fig9.savefig('results/2.4_Sigma0_Surface_Bis.png')


# ------------------------------------------- PART 2.3 : DUPIRE PRICE -------------------------------------------

"""
# Create Interpolated Volatility Surface
df_dupire_vol = pd.DataFrame()
df_dupire_vol["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"])+step, step)

# Squared Interpolation On Strikes
for matu in [3, 6, 9, 12]:
    strike_interp = interpolation.Interp3D(x_list=df_mkt["Strike"], y_list=df_mkt[f"IV ({matu}M)"])
    df_dupire_vol[f"{matu}"] = df_dupire_vol.apply(lambda x: strike_interp.get_image(x["Strike"]), axis=1)

# Linear Interpolation On Total Variance
for strike in df_dupire_vol["Strike"]:
    maturity_interp = interpolation.Interp1D(x_list=[0, 3, 6, 9, 12],
                                     y_list=[0,
                                             pow(df_dupire_vol[df_dupire_vol["Strike"] == strike]["3"].values[0], 2) * 3/12,
                                             pow(df_dupire_vol[df_dupire_vol["Strike"] == strike]["6"].values[0], 2) * 6/12,
                                             pow(df_dupire_vol[df_dupire_vol["Strike"] == strike]["9"].values[0], 2) * 9/12,
                                             pow(df_dupire_vol[df_dupire_vol["Strike"] == strike]["12"].values[0], 2) * 12/12])
    for matu in np.arange(step, 12+step, step):
        index = df_dupire_vol[df_dupire_vol["Strike"] == strike].index[0]
        df_dupire_vol.loc[index, round(matu, 2)] = np.sqrt(maturity_interp.get_image(round(matu, 2)) / (matu / 12))
df_dupire_vol.drop(["3", "6", "9", "12"], axis=1, inplace=True)

# Create Tot Variance Surfacce
df_dupire_tot_var = pd.DataFrame()
df_dupire_tot_var["k"] = np.log(df_dupire_vol["Strike"] / 100)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
for matu in df_dupire_vol.columns[1:]:
    df_dupire_tot_var[matu] = np.power(df_dupire_vol[matu], 2) * matu/12

# Create Dupire Surface
df_dupire = pd.DataFrame()
df_dupire["k"] = df_dupire_tot_var["k"]
k = df_dupire["k"]
for matu in df_dupire_tot_var.columns[1:-1]:
    w = df_dupire_tot_var[matu]
    dk = (df_dupire_tot_var[matu].diff(1)) / k.diff(1)
    dk2 = 2 / (k.diff(1) - k.diff(-1)) * ((-df_dupire_tot_var[matu].diff(1)) / (k.diff(1)) - (df_dupire_tot_var[matu].diff(-1)) / (-k.diff(-1)))
    dT = (df_dupire_tot_var[round(matu+step, 2)] - df_dupire_tot_var[matu]) / (step / 12) if round(matu, 2) != 12.0 else np.NaN
    df_dupire[matu] = np.sqrt(np.abs(dT / (1 - k/w * dk + 1/2 * dk2 + 1/4 * (np.power(k, 2) / np.power(w, 2) - 1/w - 1/4) * np.power(dk, 2))))
df_dupire = df_dupire.iloc[1:-1]

# Set Sigma Function
def LV_sigma(LV_surface, St, t, F=100):
    k = np.log(St / F)
    t = round(t*12, 2)
    # If St below k_min
    if k <= min(LV_surface["k"]):
        return LV_surface[t].values[0]
    # If St above k_max
    elif k >= max(LV_surface["k"]):
        return LV_surface[t].values[-1]
    # Else
    else:
        return interpolation.Interp1D(x_list=LV_surface["k"].values, y_list=LV_surface[t].values).get_image(k)


# Set Local Vol Monte Carlo Diffusion
def LV_Diffusion(S0, drift, maturity, LV_surface, nb_simuls, nb_steps, seed=123):
    np.random.seed(seed)
    dt = maturity / nb_steps
    S = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=S0, dtype=float)
    Z = np.random.normal(loc=0, scale=1, size=(nb_steps, nb_simuls))
    for i in range(1, nb_steps + 1):
        sigma = np.array([LV_sigma(LV_surface=LV_surface, St=s, t=i * dt) for s in S[i - 1]])
        # EULER
        S[i] = S[i - 1] * np.exp((drift - 0.5 * np.power(sigma, 2)) * dt + sigma * np.sqrt(dt) * Z[i-1, :])
        # APPROX
        # S[i] = S[i - 1] * (1 + drift * dt + sigma * np.sqrt(dt) * Z[i - 1, :])
    return S


# Price Using Dupire Surface (Local Vol)
simulations = LV_Diffusion(S0=100, drift=0, maturity=8/12, LV_surface=df_dupire, nb_simuls=10000, nb_steps=80)
price = np.mean(np.array([max(0, (sim - 99.5)) for sim in simulations[-1]]))
print("\nDupire")
print(f" - Price: {price}")
"""

# Display Graphs
plt.show()
