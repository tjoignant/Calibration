import numpy as np
import pandas as pd

from matplotlib import cm
from scipy.stats import norm
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import cev
import svi
import utils
import models
import optimization
import black_scholes

# Report Link : https://www.overleaf.com/project/639051dcb072f741700fb0f1

# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 300)
pd.set_option('display.expand_frame_repr', False)

# Set Market Options DataFrame
df_mkt = pd.DataFrame()
df_mkt["Strike"] = np.arange(95, 105, 1)
df_mkt["Price (3M)"] = [8.67, 7.14, 5.98, 4.93, 4.09, 3.99, 3.43, 3.01, 2.72, 2.53]
df_mkt["Price (6M)"] = [10.71, 8.28, 6.91, 6.36, 5.29, 5.07, 4.76, 4.47, 4.35, 4.14]
df_mkt["Price (9M)"] = [11.79, 8.95, 8.07, 7.03, 6.18, 6.04, 5.76, 5.50, 5.50, 5.39]
df_mkt["Price (12M)"] = [12.40, 9.59, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]

# Compute Option's Implied Volatility (IV), Total Variance (TV), Log Forward Moneyness (LFM)
for maturity in [3, 6, 9, 12]:
    df_mkt[f"IV ({maturity}M)"] = df_mkt.apply(
        lambda x: black_scholes.BS_IV_Newton_Raphson(
            MktPrice=x[f"Price ({maturity}M)"], df=1, f=100, k=x["Strike"], t=maturity / 12, OptType="C")[0], axis=1)


# ---------------------------------------- PART 1.1 : RISK NEUTRAL DENSITY ----------------------------------------
# Calibrate Interpolation Function
interp_function = interp1d(x=df_mkt["Strike"], y=df_mkt["IV (12M)"], kind='cubic')
my_interp_function = utils.Interp(x_list=df_mkt["Strike"], y_list=df_mkt["IV (12M)"])

# Create Interpolated Risk Neutral Density Dataframe
df_density = pd.DataFrame()
df_density["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"]), 0.05)
df_density["IV"] = df_density.apply(lambda x: interp_function(x["Strike"]), axis=1)
df_density["IV (2D)"] = df_density.apply(lambda x: my_interp_function.get_image(x["Strike"]), axis=1)
df_density["Price"] = df_density.apply(
    lambda x: black_scholes.BS_Price(df=1, f=100, k=x["Strike"], t=1, v=x["IV"], OptType="C"), axis=1)
df_density["Gamma Strike"] = (df_density["Price"].shift(1) - 2 * df_density["Price"] + df_density["Price"].shift(-1)) \
                             / pow(0.01, 2)
df_density["Density"] = df_density["Gamma Strike"] / df_density["Gamma Strike"].sum()

# Plot & Save Graph: Interpolated Volatilities
fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs1.plot(df_density["Strike"], df_density["IV (2D)"], label="Interpolated (2D)")
axs1.plot(df_density["Strike"], df_density["IV"], label="Interpolated (3D)")
axs1.scatter(df_mkt["Strike"], df_mkt["IV (12M)"], label="Implied")
axs1.grid()
axs1.set_xlabel("Strike")
axs1.set_ylabel("IV")
axs1.legend()
fig1.savefig('results/1.1_Interpolated_Volatilities.png')

# Plot & Save Graph: Interpolated Volatilities
fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs2.plot(df_density["Strike"], df_density["Density"], label="Interpolated")
axs2.grid()
axs2.set_xlabel("Strike")
axs2.set_ylabel("Density")
fig2.savefig('results/1.2_Interpolated_Density.png')

# Compute Total Variance (TV) & Log Forward Moneyness (LFM)
df_density["TV"] = df_density["IV"] * df_density["IV"] * 1
df_density["LFM"] = df_density.apply(lambda x: np.log(x["Strike"] / 100), axis=1)

# Calibrate SVI Curve
SVI_params = svi.SVI_calibration(k_list=df_density["LFM"], mktTotVar_list=df_density["TV"],
                                 weights_list=df_density["IV"] * df_density["IV"], use_durrleman_cond=True)

# Compute SVI Volatilities
df_density["IV (SVI)"] = df_density.apply(
    lambda x: np.sqrt(svi.SVI(k=np.log(x["Strike"] / 100), a_=SVI_params["a_"], b_=SVI_params["b_"],
                              rho_=SVI_params["rho_"], m_=SVI_params["m_"], sigma_=SVI_params["sigma_"])), axis=1)

# Plot & Save Graph: SVI Calibration
fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs3.plot(df_density["Strike"], df_density["IV"], label="Interpolated")
axs3.scatter(df_mkt["Strike"], df_mkt["IV (12M)"], label="Implied")
axs3.plot(df_density["Strike"], df_density["IV (SVI)"], label="SVI")
axs3.grid()
axs3.set_xlabel("Strike")
axs3.set_ylabel("IV")
axs3.legend()
fig3.savefig('results/1.3_SVI_Calibration.png')

# Create Interpolated/Extrapolated Risk Neutral Density Dataframe (12M)
df_density_bis = pd.DataFrame()
df_density_bis["Strike"] = np.arange(65, 135, 0.1)
df_density_bis["IV (SVI)"] = df_density_bis.apply(
    lambda x: np.sqrt(svi.SVI(k=np.log(x["Strike"] / 100), a_=SVI_params["a_"], b_=SVI_params["b_"],
                              rho_=SVI_params["rho_"], m_=SVI_params["m_"], sigma_=SVI_params["sigma_"])), axis=1)

# Compute Option Price Continuum
df_density_bis["Price (SVI)"] = df_density_bis.apply(
    lambda x: black_scholes.BS_Price(df=1, f=100, k=x["Strike"], t=1, v=x["IV (SVI)"], OptType="C"), axis=1)

# Compute SVI Breeden-Litzenberger Density
df_density_bis["Gamma Strike (SVI)"] = (df_density_bis["Price (SVI)"].shift(1) - 2 * df_density_bis["Price (SVI)"] +
                                        df_density_bis["Price (SVI)"].shift(-1)) / pow(0.01, 2)
df_density_bis["Density (SVI)"] = df_density_bis["Gamma Strike (SVI)"] / df_density_bis["Gamma Strike (SVI)"].sum()

# Gaussian Density Comparison
df_density_bis["Gaussian Distrib"] = norm.pdf(df_density_bis["Strike"], 99.8, 6.9)
df_density_bis["Density (Gaussian)"] = df_density_bis["Gaussian Distrib"] / df_density_bis["Gaussian Distrib"].sum()

# Plot & Save Graph: Extrapolated Density
fig4, axs4 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs4.plot(df_density_bis["Strike"], df_density_bis["Density (SVI)"], label="SVI")
axs4.plot(df_density_bis["Strike"], df_density_bis["Density (Gaussian)"], c="g", label="Gaussian")
axs4.grid()
axs4.set_xlabel("Strike")
axs4.set_ylabel("Density")
axs4.legend()
fig4.savefig('results/1.4_SVI_Density.png')


# ------------------------------------- PART 1.2 : METROPOLIS-HASTINGS ALGORITHM -------------------------------------
# Set PI Function Values
pi_x = df_density_bis["Strike"].values[1:-1]
pi_y = df_density_bis["Density (SVI)"].values[1:-1]


# Target Distribution Function
def target_distrib(x, pi_x, pi_y, mu, sigma):
    # If x in pi_x
    if pi_x[0] <= x <= pi_x[-1]:
        # Return interpolated pi_y
        for i in range(1, len(pi_x)):
            if pi_x[i] > x:
                return (pi_y[i] + pi_y[i - 1]) / 2
    # Else (tails)
    else:
        # Return fitted gaussian
        numerator = np.exp((-(x - mu) ** 2) / (2 * sigma ** 2))
        denominator = sigma * np.sqrt(2 * np.pi)
        return numerator / denominator


# Algorithm
N = 100000
x = np.arange(N, dtype=float)
x[0] = 100
counter = 0
for i in range(0, N - 1):
    x_next = np.random.normal(x[i], 1)
    if np.random.random_sample() < min(1, target_distrib(x_next, pi_x=pi_x, pi_y=pi_y, mu=99.8, sigma=6.9)/
                                          target_distrib(x[i], pi_x=pi_x, pi_y=pi_y, mu=99.8, sigma=6.9)):
        x[i + 1] = x_next
        counter = counter + 1
    else:
        x[i + 1] = x[i]
print(f"\nMetropolis-Hastings Acceptance Ratio: {counter / float(N)}")

# Plot & Save Graph: Density of x
fig5, axs5 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs5.hist(x, density=True, bins=50, color='blue', label="Density")
axs5.grid()
axs5.set_xlabel("Strike")
axs5.set_ylabel("Density")
fig5.savefig('results/1.5_Metropolis_Density.png')


# ---------------------------------------- PART 2.1 : VOLATILITY INTERPOLATION ----------------------------------------
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
df_local_vol = df_mkt
df_local_vol["Maturity Interp"] = df_local_vol.apply(
    lambda x: interp1d(x=[3, 6, 9, 12], y=[x[f"IV ({matu}M)"] for matu in [3, 6, 9, 12]], kind='cubic'), axis=1)
df_local_vol["IV (8M)"] = df_local_vol.apply(lambda x: x["Maturity Interp"](8), axis=1)
interp_function = interp1d(x=df_local_vol["Strike"], y=df_local_vol["IV (8M)"], kind='cubic')
BS_price = black_scholes.BS_Price(f=100, k=99.5, t=8 / 12, v=interp_function(99.5), df=1, OptType="C")
print(f"\nBlack-Scholes")
print(f" - Implied Vol: {interp_function(99.5)}")
print(f" - Price: {BS_price}")

# Create Interpolated Local volatilities Dataframe
my_interp_function = utils.Interp(x_list=df_local_vol["Strike"], y_list=df_local_vol["IV (8M)"])
df_local_vol_bis = pd.DataFrame()
df_local_vol_bis["Strike"] = np.arange(min(df_local_vol["Strike"]), max(df_local_vol["Strike"]), 0.05)
df_local_vol_bis["IV (2D)"] = df_local_vol_bis.apply(lambda x: my_interp_function.get_image(x=x["Strike"]), axis=1)
df_local_vol_bis["IV"] = df_local_vol_bis.apply(lambda x: interp_function(x["Strike"]), axis=1)

# Plot & Save Graph: Interpolated Volatilities (8M)
fig7, axs7 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs7.plot(df_local_vol_bis["Strike"], df_local_vol_bis["IV (2D)"], label="Interpolated Strike (2D)")
axs7.plot(df_local_vol_bis["Strike"], df_local_vol_bis["IV"], label="Interpolated Strike (3D)")
axs7.scatter(df_local_vol["Strike"], df_local_vol["IV (8M)"], label="Interpolated Maturity (3D)")
axs7.grid()
axs7.set_xlabel("Strike")
axs7.set_ylabel("IV")
axs7.legend()
fig7.savefig('results/2.2_Interpolated_Volatilities_8M.png')

# -------------------------------------------- PART 2.2 : CEV CALIBRATION ---------------------------------------------
# CEV Calibration (fixed gamma=1)
# CEV Calibration


# ------------------------------------------- PART 2.3 : DUPIRE CALIBRATION -------------------------------------------


# ----------------------------------------- PART 3 : OPTIMISATION ALGORITHMS ------------------------------------------
# Set Inputs List
mktPrice_list = list(df_mkt["Price (3M)"]) + list(df_mkt["Price (6M)"]) + list(df_mkt["Price (9M)"]) \
                + list(df_mkt["Price (12M)"])
strike_list = list(df_mkt["Strike"]) + list(df_mkt["Strike"]) + list(df_mkt["Strike"]) + list(df_mkt["Strike"])
maturity_list = [3 / 12 for _ in list(df_mkt["Price (3M)"])] + [6 / 12 for _ in list(df_mkt["Price (6M)"])] + \
                [9 / 12 for _ in list(df_mkt["Price (9M)"])] + [12 / 12 for _ in list(df_mkt["Price (12M)"])]
inputs_list = []
for i in range(0, len(mktPrice_list)):
    inputs_list.append((strike_list[i], maturity_list[i], 100, 1, "C"))

# Estimate CEV Params (Nelder-Mead)
nelder_mead_params, nb_iter = optimization.CEV_nelder_mead(inputs_list=inputs_list, mktPrice_list=mktPrice_list)
print("\nNelder-Mead")
print(f" - gamma: {nelder_mead_params[0]}")
print(f" - sigma0: {nelder_mead_params[1]}")
print(f" - Nb Iterations: {nb_iter}")

# Display Graphs
plt.show()
