import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm

import svi
import utils
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

# --------------------------- PART 1 : RISK NEUTRAL DENSITY ---------------------------

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
axs1.legend()
fig1.savefig('results/1.1_Interpolated_Volatilities.png')

# Plot & Save Graph: Interpolated Volatilities
fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs2.plot(df_density["Strike"], df_density["Density"], label="Interpolated")
axs2.grid()
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
axs3.legend()
fig3.savefig('results/1.3_SVI_Calibration.png')

# Create Interpolated/Extrapolated Risk Neutral Density Dataframe (12M)
df_density_bis = pd.DataFrame()
df_density_bis["Strike"] = np.arange(70, 130, 0.1)
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
axs4.legend()
fig4.savefig('results/1.4_SVI_Density.png')

# Display Graphs
plt.show()
