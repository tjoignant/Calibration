import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

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
strike_interp_6M = interpolation.Interp2D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (6M)"])
strike_interp_9M = interpolation.Interp2D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (9M)"])
maturity_interp = interpolation.Interp1D(x_list=[6, 9], y_list=[strike_interp_6M.get_image(99.5), strike_interp_9M.get_image(99.5)])
BS_price = black_scholes.BS_Price(f=100, k=99.5, t=8 / 12, v=maturity_interp.get_image(8), df=1, OptType="C")
print(f"\nBlack-Scholes")
print(f" - Implied Vol: {maturity_interp.get_image(8)}")
print(f" - Price: {BS_price}")

# Create Interpolated Local volatilities Dataframe
df_local_vol = pd.DataFrame()
df_local_vol["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"]+step), step)
df_local_vol["IV (6M)"] = df_local_vol.apply(lambda x: strike_interp_6M.get_image(x=x["Strike"]), axis=1)
df_local_vol["IV (9M)"] = df_local_vol.apply(lambda x: strike_interp_9M.get_image(x=x["Strike"]), axis=1)
df_local_vol["IV (8M)"] = df_local_vol.apply(lambda x: interpolation.Interp1D(x_list=[6, 9], y_list=[strike_interp_6M.get_image(x["Strike"]), strike_interp_9M.get_image(x["Strike"])]).get_image(8), axis=1)

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


# -------------------------------------------- PART 2.2 : CEV CALIBRATION ---------------------------------------------
# CEV Calibration (fixed gamma=1)
# CEV Calibration


# ------------------------------------------- PART 2.3 : DUPIRE CALIBRATION -------------------------------------------
# Create Interpolated Volatility Surface
df_dupire_vol = pd.DataFrame()
df_dupire_vol["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"])+step, step)
# Squared Interpolation On Strikes
for matu in [3, 6, 9, 12]:
    strike_interp = interpolation.Interp2D(x_list=df_mkt["Strike"], y_list=df_mkt[f"IV ({matu}M)"])
    df_dupire_vol[f"{matu}"] = df_dupire_vol.apply(lambda x: strike_interp.get_image(x["Strike"]), axis=1)
# Linear Interpolation On Maturities
for strike in df_dupire_vol["Strike"]:
    maturity_interp = interpolation.Interp1D(x_list=[3, 6, 9, 12],
                                     y_list=[df_dupire_vol[df_dupire_vol["Strike"] == strike]["3"].values[0],
                                             df_dupire_vol[df_dupire_vol["Strike"] == strike]["6"].values[0],
                                             df_dupire_vol[df_dupire_vol["Strike"] == strike]["9"].values[0],
                                             df_dupire_vol[df_dupire_vol["Strike"] == strike]["12"].values[0]])
    for matu in np.arange(3, 12+step, step):
        index = df_dupire_vol[df_dupire_vol["Strike"] == strike].index[0]
        df_dupire_vol.loc[index, round(matu, 2)] = maturity_interp.get_image(round(matu, 2))
df_dupire_vol.drop(["3", "6", "9", "12"], axis=1, inplace=True)
# Create Interpolated Price Surface
df_dupire_price = pd.DataFrame()
df_dupire_price["Strike"] = df_dupire_vol["Strike"]
for matu in df_dupire_vol.columns[1:]:
    df_dupire_price[matu] = df_dupire_vol.apply(lambda x: black_scholes.BS_Price(df=1, f=100, k=x["Strike"], t=matu / 12, v=x[matu], OptType="C"), axis=1)

# Create Dupire DataFrame
df_dupire = pd.DataFrame()
df_dupire["Strike"] = df_dupire_price["Strike"]
for matu in df_dupire_price.columns[1:]:
    gamma = (df_dupire_price[matu].shift(-1) - 2 * df_dupire_price[matu] + df_dupire_price[matu].shift(1)) / (pow(step, 2))
    gamma = [np.NaN if g <= 0 else g for g in gamma]
    theta = (df_dupire_price[round(matu+step,2)] - df_dupire_price[matu]) / (step / 12) if round(matu, 2) != 12.0 else np.NaN
    df_dupire[matu] = np.sqrt(2 * theta / (np.power(df_dupire["Strike"], 2) * gamma))
print(df_dupire)

# Display Graphs
plt.show()
