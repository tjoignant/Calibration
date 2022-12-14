import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

import svi
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

# ---------------------------------------- PART 1.1 : RISK NEUTRAL DENSITY ----------------------------------------
# Calibrate Interpolation Functions
linear_interp = interpolation.Interp1D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (12M)"])
squared_interp = interpolation.Interp2D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (12M)"])
cubic_interp = interpolation.Interp3D(x_list=df_mkt["Strike"], y_list=df_mkt["IV (12M)"])

# Create Interpolated Risk Neutral Density Dataframe (12M)
df_density = pd.DataFrame()
df_density["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"])+step, step)
df_density["IV (1D)"] = df_density.apply(lambda x: linear_interp.get_image(x["Strike"]), axis=1)
df_density["IV (2D)"] = df_density.apply(lambda x: squared_interp.get_image(x["Strike"]), axis=1)
df_density["IV (3D)"] = df_density.apply(lambda x: cubic_interp.get_image(x["Strike"]), axis=1)
df_density["IV"] = df_density["IV (3D)"]
df_density["Price"] = df_density.apply(
    lambda x: black_scholes.BS_Price(df=1, f=100, k=x["Strike"], t=1, v=x["IV"], OptType="C"), axis=1)

# Compute Breeden-Litzenberger Density
df_density["Density"] = (df_density["Price"].shift(1) - 2 * df_density["Price"] + df_density["Price"].shift(-1)) \
                             / pow(step, 2)
df_density["Density Norm"] = df_density["Density"] / df_density["Density"].sum()

# Plot & Save Graph: Interpolated Volatilities
fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs1.plot(df_density["Strike"], df_density["IV (1D)"], label="Linear")
axs1.plot(df_density["Strike"], df_density["IV (2D)"], label="Squared")
axs1.plot(df_density["Strike"], df_density["IV (3D)"], label="Cubic")
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

# Calibrate SVI Curve
implied_vol = df_mkt["IV (12M)"].to_numpy()
total_variance = np.power(implied_vol, 2)
log_forward_moneyness = np.log(df_mkt["Strike"] / 100)
weights = [1 for _ in implied_vol]
SVI_params = svi.SVI_calibration(k_list=log_forward_moneyness, mktTotVar_list=total_variance, weights_list=weights,
                                 use_durrleman_cond=False)

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

# Set Custom Density Dataframe
df_custom_density = pd.DataFrame()
df_custom_density["Strike"] = np.arange(95, 105+step, step)
df_custom_density["IV"] = df_custom_density.apply(
    lambda x: np.sqrt(svi.SVI(k=np.log(x["Strike"] / 100), a_=SVI_params["a_"], b_=SVI_params["b_"],
                              rho_=SVI_params["rho_"], m_=SVI_params["m_"], sigma_=SVI_params["sigma_"])), axis=1)
df_custom_density["Price"] = df_custom_density.apply(
    lambda x: black_scholes.BS_Price(df=1, f=100, k=x["Strike"], t=1, v=x["IV"], OptType="C"), axis=1)

# Compute Breeden-Litzenberger Density
df_custom_density["Density"] = (df_custom_density["Price"].shift(1) - 2 * df_custom_density["Price"] + df_custom_density["Price"].shift(-1)) \
                             / pow(step, 2)


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * np.power((x - mu) / sigma, 2)) / (sigma * np.sqrt(2 * np.pi))


# Gaussian Density Comparison
mu = 100.94
sigma = 1.15
right_mu = 102.1
right_sigma = 1.7
left_mu = 99.5
left_sigma = 1.9
df_custom_density["Density (Gaussian)"] = df_custom_density["Strike"].apply(lambda x: normal_pdf(x, mu, sigma))
df_custom_density["Right Density (Gaussian)"] = df_custom_density["Strike"].apply(lambda x: normal_pdf(x, right_mu, right_sigma))
df_custom_density["Left Density (Gaussian)"] = df_custom_density["Strike"].apply(lambda x: normal_pdf(x, left_mu, left_sigma))

# Compute Normalized Densities
for density in ["Density", "Density (Gaussian)", "Right Density (Gaussian)", "Left Density (Gaussian)"]:
    df_custom_density[f"Norm. {density}"] = df_custom_density[density] / df_custom_density[density].sum()

# Plot & Save Graph: Extrapolated Density
fig4, axs4 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs4.plot(df_custom_density["Strike"], df_custom_density["Density"], label="Custom")
axs4.plot(df_custom_density["Strike"], df_custom_density["Density (Gaussian)"], "--", label="Centered Gaussian")
axs4.plot(df_custom_density["Strike"], df_custom_density["Right Density (Gaussian)"], "--", label="Right Gaussian")
axs4.plot(df_custom_density["Strike"], df_custom_density["Left Density (Gaussian)"], "--", label="Left Gaussian")
axs4.grid()
axs4.set_xlabel("Strike")
axs4.set_ylabel("Density")
axs4.legend()
fig4.savefig('results/1.4_Custom_Density.png')


# ------------------------------------- PART 1.2 : METROPOLIS-HASTINGS ALGORITHM -------------------------------------
# Set PI Function Values
pi_x = df_custom_density["Strike"].values[1:-1]
pi_y = df_custom_density["Density"].values[1:-1]


# Target Distribution Function
def target_distrib(x, pi_x, pi_y):
    output = 0
    # If x in pi_x
    if pi_x[0] <= x <= pi_x[-1]:
        # Return interpolated pi_y
        for i in range(1, len(pi_x)):
            if pi_x[i-1] < x <= pi_x[i]:
                output = (pi_y[i-1] + pi_y[i]) / 2
                break
    # If x in right tail
    elif x > pi_x[-1]:
        # Return right gaussian
        output = normal_pdf(x, right_mu, right_sigma)
    # If x in left tail
    else:
        # Return left gaussian
        output = normal_pdf(x, left_mu, left_sigma)
    return output


# Metropolis-Hastings Algorithm
N = 250000
rdm_x = np.arange(N, dtype=float)
rdm_x[0] = 100
counter = 0
for i in range(0, N - 1):
    x_next = np.random.normal(rdm_x[i], 1)
    if np.random.random_sample() < min(1, target_distrib(x_next, pi_x=pi_x, pi_y=pi_y) /
                                          target_distrib(rdm_x[i], pi_x=pi_x, pi_y=pi_y)):
        rdm_x[i + 1] = x_next
        counter = counter + 1
    else:
        rdm_x[i + 1] = rdm_x[i]

# Display Acceptance Ration
print(f"\nMetropolis-Hastings Acceptance Ratio: {counter / float(N)}")

# Compute SVI & Density Prices
df_metropolis = pd.DataFrame()
df_metropolis["Strike"] = range(95, 105)
df_metropolis["IV"] = df_metropolis.apply(
    lambda x: np.sqrt(svi.SVI(k=np.log(x["Strike"] / 100), a_=SVI_params["a_"], b_=SVI_params["b_"],
                              rho_=SVI_params["rho_"], m_=SVI_params["m_"], sigma_=SVI_params["sigma_"])), axis=1)
df_metropolis["Price (SVI)"] = df_metropolis.apply(
    lambda x: black_scholes.BS_Price(df=1, f=100, k=x["Strike"], t=1, v=x["IV"], OptType="C"), axis=1)

df_metropolis["Price (Density)"] = df_metropolis.apply(
    lambda x: np.mean(np.maximum(rdm_x - x["Strike"], 0)), axis=1)
print(f"\n{df_metropolis}")

# Plot & Save Graph: Density of x
data_entries, bins = np.histogram(rdm_x, bins=80)
binscenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
data_entries_norm = np.array(data_entries) / sum(data_entries)
target_distrib = [target_distrib(x, pi_x=pi_x, pi_y=pi_y) for x in binscenters]
target_distrib = target_distrib / sum(target_distrib)
fig5, axs5 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
axs5.plot(binscenters, target_distrib, label="Target")
axs5.plot(binscenters, data_entries_norm, label="Simulated")
axs5.grid()
axs5.legend()
axs5.set_xlabel("Strike")
axs5.set_ylabel("Density")
fig5.savefig('results/1.5_Metropolis_Density.png')

# Display Graphs
plt.show()
