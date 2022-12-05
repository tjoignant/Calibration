import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm

import black_scholes

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

# Compute Option's Implied Volatility
for maturity in [3, 6, 9, 12]:
    df_mkt[f"IV ({maturity}M)"] = df_mkt.apply(
        lambda x: black_scholes.BS_IV_Newton_Raphson(
            MktPrice=x[f"Price ({maturity}M)"], df=1, f=100, k=x["Strike"], t=maturity/12, OptType="C")[0], axis=1)

# --------------------------- Q1 : RISK NEUTRAL DENSITY ---------------------------
# Create Risk Neutral Density Dataframe (12M)
f2 = interp1d(x=df_mkt["Strike"], y=df_mkt["IV (12M)"], kind='cubic')
df_density = pd.DataFrame()
df_density["Strike"] = np.arange(min(df_mkt["Strike"]), max(df_mkt["Strike"]), 0.005)
df_density["IV"] = f2(df_density["Strike"])
df_density["Price"] = df_density.apply(
    lambda x: black_scholes.BS_Price(df=1, f=100, k=x[ "Strike"], t=1, v=x["IV"], OptType="C"), axis=1)
df_density["Cummulative Distribution"] = 1 + (df_density["Price"].shift(-1) - df_density["Price"].shift(1)) / 0.01 * df_density["Strike"]
df_density["Density"] = (df_density["Price"].shift(1) - 2 * df_density["Price"] + df_density["Price"].shift(-1)) / pow(0.01, 2)
df_density["Density"] = df_density["Density"] / df_density["Density"].sum() * 100
df_density["Gamma Strike"] = df_density.apply(lambda x: black_scholes.BS_Gamma_Strike(f=100, k=x["Strike"], t=1, v=x["IV"], df=1, OptType="C"), axis=1)
df_density["Density Bis"] = df_density["Gamma Strike"] / df_density["Gamma Strike"].sum() * 100

# Gaussian Density Comparison
mean, std = np.mean(df_density["Strike"]), np.std(df_density["Strike"])
rvs = np.linspace(mean - 3*std, mean + 3*std, 100)
pdf = norm.pdf(rvs, mean, std)


# Results (dataframes + graphs)
print(df_mkt)
print(df_density)
plt.plot(df_density["Strike"], df_density["IV"], label="Interpolated")
plt.scatter(df_mkt["Strike"], df_mkt["IV (12M)"], label="Implied")
plt.title("Volatilities")
plt.grid()
plt.legend()
plt.show()
plt.plot(df_density["Strike"], df_density["Density Bis"])
plt.title("Density (Gamma Strike)")
plt.grid()
plt.show()
plt.plot(df_density["Strike"], df_density["Density"])
plt.title("Density (Empirical)")
plt.grid()
plt.show()
plt.plot(rvs, pdf, c="g", label="Normal Distribution (PDF)")
plt.title("Normal Distribution")
plt.grid()
plt.show()

# --------------------------- Q2 : Metropolis–Hastings algorithm ---------------------------


















