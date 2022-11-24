from black_scholes import BS_Delta, BS_Gamma, BS_IV_Newton_Raphson, BS_Vega

import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 300)
pd.set_option('display.expand_frame_repr', False)

# Set Constant
nb_contract = 1
maturity = dt.datetime(day=31, month=12, year=2018)
tick_font_size = 8.5
title_font_size = 11
legend_loc = "upper right"

# Load Datas
df = pd.read_excel("datas/TD3.xlsx", header=2)

# Rename Columns
df.columns = ["Date", "Spot Price", "Call Price (K=6)", "Call Price (K=6.5)"]

# Drop Useless Rows
df = df.iloc[2:, :]
df = df.reset_index(drop=True)

# Add Contract/Underlying Columns
df["Maturity"] = maturity
df["Maturity (in Y)"] = df.apply(lambda x: (x["Maturity"] - x["Date"]).days / 365, axis=1)
df["Forward"] = df["Spot Price"]
df["Discount Factor"] = 1

# Compute Options Implied Vol
df["Call IV (K=6)"] = df.apply(lambda x: BS_IV_Newton_Raphson(f=x["Forward"], k=6, t=x["Maturity (in Y)"],
                                                                 df=x["Discount Factor"], OptType="C",
                                                                 MktPrice=x["Call Price (K=6)"])[0], axis=1)
df["Call IV (K=6.5)"] = df.apply(lambda x: BS_IV_Newton_Raphson(f=x["Forward"], k=6.5, t=x["Maturity (in Y)"],
                                                                   df=x["Discount Factor"], OptType="C",
                                                                   MktPrice=x["Call Price (K=6.5)"])[0], axis=1)
# Compute Diff IV
df["Call Diff IV (K=6)"] = df["Call IV (K=6)"].diff(1)
df["Call Diff IV (K=6.5)"] = df["Call IV (K=6.5)"].diff(1)

# Compute DIV IV Regression (before & after crash)
df_bis = df[df["Date"] <= pd.to_datetime("2018-05-27")].copy()
a_bc, b_bc = np.polyfit(df_bis["Call Diff IV (K=6)"].array[1:], df_bis["Call Diff IV (K=6.5)"].array[1:], 1)
df_bis = df[df["Date"] > pd.to_datetime("2018-05-27")].copy()
a_ac, b_ac = np.polyfit(df_bis["Call Diff IV (K=6)"].array, df_bis["Call Diff IV (K=6.5)"].array, 1)

print(a_bc, b_bc)
print(a_ac, b_ac)

# Compute Options Gamma
df["Call Gamma (K=6)"] = df.apply(lambda x: BS_Gamma(f=x["Forward"], k=6, t=x["Maturity (in Y)"],
                                                        v=x["Call IV (K=6)"], df=x["Discount Factor"],
                                                        OptType="C"), axis=1)
df["Call Gamma (K=6.5)"] = df.apply(lambda x: BS_Gamma(f=x["Forward"], k=6.5, t=x["Maturity (in Y)"],
                                                          v=x["Call IV (K=6.5)"], df=x["Discount Factor"],
                                                          OptType="C"), axis=1)

# Compute Options Vega
df["Call Vega (K=6)"] = df.apply(lambda x: BS_Vega(f=x["Forward"], k=6, t=x["Maturity (in Y)"],
                                                        v=x["Call IV (K=6)"], df=x["Discount Factor"],
                                                        OptType="C"), axis=1)
df["Call Vega (K=6.5)"] = df.apply(lambda x: BS_Vega(f=x["Forward"], k=6.5, t=x["Maturity (in Y)"],
                                                          v=x["Call IV (K=6.5)"], df=x["Discount Factor"],
                                                          OptType="C"), axis=1)

# Compute Options Vol Ratios (Call K=6.5 / Call K=6)
df["IV Ratio"] = df["Call IV (K=6.5)"] / df["Call IV (K=6)"]
df["Vega Ratio"] = df["Call Vega (K=6.5)"] / df["Call Vega (K=6)"]
df["IV Ratio / Vega Ratio"] = df["IV Ratio"] / df["Vega Ratio"]

# Compute Options Delta
df["Call Delta (K=6)"] = df.apply(lambda x: BS_Delta(f=x["Forward"], k=6, t=x["Maturity (in Y)"],
                                                        v=x["Call IV (K=6)"], df=x["Discount Factor"],
                                                        OptType="C"), axis=1)
df["Call Delta (K=6.5)"] = df.apply(lambda x: BS_Delta(f=x["Forward"], k=6.5, t=x["Maturity (in Y)"],
                                                          v=x["Call IV (K=6.5)"], df=x["Discount Factor"],
                                                          OptType="C"), axis=1)

# Compute Vega Hedged Call Positions
df["Call Pos (K=6)"] = -nb_contract
df["Call Real Vega (K=6.5)"] = df["Call Vega (K=6.5)"] * df.apply(lambda x: a_ac if x["Date"] > pd.to_datetime("2018-05-27") else a_bc, axis=1)
df["Call Pos (K=6.5)"] = df.apply(lambda x: -(x["Call Vega (K=6)"]/x["Call Real Vega (K=6.5)"])*x["Call Pos (K=6)"], axis=1)

# Compute Remaining Delta
df["Remaining Delta"] = df["Call Pos (K=6)"] * df["Call Delta (K=6)"] + df["Call Pos (K=6.5)"] * df["Call Delta (K=6.5)"]
df["Total Delta"] = - df["Remaining Delta"]

# Compute Replication Portfolio Value
df["Delta Hedged Replication P&L"] = df["Call Delta (K=6)"].shift(1) * df["Spot Price"].diff()
df["Delta/Vega Hedged Replication P&L"] = df["Call Pos (K=6.5)"].shift(1) * df["Call Price (K=6.5)"].diff() + df["Total Delta"].shift(1) * df["Spot Price"].diff()
df.loc[0, "Delta Hedged Replication P&L"] = df.iloc[0]["Call Price (K=6)"]
df.loc[0, "Delta/Vega Hedged Replication P&L"] = df.iloc[0]["Call Price (K=6)"]
df["Delta Hedged Replication"] = df["Delta Hedged Replication P&L"].cumsum()
df["Delta/Vega Hedged Replication"] = df["Delta/Vega Hedged Replication P&L"].cumsum()

# Print Results
print(df)
print("\nI Sell :")
print(f" ° {nb_contract} Call (K=6)")
print("\nI Buy (delta hedged replication) :")
print(f" ° {round(df.iloc[-1]['Call Delta (K=6)'], 16)} Shares")
print("\nI Buy (delta & vega hedged replication) :")
print(f" ° {round(df.iloc[-1]['Call Pos (K=6.5)'], 16)} Call (K=6.5)")
print(f" ° {round(df.iloc[-1]['Total Delta'], 16)} Shares")

# Show Graphs
fig1, axs1 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5))
fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 7.5))
fig1.suptitle(f"Parameters Calibration", fontweight='bold', fontsize=12.5)
fig2.suptitle(f"Portfolio Replication", fontweight='bold', fontsize=12.5)

# Plot Implied Volatilities Evolution
axs1[0, 0].plot(df["Date"], df["Call IV (K=6)"], label="Call (K=6)")
axs1[0, 0].plot(df["Date"], df["Call IV (K=6.5)"], label="Call (K=6.5)")
axs1[0, 0].set_title("Implied Volatilities Evolution")
axs1[0, 0].grid()
axs1[0, 0].legend(loc=legend_loc)
axs1[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)

# Plot Implied Volatilities
axs1[1, 0].plot(df["Date"], df["Call Vega (K=6)"], label="Call (K=6)")
axs1[1, 0].plot(df["Date"], df["Call Vega (K=6.5)"], label="Call (K=6.5)")
axs1[1, 0].set_title("Implied Vegas Evolution")
axs1[1, 0].grid()
axs1[1, 0].legend(loc=legend_loc)
axs1[1, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)

# Plot Volatility Ratio Evolution
axs1[0, 1].plot(df["Date"], df["Vega Ratio"], label="Vega Ratio")
axs1[0, 1].plot(df["Date"], df["IV Ratio"], label="IV Ratio")
axs1[0, 1].grid()
axs1[0, 1].set_title("Volatility Ratio Evolution")
axs1[0, 1].legend(loc=legend_loc)
axs1[0, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)

# Plot Ratio of Vol Ratio Evolution
axs1[1, 1].plot(df["Date"], df["IV Ratio / Vega Ratio"])
axs1[1, 1].grid()
axs1[1, 1].set_title("IV Ratio / Vega Ratio")
axs1[1, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)

# Plot Replications Portfolios
axs2.plot(df["Date"], df["Call Price (K=6)"], label="Call Price")
axs2.plot(df["Date"], df["Delta Hedged Replication"], label="Delta Hedged Replication")
axs2.plot(df["Date"], df["Delta/Vega Hedged Replication"], label="Delta/Vega Hedged Replication")
axs2.grid()
axs2.legend(loc=legend_loc)
axs2.tick_params(axis='both', which='major', labelsize=tick_font_size)

# Plot Diff IV Call (K=6.5) / Diff IV Call (K=6) (before crash)
df_bis = df[df["Date"] <= pd.to_datetime("2018-06-01")].copy()
axs1[0, 2].scatter(x=df_bis["Call Diff IV (K=6)"], y=df_bis["Call Diff IV (K=6.5)"], c=df_bis["Maturity (in Y)"],
                  cmap="Spectral", vmin=df["Maturity (in Y)"].min(), vmax=df["Maturity (in Y)"].max())
axs1[0, 2].grid()
axs1[0, 2].set_title("Diff IV Call - Before Crash")
axs1[0, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)

# Plot Diff IV Call (K=6.5) / Diff IV Call (K=6) (after crash)
df_bis = df[df["Date"] > pd.to_datetime("2018-06-01")].copy()
axs1[1, 2].scatter(x=df_bis["Call Diff IV (K=6)"], y=df_bis["Call Diff IV (K=6.5)"], c=df_bis["Maturity (in Y)"],
                  cmap="Spectral", vmin=df["Maturity (in Y)"].min(), vmax=df["Maturity (in Y)"].max())
axs1[1, 2].grid()
axs1[1, 2].set_title("Diff IV - After Crash")
axs1[1, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)

# Add Color Bar
cmap = plt.get_cmap("Spectral")
norm = plt.Normalize(df["Maturity (in Y)"].min(), df["Maturity (in Y)"].max())
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=axs1[:, 2])
cbar.ax.set_title("Maturity")

# Export Dataframe
if not os.path.exists('results'):
    os.makedirs('results')
with pd.ExcelWriter("results/Results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Dataframe")

# Show Graph
plt.show()
