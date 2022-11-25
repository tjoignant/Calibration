import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import black_scholes
#hello test
# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 300)
pd.set_option('display.expand_frame_repr', False)

# Set Datas
strikes = list(i for i in range(95, 105))
prices = [12.40, 9.59, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]

df = pd.DataFrame()
df["Strikes"] = strikes
df["Prices"] = prices

NewtonRaphson_IV = []
for i in range(len(df)):
    sigma = black_scholes.BS_IV_Newton_Raphson(MktPrice=df["Prices"][i], df=1, f=100, k=df["Strikes"][i], t=1, OptType="C")[0]
    NewtonRaphson_IV.append(sigma)

df["Implied_Vols"] = NewtonRaphson_IV

# ----------------Interpolation----------------------
x = list(df["Strikes"])
y = list(df["Implied_Vols"])
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
xnew = np.arange(95, 104, 0.1)  # new strikes
ynew = f2(xnew)
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, ynew, '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

# ----------------New Options------------------------
df_new = pd.DataFrame()

New_Options_Prices = []
for i in range(len(xnew)):
    New_Options_Prices.append(black_scholes.BS_Price(df=1, f=100, k=xnew[i], t=1, v=ynew[i], OptType="C"))

df_new["Strikes"] = list(xnew)
df_new["Prices"] = New_Options_Prices
df_new["Implied_Vols"] = list(ynew)

df_new["Delta Strike"] = df_new["Prices"].diff(1) / df_new["Strikes"].diff(1)
df_new["Gamma Strike"] = df_new["Delta Strike"].diff(1) / df_new["Strikes"].diff(1)
df_new["Density"] = df_new["Gamma Strike"] / df_new["Gamma Strike"].sum()

df_new["Delta Strike Bis"] = df.apply(lambda x: black_scholes.BS_Delta_Strike(f=1, k=x["Strikes"]/100, t=1, v=x["Implied_Vols"], df=1, OptType="C"), axis=1)

print(df_new)
