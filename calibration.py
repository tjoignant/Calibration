from black_scholes import BS_IV_Dichotomy, BS_IV_Newton_Raphson, BS_IV_Nelder_Mead_1D, BS_IV_Nelder_Mead

maturity = 2
discount_factor = 0.99
strike = 1
forward = 1.01
price = 0.1

# 1) Dichotomy
iv_dichotomy, nb_iter = BS_IV_Dichotomy(f=forward, k=strike, t=maturity, MktPrice=price, df=discount_factor, OptType="C")
print(f"Dichotomy : {iv_dichotomy} (nb iterations: {nb_iter})")

# 2) Newton-Raphson
iv_NR, nb_iter = BS_IV_Newton_Raphson(f=forward, k=strike, t=maturity, MktPrice=price, df=discount_factor, OptType="C")
print(f"Newton-Raphson : {iv_NR} (nb iterations: {nb_iter})")

# 3) Nelder-Mead (1D)
iv_NM, nb_iter = BS_IV_Nelder_Mead_1D(f=forward, k=strike, t=maturity, MktPrice=price, df=discount_factor, OptType="C")
print(f"Nelder-Mead (1D) : {iv_NM} (nb iterations: {nb_iter})")

# 4) Nelder-Mead
iv_NM, nb_iter = BS_IV_Nelder_Mead(f=forward, k=strike, t=maturity, MktPrice=price, df=discount_factor, OptType="C")
print(f"Nelder-Mead : {iv_NM} (nb iterations: {nb_iter})")
