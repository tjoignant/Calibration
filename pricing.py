import numpy as np

import models

def CEV_Price(k, t, f, df, sigma0, gamma, OptType):
    simulations = models.cev_sim(S0=f/df, sigma0=sigma0, gamma=gamma, drift=f - f/df, maturity=t)
    switcher = {
        "C": np.mean(np.array([max(0, df * (sim - k)) for sim in simulations])),
        "P": np.mean(np.array([max(0, df * (k - sim)) for sim in simulations])),
    }
    return switcher.get(OptType.upper(), 0)
