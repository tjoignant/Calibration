import numpy as np


def black_scholes_sim(S0: float, volatility: float, drift: float, maturity: float, nb_simuls=100000):
    """
    Inputs:
     - S0         : initial asset spot (float)
     - volatility : asset yearly volatility (perc)
     - drift      : asset yearly drift (perc)
     - maturity   : duration of simulation (float)
     - nb_steps   : number of time steps (int)
     - nb_simuls  : number of simulations (int)
    Outputs:
     - S          : asset prices over time (2D array)
    """
    nb_steps = int(maturity * 252)
    dt = maturity / nb_steps
    S = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=S0)
    Z = np.random.normal(loc=0, scale=1, size=(nb_steps, nb_simuls))
    for i in range(1, nb_steps + 1):
        S[i] = S[i - 1] * np.exp((drift - 0.5 * pow(volatility, 2)) * dt + volatility * np.sqrt(dt) * Z[i - 1])
    return S


def cev_sim(S0: float, sigma0: float, gamma: float, drift: float, maturity: float, nb_simuls=25000):
    """
    Inputs:
     - S0         : initial asset spot (float)
     - sigma0     : initial volatility (perc)
     - gamma      : constant elasticity variance (perc)
     - drift      : asset yearly drift (perc)
     - maturity   : duration of simulation (float)
     - nb_steps   : number of time steps (int)
     - nb_simuls  : number of simulations (int)
    Outputs:
     - S          : asset prices over time (2D array)
    """
    nb_steps = int(maturity * 252)
    dt = maturity / nb_steps
    S = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=S0)
    Z = np.random.normal(loc=0, scale=1, size=(nb_steps, nb_simuls))
    for i in range(1, nb_steps + 1):
        S[i] = S[i - 1] + S[i - 1] * drift * dt + sigma0 * pow(S[i - 1], gamma) * np.sqrt(dt) * Z[i - 1]
    return S


def heston_sim(S0, v0, drift, maturity, rho, kappa, theta, sigma, nb_simuls=100000):
    """
    Inputs:
     - S0, v0    : initial asset spot and variance (float)
     - drift     : asset yearly drift (perc)
     - maturity  : duration of simulation (float)
     - rho       : correlation between asset returns and variance (float)
     - kappa     : rate of mean reversion in variance process (float)
     - theta     : long-term mean of variance process (float)
     - sigma     : vol of vol / volatility of variance process (float)
     - nb_steps  : number of time steps (int)
     - nb_simuls : number of simulations (int)
    Outputs:
     - asset prices over time (2D array)
     - variance over time (2D array)
    """
    nb_steps = int(maturity * 252)
    dt = maturity / nb_steps
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])
    S = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=S0)
    v = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=v0)
    Z = np.random.multivariate_normal(mu, cov, (nb_steps, nb_simuls))
    for i in range(1, nb_steps + 1):
        S[i] = S[i - 1] * np.exp((drift - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i] = np.max(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1], 0)
    return S, v
