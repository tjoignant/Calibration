import numpy as np
import scipy.optimize as optimize

import models


def CEV_price(k, t, f, df, OptType, gamma, sigma0):
    simulations = models.cev_sim(S0=f/df, sigma0=sigma0, gamma=gamma, drift=f - f/df, maturity=t)
    switcher = {
        "C": np.mean(np.array([max(0, df * (sim - k)) for sim in simulations[-1]])),
        "P": np.mean(np.array([max(0, df * (k - sim)) for sim in simulations[-1]])),
    }
    return switcher.get(OptType.upper(), 0)


def CEV_minimisation_gamma_function(params_list: list, inputs_list: list, mktPrice_list: list):
    """
    :param params_list: [sigma0_1, sigma0_2, sigma0_3]
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1, gamma_1), (K_2, T_2, f_2, df_2, OptType_2, gamma_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, mktPrice_3, ...]
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(mktPrice_list)):
        SVE = SVE + pow(CEV_price(k=inputs_list[i][0], t=inputs_list[i][1], f=inputs_list[i][2], df=inputs_list[i][3],
                                  OptType=inputs_list[i][4], gamma=inputs_list[i][5], sigma0=params_list[i])
                        - mktPrice_list[i], 2)
    MSVE = SVE / len(inputs_list)
    print(MSVE)
    # Penality (Regularization)
    penality = 0
    return MSVE + penality


def CEV_calibration_sigma(mktPrice_list: list, impVol_list: list, strike_list: list, maturity_list: list,
                          forward_list: list, df_list: list, option_type_list: list, gamma_: float):
    """
    :param mktPrice_list: [mktPrice_1, mktPrice_2, mktPrice_3, ...]
    :param impVol_list: [impVol_1, impVol_2, impVol_3, ...]
    :param strike_list: [K_1, K_2, K_3, ...]
    :param maturity_list: [T_1, T_2, T_3, ...]
    :param forward_list: [f_1, f_2, f_3, ...]
    :param df_list: [df_1, df_2, df_3, ...]
    :param option_type_list: [OptType_1, OptType_2, OptType_3, ...]
    :param gamma_: constant elasticity variance (param)
    :return: calibrated parameters dict {sigma0}
    """
    init_params_list = impVol_list
    inputs_list = [(K, T, f, df, type, gamma_) for K, T, f, df, type in zip(strike_list, maturity_list, forward_list,
                                                                            df_list, option_type_list)]
    result = optimize.minimize(
        CEV_minimisation_gamma_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktPrice_list),
        tol=1e-2,
    )
    final_params = list(result.x)
    return {
        "sigma0_": final_params[0],
    }
