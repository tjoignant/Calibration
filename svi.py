import math
import numpy as np
import scipy.optimize as optimize


def Durrleman_Condition(k_list, tot_var_list, log_forward_skew_list, log_forward_convexity_list):
    return np.power(
        1 - (np.array(k_list) * np.array(log_forward_skew_list)) / (2 * np.array(tot_var_list)), 2) - \
           (np.power(np.array(log_forward_skew_list), 2) / 4) * (1 / np.array(tot_var_list) + 1 / 4) + \
           np.array(log_forward_convexity_list) / 2


def SVI(k: float, a_: float, b_: float, rho_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness (input)
    :param a_: adjusts the vertical deplacement of the smile (param)
    :param b_: adjusts the angle between left and right asymptotes (param)
    :param rho_: adjusts the orientation of the graph (param)
    :param m_: adjusts the horizontal deplacement of the smile (param)
    :param sigma_: adjusts the smoothness of the vertex (param)
    :return: total variance
    """
    return a_ + b_ * (rho_ * (k - m_) + math.sqrt(pow(k - m_, 2) + pow(sigma_, 2)))


def SVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list,
                              use_durrleman_cond: bool):
    """
    :param params_list: [a_, b_, rho_, m_, sigma_]
    :param inputs_list: [(k_1), (k_2), (k_3), ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            SVI(k=inputs_list[i][0], a_=params_list[0], b_=params_list[1], rho_=params_list[2], m_=params_list[3],
                sigma_=params_list[4]) - mktTotVar_list[i], 2)
    MSVE = SVE / len(inputs_list)
    penality = 0
    # Penality (Durrleman)
    if use_durrleman_cond:
        k_list, g_list = SVI_Durrleman_Condition(a_=params_list[0], b_=params_list[1], rho_=params_list[2],
                                                 m_=params_list[3], sigma_=params_list[4])
        penality = 0
        if min(g_list) < 0:
            penality = 10e5
    return MSVE + penality


def SVI_calibration(k_list: list, mktTotVar_list: list, weights_list: list, use_durrleman_cond: bool):
    """
    :param k_list: [k_1, k_2, k_3, ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {a_, b_, rho_, m_, sigma_}
    """
    init_params_list = [-0.01, 0.05, -0.01, 0.03, 0.3]
    inputs_list = [(k,) for k in k_list]
    result = optimize.minimize(
        SVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "a_": final_params[0],
        "b_": final_params[1],
        "rho_": final_params[2],
        "m_": final_params[3],
        "sigma_": final_params[4],
    }


def SVI_log_forward_skew(k: float, b_: float, rho_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness
    :param b_: SVI parameter
    :param rho_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :return: SVI log forward skew
    """
    return b_ * ((k - m_) / (np.sqrt(pow(k - m_, 2) + pow(sigma_, 2))) + rho_)


def SVI_log_forward_convexity(k: float, b_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness
    :param b_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :return: SVI log forward convexity
    """
    return (b_ * pow(sigma_, 2)) / (pow(pow(m_ - k, 2) + pow(sigma_, 2), 3 / 2))


def SVI_Durrleman_Condition(a_: float, b_: float, rho_: float, m_: float, sigma_: float, min_k=-1, max_k=1, nb_k=200):
    """
    :param a_: SVI parameter
    :param b_: SVI parameter
    :param rho_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    tot_var_list = [SVI(k=k, a_=a_, b_=b_, rho_=rho_, m_=m_, sigma_=sigma_) for k in k_list]
    log_forward_skew_list = [SVI_log_forward_skew(k=k, b_=b_, rho_=rho_, m_=m_, sigma_=sigma_) for k in k_list]
    log_forward_convexity_list = [SVI_log_forward_convexity(k=k, b_=b_, m_=m_, sigma_=sigma_) for k in k_list]
    return k_list, Durrleman_Condition(k_list=k_list, tot_var_list=tot_var_list,
                                       log_forward_skew_list=log_forward_skew_list,
                                       log_forward_convexity_list=log_forward_convexity_list)
