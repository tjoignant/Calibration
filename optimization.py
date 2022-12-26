import models
import numpy as np

MAX_ITERS = 1000
MAX_ERROR = pow(10, -5)


def CEV_price(k, t, f, df, OptType, gamma_, sigma0_):
    simulations = models.cev_sim(S0=f/df, sigma0=sigma0_, gamma=gamma_, drift=f - f/df, maturity=t)
    switcher = {
        "C": np.mean(np.array([max(0, df * (sim - k)) for sim in simulations[-1]])),
        "P": np.mean(np.array([max(0, df * (k - sim)) for sim in simulations[-1]])),
    }
    return switcher.get(OptType.upper(), 0)


def CEV_minimisation_function(params_list: list, inputs_list: list, mktPrice_list: list):
    """
    :param params_list: [gamma_, sigma0_]
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1), (K_2, T_2, f_2, df_2, OptType_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, ...]
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(mktPrice_list)):
        SVE = SVE + pow(CEV_price(k=inputs_list[i][0], t=inputs_list[i][1], f=inputs_list[i][2], df=inputs_list[i][3],
                                  OptType=inputs_list[i][4], gamma_=params_list[0], sigma0_=params_list[1])
                        - mktPrice_list[i], 2)
    MSVE = SVE / len(inputs_list)
    return MSVE


def CEV_nelder_mead(inputs_list: list, mktPrice_list: list):
    """
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1), (K_2, T_2, f_2, df_2, OptType_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, ...]
    :return: params_list: [gamma_, sigma0_]
    """
    # Initialization
    nb_iter = 0
    d = 2
    init_params = [[0.01, 0.05], [0.99, 0.7], [0.5, 0.35]]
    # Set Initial Values, example for d=2 : {(a1, b1): x1, (a2, b2): x2, (a3, b3): x3}
    solver = {}
    for i in range(0, d + 1):
        params_list = [init_params[i][0], init_params[i][1]]
        solver[tuple(init_params[i])] = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                                  mktPrice_list=mktPrice_list)
    # Sorting
    solver = {k: v for k, v in sorted(solver.items(), key=lambda item: item[1])}
    while list(solver.values())[1] - list(solver.values())[0] > MAX_ERROR and nb_iter < MAX_ITERS:
        # Compute The Barycenter (excluding d+1 params vector)
        x0 = []
        for i in range(0, d):
            x = 0
            for j in range(0, d):
                x = x + list(solver.keys())[j][i]
            x0.append(x/d)
        # Compute Reflexion Point
        xr = []
        for i in range(0, d):
            xr.append(x0[i] + (x0[i] - list(solver.keys())[-1][i]))
        params_list = [xr[0], xr[1]]
        fxr = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list, mktPrice_list=mktPrice_list)
        # If Reflexion Point is the best
        if fxr < list(solver.values())[0]:
            # Compute Expansion Point
            xe = []
            for i in range(0, d):
                xe.append(x0[i] + 2 * (x0[i] - list(solver.keys())[-1][i]))
            params_list = [xe[0], xe[1]]
            fxe = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                            mktPrice_list=mktPrice_list)
            # Drop Previous Worst Point
            solver.pop(list(solver.keys())[-1])
            # If Expansion Point is better than Reflexion Point, Add Expansion Point
            if fxe <= fxr:
                solver[tuple(xe)] = fxe
            # Else, Add Reflexion Point
            else:
                solver[tuple(xr)] = fxr
        # Else, If Reflexion Point is better than second to last (Keeping the reflexion)
        elif fxr < list(solver.values())[-2]:
            # Drop Previous Worst Point
            solver.pop(list(solver.keys())[-1])
            # Add Reflexion Point
            solver[tuple(xr)] = fxr
        # Else
        else:
            # If Reflexion Point is in Between second to last and last (inside contraction)
            if list(solver.values())[-2] <= fxr <= list(solver.values())[-1]:
                # Compute Contraction Point
                xc = []
                for i in range(0, d):
                    xc.append(0.5 * (x0[i] + xr[i]))
                params_list = [xc[0], xc[1]]
                fxc = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                mktPrice_list=mktPrice_list)
                # If Contraction Point is better than reflexion Point
                if fxc <= fxr:
                    # Drop Previous Worst Point
                    solver.pop(list(solver.keys())[-1])
                    # Add Contraction Point
                    solver[tuple(xc)] = fxc
                # Else (xc no good) --> contracting every point towards best one
                else:
                    new_points = {}
                    for i in range(1, d+1):
                        new_x = []
                        for j in range(0, d):
                            new_x.append(0.5 * (list(solver.keys())[0][j] + list(solver.keys())[i][j]))
                        params_list = [new_x[0], new_x[1]]
                        new_points[tuple(new_x)] = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                        mktPrice_list=mktPrice_list)
                    for i in range(1, d + 1):
                        solver.pop(list(solver.keys())[-1])
                    solver.update(new_points)
            # Else If Reflexion Point is worst than previous worst
            else:
                # Compute Contraction Point
                xc = []
                for i in range(0, d):
                    xc.append(0.5 * (x0[i] + list(solver.keys())[-1][i]))
                params_list = [xc[0], xc[1]]
                fxc = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                mktPrice_list=mktPrice_list)
                if fxc <= list(solver.values())[-1]:
                    # Drop Previous Worst Point
                    solver.pop(list(solver.keys())[-1])
                    # Add Contraction Point
                    solver[tuple(xc)] = fxc
                # Else (xc no good) --> contracting every point towards best one
                else:
                    new_points = {}
                    for i in range(1, d+1):
                        new_x = []
                        for j in range(0, d):
                            new_x.append(0.5 * (list(solver.keys())[0][j] + list(solver.keys())[i][j]))
                        params_list = [new_x[0], new_x[1]]
                        new_points[tuple(new_x)] = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                        mktPrice_list=mktPrice_list)
                    for i in range(1, d + 1):
                        solver.pop(list(solver.keys())[-1])
                    solver.update(new_points)
        # Sorting
        solver = {k: v for k, v in sorted(solver.items(), key=lambda item: item[1])}
        # Add Nb Iter
        nb_iter = nb_iter + 1
        print(f"{nb_iter} - {list(solver.values())[0]}")
    return list(solver.keys())[0], nb_iter
