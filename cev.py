import copy
import random
import numpy as np
import pandas as pd
import scipy.optimize as optimize

MAX_ITERS = 500
MAX_ERROR = pow(10, -2)

def CEV_monte_carlo(S0: float, sigma0: float, gamma: float, drift: float, maturity: float, nb_simuls=1000, seed=1):
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
    np.random.seed(seed)
    nb_steps = int(maturity * 252)
    dt = maturity / nb_steps
    S = np.full(shape=(nb_steps + 1, nb_simuls), fill_value=S0)
    Z = np.random.normal(loc=0, scale=1, size=(nb_steps, nb_simuls))
    for i in range(1, nb_steps + 1):
        S[i] = S[i - 1] + S[i - 1] * drift * dt + sigma0 * np.power(S[i - 1], gamma) * np.sqrt(dt) * Z[i - 1]
    return S


def CEV_Price(k, t, f, df, OptType, gamma, sigma0):
    simulations = CEV_monte_carlo(S0=f/df, sigma0=sigma0, gamma=gamma, drift=f - f/df, maturity=t)
    switcher = {
        "C": np.mean(np.array([max(0, df * (sim - k)) for sim in simulations[-1]])),
        "P": np.mean(np.array([max(0, df * (k - sim)) for sim in simulations[-1]])),
    }
    return switcher.get(OptType.upper(), 0)


def CEV_Sigma_Nelder_Mead_1D(k, t, f, df, OptType, gamma, MktPrice):
    nb_iter = 0
    x_list = [0.2, 0.5]
    fx_list = [abs(CEV_Price(k, t, f, df, OptType, gamma, x) - MktPrice) for x in x_list]
    # Sorting
    if fx_list[1] < fx_list[0]:
        temp = x_list[0]
        x_list[0] = x_list[1]
        x_list[1] = temp
    fx_list = [abs(CEV_Price(k, t, f, df, OptType, gamma, x) - MktPrice) for x in x_list]
    while fx_list[0] > MAX_ERROR and nb_iter < MAX_ITERS:
        # Reflexion
        xr = x_list[0] + (x_list[0] - x_list[1])
        fxr = abs(CEV_Price(k, t, f, df, OptType, gamma, xr) - MktPrice)
        # Expansion
        if fxr < fx_list[0]:
            xe = x_list[0] + 2 * (x_list[0] - x_list[1])
            fxe = abs(CEV_Price(k, t, f, df, OptType, gamma, xe) - MktPrice)
            if fxe <= fxr:
                x_list = [xe, x_list[0]]
            else:
                x_list = [xr, x_list[0]]
        # Contraction
        else:
            x_list = [x_list[0], 0.5 * (x_list[0] + x_list[1])]
        # Recompute Each Error
        fx_list = [abs(CEV_Price(k, t, f, df, OptType, gamma, x) - MktPrice) for x in x_list]
        # Sorting X List
        if fx_list[1] < fx_list[0]:
            temp = x_list[0]
            x_list[0] = x_list[1]
            x_list[1] = temp
        fx_list = [abs(CEV_Price(k, t, f, df, OptType, gamma, x) - MktPrice) for x in x_list]
        # Add Nb Iter
        nb_iter = nb_iter + 1
    return x_list[0], nb_iter


def CEV_sigma_regularisation_minimisation_function(params_list: list, inputs_list: list, mktPrice_list: list):
    """
    :param params_list: [gamma_]
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1), (K_2, T_2, f_2, df_2, OptType_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, ...]
    :return: calibration error
    """
    # Compute Each Sigma0
    sigma0_list = []
    for i in range(0, len(mktPrice_list)):
        sigma0_list.append(CEV_Sigma_Nelder_Mead_1D(k=inputs_list[i][0], t=inputs_list[i][1],f=inputs_list[i][2],
                                                    df=inputs_list[i][3], OptType=inputs_list[i][4],
                                                    gamma=params_list[0], MktPrice=mktPrice_list[i])[0])
    # Build Sigma0 Surface
    df_surface = pd.DataFrame(index=range(95, 105), columns=[3, 6, 9, 12])
    for i in range(0, len(mktPrice_list)):
        df_surface.loc[inputs_list[i][0], inputs_list[i][1]] = sigma0_list[i]
    # Compute Mean Regularisation Error (MRE)
    RE = 0
    for i in range(0, len(mktPrice_list)):
        # Absolute Diff over strikes
        if inputs_list[i][0] != 104:
            RE = RE + np.power(df_surface.loc[inputs_list[i][0], inputs_list[i][1]] -
                               df_surface.loc[inputs_list[i][0] + 1, inputs_list[i][1]], 2)
        # Diff over maturities
        if inputs_list[i][1] != 1:
            RE = RE + np.power(df_surface.loc[inputs_list[i][0], inputs_list[i][1]] -
                               df_surface.loc[inputs_list[i][0], inputs_list[i][1] + 0.25], 2)
    MRE = RE / len(mktPrice_list)
    return MRE


def CEV_Gamma_Calibration_Nelder_Mead_1D(inputs_list: list, mktPrice_list: list):
    nb_iter = 0
    x_list = [0.8, 1.2]
    fx_list = [CEV_sigma_regularisation_minimisation_function([x], inputs_list, mktPrice_list) for x in x_list]
    # Sorting
    if fx_list[1] < fx_list[0]:
        temp_x = x_list[0]
        x_list[0] = x_list[1]
        x_list[1] = temp_x
        temp_fx = fx_list[0]
        fx_list[0] = fx_list[1]
        fx_list[1] = temp_fx
    while fx_list[1] - fx_list[0] > MAX_ERROR / 100 and nb_iter < MAX_ITERS:
        print(nb_iter, x_list, fx_list)
        # Reflexion
        xr = x_list[0] + (x_list[0] - x_list[1])
        fxr = CEV_sigma_regularisation_minimisation_function([xr], inputs_list, mktPrice_list)
        # Expansion
        if fxr < fx_list[0]:
            xe = x_list[0] + 2 * (x_list[0] - x_list[1])
            fxe = CEV_sigma_regularisation_minimisation_function([xe], inputs_list, mktPrice_list)
            if fxe <= fxr:
                x_list = [xe, x_list[0]]
            else:
                x_list = [xr, x_list[0]]
        # Contraction
        else:
            x_list = [x_list[0], 0.5 * (x_list[0] + x_list[1])]
        # Recompute Each Error
        fx_list = [CEV_sigma_regularisation_minimisation_function([x], inputs_list, mktPrice_list) for x in x_list]
        # Sorting
        if fx_list[1] < fx_list[0]:
            temp_x = x_list[0]
            x_list[0] = x_list[1]
            x_list[1] = temp_x
            temp_fx = fx_list[0]
            fx_list[0] = fx_list[1]
            fx_list[1] = temp_fx
        # Add Nb Iter
        nb_iter = nb_iter + 1
    return x_list[0], nb_iter


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
        SVE = SVE + pow(CEV_Price(k=inputs_list[i][0], t=inputs_list[i][1], f=inputs_list[i][2], df=inputs_list[i][3],
                                  OptType=inputs_list[i][4], gamma=params_list[0], sigma0=params_list[1])
                        - mktPrice_list[i], 2)
    MSVE = SVE / len(inputs_list)
    return MSVE


def CEV_calibration_scipy(inputs_list: list, mktPrice_list: list):
    """
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1), (K_2, T_2, f_2, df_2, OptType_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, ...]
    :return: calibrated parameters dict {gamma_: sigma0_}
    """
    result = optimize.minimize(
        CEV_minimisation_function,
        x0=[0.5, 0.5],
        method='nelder-mead',
        args=(inputs_list, mktPrice_list),
        tol=MAX_ERROR,
    )
    return list(result.x), result.fun, result.nit


def CEV_calibration_nelder_mead(inputs_list: list, mktPrice_list: list):
    """
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1), (K_2, T_2, f_2, df_2, OptType_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, ...]
    :return: calibrated parameters dict {gamma_: sigma0_}
    """
    # Initialization
    nb_iter = 0
    d = 2
    init_params = [[0.01, 1], [1, 0.5], [0.5, 0.01]]
    # Set Initial Solver Values, example for d=2 : {(a1, b1): x1, (a2, b2): x2, (a3, b3): x3}
    solver = {}
    for i in range(0, d + 1):
        params_list = [init_params[i][0], init_params[i][1]]
        solver[tuple(init_params[i])] = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                                  mktPrice_list=mktPrice_list)
    # Sorting Solver
    solver = {k: v for k, v in sorted(solver.items(), key=lambda item: item[1])}
    list_best_score = []
    list_best_score.append(copy.copy(list(solver.values())[0]))

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
                    for i in range(1, d + 1):
                        new_x = []
                        for j in range(0, d):
                            new_x.append(0.5 * (list(solver.keys())[0][j] + list(solver.keys())[i][j]))
                        params_list = [new_x[0], new_x[1]]
                        new_points[tuple(new_x)] = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                        mktPrice_list=mktPrice_list)
                    for i in range(1, d + 1):
                        solver.pop(list(solver.keys())[-1])
                    solver.update(new_points)
        # Sorting Solver
        solver = {k: v for k, v in sorted(solver.items(), key=lambda item: item[1])}
        list_best_score.append(copy.copy(list(solver.values())[0]))
        # Update Nb Iter
        nb_iter = nb_iter + 1
    return list(solver.keys())[0], list(solver.values())[0], nb_iter, list_best_score


# particle class
class Particle:
    def __init__(self, dim: int, lowx: float, uppx: float, seed: int):
        """
        :param self: this Particle
        :param dim: dimension, i.e. in our case 2 since we have sigma and gamma
        :param lowx: lower bound
        :param uppx: upper bound
        :param seed: seed for random
        :return: new Particle
        """
        self.rnd = random.Random(seed)
        # initializing position of the particle
        self.position = [0.0 for i in range(dim)]
        # initializing velocity of the particle
        self.velocity = [0.0 for i in range(dim)]
        # initializing the best particle position of the particle
        self.best_particle_position = [0.0 for i in range(dim)]
        # for every position parameter (in our case sigma and gamma) initialize position and velocity
        # range of position and velocity is [lowx, uppx]
        for i in range(dim):
            self.position[i] = ((uppx - lowx) *
                                self.rnd.random() + lowx)
            self.velocity[i] = ((uppx - lowx) *
                                self.rnd.random() + lowx)
        # initializing score of the particle
        self.score = 0  # score will be updated with CEV_minimisation_function
        # initializing best position and score of this particle
        self.best_particle_position = copy.copy(self.position)
        self.best_particle_score = self.score  # best fitness


# particle swarm optimization function
def CEV_calibration_pso(n: int, dim: int, lowx: float, uppx: float, inputs_list: list, mktPrice_list: list):
    """
    :param n: number of Particles in a swarm
    :param dim: dimension, i.e. in our case 2 since we have sigma and gamma
    :param lowx: lower bound
    :param uppx: upper bound
    :param inputs_list: [(K_1, T_1, f_1, df_1, OptType_1), (K_2, T_2, f_2, df_2, OptType_2), ...]
    :param mktPrice_list: [mktPrice_1, mktPrice_2, ...]
    :return: params_list: [gamma_, sigma0_]
    """
    # hyper parameters
    w = 0.8  # inertia
    c1 = 0.1  # cognitive (particle)
    c2 = 0.1  # social (swarm)

    rnd = random.Random(0)

    # create n random particles
    swarm = [Particle(dim, lowx, uppx, i) for i in range(n)]
    for particle in swarm:
        params_list = particle.position
        particle.score = CEV_minimisation_function(params_list=params_list, inputs_list=inputs_list,
                                                        mktPrice_list=mktPrice_list)
        particle.best_particle_score = copy.copy(particle.score)

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = 1000  # swarm best

    # compute the best particle of swarm and it's fitness
    for i in range(n):  # check each particle
        if swarm[i].score < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].score
            best_swarm_pos = copy.copy(swarm[i].position)

    previous_best_swarm_fitnessVal = 10000
    list_best_swarm_fitness = []
    list_best_swarm_fitness.append(copy.copy(best_swarm_fitnessVal))

    # main loop of pso
    iteration = 0
    while previous_best_swarm_fitnessVal - best_swarm_fitnessVal > MAX_ERROR and iteration < MAX_ITERS:

        for i in range(n):  # process each particle

            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()  # randomizations
                r2 = rnd.random()
                swarm[i].velocity[k] = (
                        (w * swarm[i].velocity[k]) +
                        (c1 * r1 * (swarm[i].best_particle_position[k] - swarm[i].position[k])) +
                        (c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k]))
                )

            # compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]
                swarm[i].position[k] = max(lowx, min(uppx, swarm[i].position[k]))

            # compute fitness of new position
            params_list_i = swarm[i].position
            swarm[i].score = CEV_minimisation_function(params_list=params_list_i, inputs_list=inputs_list,
                                                       mktPrice_list=mktPrice_list)

            # is new position a new best for the particle?
            if swarm[i].score < swarm[i].best_particle_score:
                swarm[i].best_particle_score = swarm[i].score
                swarm[i].best_particle_position = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].score < best_swarm_fitnessVal:
                previous_best_swarm_fitnessVal = copy.copy(best_swarm_fitnessVal)
                best_swarm_fitnessVal = swarm[i].score
                best_swarm_pos = copy.copy(swarm[i].position)

        # for-each particle
        iteration += 1
        list_best_swarm_fitness.append(copy.copy(best_swarm_fitnessVal))

    return best_swarm_pos, best_swarm_fitnessVal, iteration, list_best_swarm_fitness
