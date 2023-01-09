import numpy as np
from scipy.stats import norm

MAX_ITERS = 10000
MAX_ERROR = pow(10, -6)
EPS = 0.01


def BS_d1(f, k, t, v):
    return (np.log(f / k) + v * v * t / 2) / v / np.sqrt(t)

def BS_d2(f, k, t, v):
    return BS_d1(f, k, t, v) - v * np.sqrt(t)

def NormalDistrib(z, mean=0, stdev=1):
    return norm.pdf(z, loc=mean, scale=stdev)

def SNorm(z):
    return norm.cdf(z)

def BS_Price(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    switcher = {
        "C": df * (f * SNorm(d1) - k * SNorm(d2)),
        "P": df * (-f * SNorm(-d1) + k * SNorm(-d2)),
        "C+P": df * (f * SNorm(d1) - SNorm(-d1)) - k * (SNorm(d2) - SNorm(-d2)),
        "C-P": df * (f * SNorm(d1) + SNorm(-d1)) - k * (SNorm(d2) + SNorm(-d2)),
    }
    return switcher.get(OptType.upper(), 0)

def BS_Delta(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    switcher = {
        "C": df * SNorm(d1),
        "P": -df * SNorm(-d1),
        "C+P": df * (SNorm(d1) - SNorm(-d1)),
        "C-P": df * (SNorm(d1) + SNorm(-d1)),
    }
    return switcher.get(OptType.upper(), 0)

def BS_Delta_Strike(f, k, t, v, df, OptType):
    d2 = BS_d2(f, k, t, v)
    switcher = {
        "C": -df * SNorm(d2),
        "P": df * SNorm(-d2),
        "C+P": df * (-SNorm(d2) + SNorm(-d2)),
        "C-P": df * (-SNorm(d2) - SNorm(-d2)),
    }
    return switcher.get(OptType.upper(), 0)

def BS_Theta(f, k, t, v, df, r, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    switcher = {
        "C": -df * ((f * SNorm(d1) * v) / (2 * np.sqrt(t)) - (r * f * SNorm(d1)) + (r * k * SNorm(d2))),
        "P": -df * ((f * SNorm(d1) * v) / (2 * np.sqrt(t)) + (r * f * SNorm(-d1)) - (r * k * SNorm(-d2))),
        "C+P": BS_Theta(f, k, t, v, df, r, "C") + BS_Theta(f, k, t, v, df, r, "P"),
        "C-P": BS_Theta(f, k, t, v, df, r, "C") - BS_Theta(f, k, t, v, df, r, "P"),
    }
    return switcher.get(OptType.upper(), 0)

def BS_Gamma(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": (df * fd1) / (f * v * np.sqrt(t)),
        "P": (df * fd1) / (f * v * np.sqrt(t)),
        "C+P": 2 * df * fd1 / (f * v * np.sqrt(t)),
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)

def BS_Gamma_Strike(f, k, t, v, df, OptType):
    d2 = BS_d2(f, k, t, v)
    fd1 = NormalDistrib(d2)
    switcher = {
        "C": -(df * fd1) / (f * v * np.sqrt(t)),
        "P": -(df * fd1) / (f * v * np.sqrt(t)),
        "C+P": -2 * df * fd1 / (f * v * np.sqrt(t)),
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)

def BS_Vega(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": df * f * fd1 * np.sqrt(t),
        "P": df * f * fd1 * np.sqrt(t),
        "C+P": 2 * df * f * fd1 * np.sqrt(t),
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)

def BS_Vanna(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": -df * fd1 * d2 / v,
        "P": -df * fd1 * d2 / v,
        "C+P": -2 * df * fd1 * d2 / v,
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)

def BS_Volga(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "P": df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "C+P": 2 * df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)

def BS_IV_Dichotomy(f, k, t, MktPrice, df, OptType):
    nb_iter = 0
    v_list = [0.01, 1]
    v = (v_list[0] + v_list[1]) / 2
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        v = (v_list[0] + v_list[1]) / 2
        func = MktPrice - BS_Price(f, k, t, v, df, OptType)
        if func > 0:
            v_list[0] = v
        else:
            v_list[1] = v
        nb_iter = nb_iter + 1
    return v, nb_iter


def BS_IV_Newton_Raphson(f, k, t, MktPrice, df, OptType):
    nb_iter = 0
    v = 0.30
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        veg = BS_Vega(f, k, t, v, df, OptType)
        if veg == 0:
            return -1
        else:
            v = v + func / veg
            func = MktPrice - BS_Price(f, k, t, v, df, OptType)
            nb_iter = nb_iter + 1
    return v, nb_iter


def BS_IV_Nelder_Mead_1D(f, k, t, MktPrice, df, OptType):
    nb_iter = 0
    x_list = [0.01, 1]
    fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
    # Sorting
    if fx_list[1] < fx_list[0]:
        temp = x_list[0]
        x_list[0] = x_list[1]
        x_list[1] = temp
    fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
    while fx_list[0] > MAX_ERROR and nb_iter < MAX_ITERS:
        # Reflexion
        xr = x_list[0] + (x_list[0] - x_list[1])
        fxr = abs(BS_Price(f, k, t, xr, df, OptType) - MktPrice)
        # Expansion
        if fxr < fx_list[0]:
            xe = x_list[0] + 2 * (x_list[0] - x_list[1])
            fxe = abs(BS_Price(f, k, t, xe, df, OptType) - MktPrice)
            if fxe <= fxr:
                x_list = [xe, x_list[0]]
            else:
                x_list = [xr, x_list[0]]
        # Contraction
        else:
            x_list = [x_list[0], 0.5 * (x_list[0] + x_list[1])]
        # Recompute Each Error
        fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
        # Sorting X List
        if fx_list[1] < fx_list[0]:
            temp = x_list[0]
            x_list[0] = x_list[1]
            x_list[1] = temp
        fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
        # Add Nb Iter
        nb_iter = nb_iter + 1
    return x_list[0], nb_iter


def BS_IV_Nelder_Mead(f, k, t, MktPrice, df, OptType):
    nb_iter = 0
    d = 1
    init_params = [[0.01], [1]]
    solver = {}
    # Set Initial Values, example for d=2 : {(a1, b1): x1, (a2, b2): x2, (a3, b3): x3}
    for i in range(0, d + 1):
        solver[tuple(init_params[i])] = abs(BS_Price(f, k, t, init_params[i][0], df, OptType) - MktPrice)
    # Sorting
    solver = {k: v for k, v in sorted(solver.items(), key=lambda item: item[1])}
    while list(solver.values())[0] > MAX_ERROR and nb_iter < MAX_ITERS:
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
        fxr = abs(BS_Price(f, k, t, xr[0], df, OptType) - MktPrice)
        # If Reflexion Point is the best
        if fxr < list(solver.values())[0]:
            # Compute Expansion Point
            xe = []
            for i in range(0, d):
                xe.append(x0[i] + 2 * (x0[i] - list(solver.keys())[-1][i]))
            fxe = abs(BS_Price(f, k, t, xe[0], df, OptType) - MktPrice)
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
                fxc = abs(BS_Price(f, k, t, xc[0], df, OptType) - MktPrice)
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
                        new_points[tuple(new_x)] = abs(BS_Price(f, k, t, new_x[0], df, OptType) - MktPrice)
                    for i in range(1, d + 1):
                        solver.pop(list(solver.keys())[-1])
                    solver.update(new_points)
            # Else If Reflexion Point is worst than previous worst
            else:
                # Compute Contraction Point
                xc = []
                for i in range(0, d):
                    xc.append(0.5 * (x0[i] + list(solver.keys())[-1][i]))
                fxc = abs(BS_Price(f, k, t, xc[0], df, OptType) - MktPrice)
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
                        new_points[tuple(new_x)] = abs(BS_Price(f, k, t, new_x[0], df, OptType) - MktPrice)
                    for i in range(1, d + 1):
                        solver.pop(list(solver.keys())[-1])
                    solver.update(new_points)
        # Sorting
        solver = {k: v for k, v in sorted(solver.items(), key=lambda item: item[1])}
        # Add Nb Iter
        nb_iter = nb_iter + 1
    return list(solver.keys())[0][0], nb_iter
