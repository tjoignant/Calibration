import scipy
import numpy as np


class Interp1D:
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        self.n = len(x_list)

    def get_image(self, x):
        for i in range(0, self.n-1):
            if self.x_list[i] <= x <= self.x_list[i + 1]:
                return self.y_list[i] + (self.y_list[i+1] - self.y_list[i]) / (self.x_list[i+1] - self.x_list[i]) * (x - self.x_list[i])


class Interp2D:
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        self.n = len(x_list)
        self.U = self.__compute_coefs()

    def get_image(self, x):
        a, b, c = 0, 0, 0
        for i in range(0, self.n-1):
            if self.x_list[i] <= x <= self.x_list[i + 1]:
                return self.U[3 * i] * pow(x, 2) + self.U[3 * i + 1] * x + self.U[3 * i + 2]

    def __compute_coefs(self):
        # Compute V
        V = []
        for i in range(1, self.n):
            V.append(self.y_list[i])
        for i in range(0, self.n - 1):
            V.append(self.y_list[i])
        while len(V) < 3 * (self.n - 1):
            V.append(0)
        # Compute M
        matrix = []
        # P_i+1 = f(x_i+1)
        for i in range(0, self.n - 1):
            row = []
            for j in range(0, 3 * (self.n - 1)):
                if j == 3 * i:
                    row.append(pow(self.x_list[i + 1], 2))
                elif j == 3 * i + 1:
                    row.append(self.x_list[i + 1])
                elif j == 3 * i + 2:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        # P_i = f(x_i)
        for i in range(0, self.n - 1):
            row = []
            for j in range(0, 3 * (self.n - 1)):
                if j == 3 * i:
                    row.append(pow(self.x_list[i], 2))
                elif j == 3 * i + 1:
                    row.append(self.x_list[i])
                elif j == 3 * i + 2:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        # P'_i = P'_i+1
        for i in range(1, self.n - 1):
            row = []
            for j in range(0, 3 * (self.n - 1)):
                if j == 3 * (i-1):
                    row.append(2 * self.x_list[i])
                elif j == 3 * (i-1) + 1:
                    row.append(1)
                elif j == 3 * (i-1) + 3:
                    row.append(-2 * self.x_list[i])
                elif j == 3 * (i-1) + 4:
                    row.append(-1)
                else:
                    row.append(0)
            matrix.append(row)
        # P''_n/2 = 0
        row = []
        for i in range(0, 3 * (self.n - 1)):
            if i == 3 * (int(self.n / 2) - 1):
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
        M = np.matrix(matrix)
        U = np.matmul(M.I, np.array(V))
        return np.array(U)[0]


class Interp3D:
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        self.interp = scipy.interpolate.interp1d(x=self.x_list, y=self.y_list, kind='cubic')

    def get_image(self, x):
        return self.interp(x)
