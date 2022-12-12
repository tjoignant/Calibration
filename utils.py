import numpy as np


class Interp:
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        self.n = len(x_list)
        self.U = self.__compute_coefs()

    def get_image(self, x):
        a, b, c = 0, 0, 0
        for i in range(0, self.n - 1):
            if self.x_list[i] <= x <= self.x_list[i + 1]:
                a, b, c = self.U[3 * i], self.U[3 * i + 1], self.U[3 * i + 2]
                break
        return a * pow(x, 2) + b * x + c

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


if __name__ == '__main__':
    my_interp = Interp(x_list=[-3, -2, -1, 0, 1, 2, 3], y_list=[10, 5, 2, 0, 2, 5, 10])
    image = my_interp.get_image(x=2.5)
    print(image)
