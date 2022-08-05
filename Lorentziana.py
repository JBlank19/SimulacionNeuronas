import numpy as np


class Distribucion:
    def __init__(self, mu, N, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.N = N

    def Lorentziana(self):
        """
        Devuelve una istribución lorentziana de etas determinista
        """

        lista_etas = []
        for j in range(1, self.N+1):
            lista_etas = np.append(lista_etas, self.mu + self.sigma * np.tan(np.pi / 2 * (2 * j - self.N - 1) / (self.N + 1)))
        return lista_etas
