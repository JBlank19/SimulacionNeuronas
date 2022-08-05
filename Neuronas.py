import time
from ODESolver import *


class Neuronas:
    def __init__(self, lista_etas, N, J, V_p, I0, omega, dt, tamagnotjk, start_estimulo, end_estimulo):
        self.lista_etas = lista_etas
        self.N = N
        self.J = J
        self.V_p = V_p
        self.V_r = - V_p

        self.dt = dt

        self.I0 = I0
        self.omega = omega

        self.tau = dt * 10
        self.tamagnotjk = tamagnotjk

        self.start_estimulo = start_estimulo
        self.end_estimulo = end_estimulo

        self.rate_actual = 0

    def generador_edo(self, lista_t, lista_V, eta, J, lista_tjk):
        self.rate_actual = self.rate_instantaneo(lista_tjk)
        I = eta + J * self.rate_actual + self.pulso_externo(lista_t)
        dVdt = lista_V ** 2 + I

        return dVdt

    def solver_edo(self, lista_V, a, b, lista_tjk, random_indexes, matrix_raster, tiempo_congelado):
        # if V_j >= self.V_p
        #     resuelve la edo
        # else:
        #     registro el tiempo en el que V_j deja de cumplir la condición y lo reseteo a V_j = - V_j
        #     cuando haya pasado un tiempo: delta t = 2/V_j entonces hago cumplir la condición
        #     de nuevo haciendo V_j = V_r.

        sol = np.zeros(self.N)
        num_spikes_en_t = 0

        for j in range(len(lista_V)):
            if lista_V[j] < self.V_p and tiempo_congelado[j] == 0:
                sol[j] = euler1d1step(self.generador_edo, a, b,
                                      lista_V[j], args=(self.lista_etas[j], self.J, lista_tjk))
            else:
                sol[j] = - abs(lista_V[j])
                if abs(1/lista_V[j]) < tiempo_congelado[j] <= abs(1/lista_V[j]) + self.dt:
                    num_spikes_en_t += 1
                    tiempo_congelado[j] += self.dt

                    if j in random_indexes:
                        index = int(np.where(random_indexes == j)[0])
                        matrix_raster[index].append(a)
                elif tiempo_congelado[j] >= abs(2/lista_V[j]):
                    tiempo_congelado[j] = 0
                else:
                    tiempo_congelado[j] += self.dt

        return sol, num_spikes_en_t, matrix_raster, tiempo_congelado

    def paso_edo(self, V_init, T_init, T_fin, lista_tjk, lista_medias, lista_medias_bin, lista_rate, lista_rate_bin,
                 random_indexes, matrix_raster, tiempo_congelado):
        n = (T_fin - T_init) / self.dt
        n = rounder(n)
        n = int(n)

        start_time = time.time()

        solucion = np.zeros((self.N, n))
        solucion[:, 0] = V_init

        for i in range(1, n):
            if i == n * 0.25 or i == n * 0.50 or i == n * 0.75:
                print(f"He terminado un cuarto en {time.time() - start_time} segundos")
            t_instant = T_init + i * self.dt
            solucion[:, i], num_spikes_en_t, matrix_raster, tiempo_congelado = self.solver_edo(solucion[:, i - 1], t_instant,
                                                                             t_instant + self.dt, lista_tjk,
                                                                             random_indexes, matrix_raster, tiempo_congelado)

            q = i % self.tamagnotjk

            indicek = np.where(tiempo_congelado == 0)[0]
            lista_medias[q] = np.mean(solucion[indicek, i])

            lista_tjk[q] = num_spikes_en_t
            lista_rate[q] = self.get_rate_actual()

            if q == self.tamagnotjk - 1:
                lista_rate_bin = np.append(lista_rate_bin, np.mean(lista_rate))
                lista_medias_bin = np.append(lista_medias_bin, np.mean(lista_medias))

        print(f"He tardado: {time.time() - start_time} segundos")
        tiempo = np.linspace(T_init, T_fin, int(n / self.tamagnotjk))

        return solucion, tiempo, lista_tjk, lista_medias, lista_medias_bin, lista_rate, lista_rate_bin, matrix_raster, tiempo_congelado

    def pulso_externo(self, t):
        if self.start_estimulo < t < self.end_estimulo:
            return self.I0
        else:
            return 0
        # return self.I0 * np.sin(self.omega * t + np.pi / 4)

    def rate_instantaneo(self, lista_tjk):
        # 1/N * 1/tau * numero de spikes entre (t-tau, t)

        return (1 / self.N) * (1 / self.tau) * np.sum(lista_tjk)

    def get_rate_actual(self):
        return self.rate_actual


def rounder(x):
    if x-int(x) >= 0.5:
        return np.ceil(x)
    else:
        return np.floor(x)
