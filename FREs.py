# Firing Rate Equations
import numpy as np


def FRE(t, y, sigma, eta_mu, J, simulacion):
    r = y[0]
    v = y[1]
    drdt = sigma / np.pi + 2 * r * v
    dvdt = v**2 + eta_mu + J * r + simulacion.pulso_externo(t) - np.pi**2 * r**2
    return np.array((drdt, dvdt))
