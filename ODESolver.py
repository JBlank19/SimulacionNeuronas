import numpy as np


def heun(f, a, b, r0, n=1000, args=()):
    # Calcularemos la solucion dr/dt=f(t,r) con valor inicial r(a)=r0, y con paso h=(b-a)/N, usando Heun.
    # Notese que r0 tiene que ser un vector, esto es, un np.array, en cuyo caso r(t) sera una funcion vectorial,
    # de la que nos devolvera el metodo: r(0), r(h), r(2h),...r(Nh=b). Por lo tanto, lo que nos devuelve el metodo
    # es una matriz "T" con N+1 filas y numero de columnas igual a la dimension de r0.

    # Fijamos el tamaño de sol (Columnas: osciladores, filas: tiempo)
    sol = np.zeros((n, r0.size))

    # Se escribe la primera fila de la solución como el valor inicial dado a la función
    sol[0, :] = r0

    # El valor de h
    h = (b - a) / n

    for i in range(n-1):
        # Primero calculamos el valor medio de y_i+1
        y_medio = sol[i, :] + h * f(a + i * h, sol[i, :], *args)
        # Después se calcula la aproximación final de y_i+1
        sol[i + 1, :] = sol[i, :] + h/2 * (f(a + i * h, sol[i, :], *args) + f(a + (i + 1) * h, y_medio, *args))

    tiempo = np.linspace(a, b, n)
    return tiempo, sol.T


def heun1d1step(f, a, b, r0, args=()):
    h = b-a

    y_medio = r0 + h * f(a + h, r0, *args)
    sol = r0 + h / 2 * (f(a + h, r0, *args) + f(a + 2 * h, y_medio, *args))

    return sol


def euler(f, a, b, r0, n=1000, args=()):
    # Calcularemos la solucion dr/dt=f(t,r) con valor inicial r(a)=r0, y con paso h=(b-a)/N, usando Euler.
    # Notese que r0 tiene que ser un vector, esto es, un np.array, en cuyo caso r(t) sera una funcion vectorial,
    # de la que nos devolvera el metodo: r(0), r(h), r(2h),...r(Nh=b). Por lo tanto, lo que nos devuelve el metodo
    # es una matriz "T" con N+1 filas y numero de columnas igual a la dimension de r0.

    # Fijamos el tamaño de sol (Columnas: osciladores, filas: tiempo)
    sol = np.zeros((n, r0.size))

    # Se escribe la primera fila de la solución como el valor inicial dado a la función
    sol[0, :] = r0

    # El valor de h
    h = (b - a) / n

    for i in range(n-1):
        sol[i + 1, :] = sol[i, :] + h * f(a + i * h, sol[i, :], *args)  # Formula del metodo de Euler

    tiempo = np.linspace(a, b, n)
    return tiempo, sol.T


def euler1d1step(f, a, b, r0, args=()):
    h = b - a
    return r0 + h * f(b, r0, *args)


def rk4(f, a, b, r0, n=999, args=()):
    # Calcularemos la solucion dr/dt=f(t,r) con valor inicial r(a)=r0, y con paso h=(b-a)/N, usando RK4.
    # Notese que r0 tiene que ser un vector, esto es, un np.array, en cuyo caso r(t) sera una funcion vectorial,
    # de la que nos devolvera el metodo: r(0), r(h), r(2h),...r(Nh=b).
    # Por lo tanto, lo que nos devuelve el metodo es una matriz "T" con N+1 filas y
    # numero de columnas igual a la dimension de r0.

    # Fijamos el tamaño de sol (Columnas: osciladores, filas: tiempo)
    sol = np.zeros((n + 1, r0.size))

    # Se escribe la primera fila de la solución como el valor inicial dado a la función
    sol[0, :] = r0

    # Calculamos el valor de h
    h = (b - a) / n

    # Hacemos el bucle de RK4
    for k in range(1, n + 1):
        k1 = h * f((k - 1) * h, sol[k - 1, :], *args)
        k2 = h * f((k - 1 / 2) * h, sol[k - 1, :] + k1 / 2, *args)
        k3 = h * f((k - 1 / 2) * h, sol[k - 1, :] + k2 / 2, *args)
        k4 = h * f(k * h, sol[k - 1, :] + k3, *args)
        sol[k, :] = sol[k - 1, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    tiempo = np.linspace(a, b, n + 1)
    return tiempo, sol
