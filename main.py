import os
import sys
import time
from threading import Thread

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import TextBox

from FREs import *
from Lorentziana import *
from Neuronas import *
from ODESolver import rk4
from Neuronas import rounder

# Inicializar variables desde el fichero.
variables_fichero = []

# Se abre el fichero creado
with open(r'new_vars.txt', 'r') as fp:
    for line in fp:
        x = line
        if "." in x:
            variables_fichero.append(float(x))
        elif "True" in x:
            variables_fichero.append(bool(x))
        elif "False" in x:
            variables_fichero.append(bool(""))
        else:
            variables_fichero.append(int(x))

N, tiempo_d, V_p, eta_mu, sigma, dt, J, num_raster_plot, I0, start_estimulo, end_estimulo, show_raster = variables_fichero
new_raster_bool = show_raster

distribucion = Distribucion(eta_mu, N, sigma)
lista_etas = distribucion.Lorentziana()

random_indexes = np.linspace(0, len(lista_etas) - 1, len(lista_etas))
np.random.shuffle(random_indexes)
random_indexes = random_indexes[0:num_raster_plot]


# def initialize_distribution(es_lorentziana=True):
#     if es_lorentziana:
#         count, bins, paquetes = plt.hist(lista_etas, int(100), ec="black", density=True)


def initialize_axes(c, d):
    global ax_vmedio, ax_rate, ax_pulso, ax_texto, fig, show_raster

    fig = plt.figure()
    ax_rate = plt.axes([0.1, 0.7, 0.67, 0.2])
    ax_vmedio = plt.axes([0.1, 0.4, 0.67, 0.2])
    ax_pulso = plt.axes([0.1, 0.1, 0.67, 0.2])

    ax_rate.set_title("Rate instantáneo")
    ax_pulso.set_title("Estímulo externo")

    if show_raster:
        ax_vmedio.set_title("Raster plot")
        ax_vmedio.set_ylim(0, num_raster_plot)
    else:
        ax_vmedio.set_title("Potencial de membrana")
        ax_vmedio.set_ylim(-3, 3)

    ax_rate.set_xlim(c, d)
    ax_vmedio.set_xlim(c, d)
    ax_pulso.set_xlim(c, d)


def initialize_buttons():
    width = 0.05
    height = 0.03

    # Axes botón reset
    ax_reset = plt.axes([0.91, 0.05, width, height])
    breset = Button(ax_reset, r"Reset", color="lightcyan", hovercolor='paleturquoise')
    breset.on_clicked(reset)

    # Axes botón cerrar
    ax_cerrar = plt.axes([0.85, 0.05, width, height])
    bcerrar = Button(ax_cerrar, r"Close", color="lightcyan", hovercolor="paleturquoise")
    bcerrar.on_clicked(cerrar)

    return ax_reset, breset, ax_cerrar, bcerrar


def initialize_boxes():
    width = 0.05
    height = 0.03
    position = 0.9

    ax_rasbox = plt.axes([position - 0.1, 0.75, 0.15, 0.15])
    if show_raster:
        active = 0
    else:
        active = 1
    raster_box = RadioButtons(ax_rasbox, labels=["Raster plot", "Pot. membrana"],
                              active=active, activecolor="black")
    for circle in raster_box.circles:  # adjust radius here. The default is 0.05
        circle.set_radius(0.05)

    # Axes for sliders / boxes / etc.
    ax_N = plt.axes([position, 0.65, width, height])
    N_box = TextBox(ax_N, r'N total de neuronas. ', initial=N,
                    color="lightcyan", hovercolor='paleturquoise')
    ax_d = plt.axes([position, 0.6, width, height])
    tiempo_d_box = TextBox(ax_d, r'Tiempo total. ', initial=tiempo_d,
                           color="lightcyan", hovercolor='paleturquoise')

    ax_N_raster = plt.axes([position, 0.55, width, height])
    N_raster_box = TextBox(ax_N_raster, r'N raster plot. ', initial=num_raster_plot,
                           color="lightcyan", hovercolor='paleturquoise')
    ax_J = plt.axes([position, 0.5, width, height])
    J_box = TextBox(ax_J, r'Cte de acoplamiento. ', initial=J, color="lightcyan", hovercolor='paleturquoise')

    ax_dt = plt.axes([position, 0.45, width, height])
    dt_box = TextBox(ax_dt, r'dt. ', initial=dt, color="lightcyan", hovercolor='paleturquoise')
    ax_vp = plt.axes([position, 0.4, width, height])
    vp_box = TextBox(ax_vp, r'$V_p$. ', initial=V_p, color="lightcyan", hovercolor='paleturquoise')

    ax_eta_mu = plt.axes([position, 0.35, width, height])
    eta_mu_box = TextBox(ax_eta_mu, r'$\bar{\eta}$.  ', initial=eta_mu, color="lightcyan", hovercolor='paleturquoise')
    ax_sigma = plt.axes([position, 0.3, width, height])
    sigma_box = TextBox(ax_sigma, r'Desviación ($\eta$). ', initial=sigma, color="lightcyan", hovercolor='paleturquoise')

    ax_I0 = plt.axes([position, 0.25, width, height])
    I0_box = TextBox(ax_I0, r'Amplitud estímulo. ', initial=I0, color="lightcyan", hovercolor='paleturquoise')
    ax_start_estimulo = plt.axes([position, 0.2, width, height])
    start_estimulo_box = TextBox(ax_start_estimulo, r'Inicio estímulo. ', initial=start_estimulo, color="lightcyan",
                                 hovercolor='paleturquoise')
    ax_end_estimulo = plt.axes([position, 0.15, width, height])
    end_estimulo_box = TextBox(ax_end_estimulo, r'Fin estímulo. ', initial=end_estimulo, color="lightcyan",
                               hovercolor='paleturquoise')

    return raster_box, ax_N, N_box, ax_d, tiempo_d_box, ax_N_raster, N_raster_box, ax_dt, dt_box, ax_vp, \
           vp_box, ax_J, J_box, ax_eta_mu, eta_mu_box, ax_sigma, sigma_box, ax_I0, I0_box, ax_start_estimulo, \
           start_estimulo_box, ax_end_estimulo, end_estimulo_box


def reset(event):
    global anim
    global new_N, new_tiempo_d, new_vp, new_etam, new_sigma, new_dt, new_J, new_num_rasterplot, \
        new_I0, new_start_estimulo, new_end_estimulo, new_raster_bool

    anim.pause()
    plt.close()

    # Escribe en un archivo los parametros para reiniciar el script desde 0 con ellos.
    new_vars = [new_N, new_tiempo_d, new_vp, new_etam, new_sigma, new_dt, new_J, new_num_rasterplot,
                new_I0, new_start_estimulo, new_end_estimulo, new_raster_bool]

    # open file in write mode
    with open(r'new_vars.txt', 'w') as fp:
        for item in new_vars:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done writing variables to savefile.')

    print("Restarting!")
    os.execl(sys.executable, sys.executable, *sys.argv)


def cerrar(event):
    exit()


def submit_N(texto_N):
    global new_N
    new_N = texto_N


def submit_tiempo_d(texto_tiempo_d):
    global new_tiempo_d
    new_tiempo_d = texto_tiempo_d


def submit_num_raster_plot(texto_N_raster):
    global new_num_rasterplot
    new_num_rasterplot = texto_N_raster


def submit_J(texto_J):
    global new_J
    new_J = texto_J


def submit_dt(texto_dt):
    global new_dt
    new_dt = texto_dt


def submit_vp(texto_vp):
    global new_vp
    new_vp = texto_vp


def submit_eta_mu(texto_eta_mu):
    global new_etam
    new_etam = texto_eta_mu


def submit_sigma(texto_sigma):
    global new_sigma
    new_sigma = texto_sigma


def submit_I0(texto_I0):
    global new_I0
    new_I0 = texto_I0


def submit_start_estimulo(texto_start_estimulo):
    global new_start_estimulo
    new_start_estimulo = texto_start_estimulo


def submit_end_estimulo(texto_end_estimulo):
    global new_end_estimulo
    new_end_estimulo = texto_end_estimulo


def rasterfunc(label):
    global new_raster_bool
    if "Pot" in label:
        new_raster_bool = False
    else:
        new_raster_bool = True


c = 0
d = tiempo_d
T = d / 10  # paso de tiempo de calculo de threads
tamagnolistatjk = 10

initialize_axes(c, d)
raster_box, ax_N, N_box, ax_d, tiempo_d_box, ax_N_raster, N_raster_box, ax_dt, dt_box, ax_vp, \
vp_box, ax_J, J_box, ax_eta_mu, eta_mu_box, ax_sigma, sigma_box, ax_I0, I0_box, ax_start_estimulo, \
start_estimulo_box, ax_end_estimulo, end_estimulo_box = initialize_boxes()
ax_reset, breset, ax_cerrar, bcerrar = initialize_buttons()

# initialize_distribution()

# register the update function with each slider
N_box.on_submit(submit_N)
tiempo_d_box.on_submit(submit_tiempo_d)
N_raster_box.on_submit(submit_num_raster_plot)
dt_box.on_submit(submit_dt)
vp_box.on_submit(submit_vp)
J_box.on_submit(submit_J)
eta_mu_box.on_submit(submit_eta_mu)
sigma_box.on_submit(submit_sigma)
I0_box.on_submit(submit_I0)
start_estimulo_box.on_submit(submit_start_estimulo)
end_estimulo_box.on_submit(submit_end_estimulo)
raster_box.on_clicked(rasterfunc)

V_r = - V_p

omega = np.pi / 20

simulacion = Neuronas(lista_etas, N, J, V_p, I0, omega, dt, tamagnolistatjk, start_estimulo, end_estimulo)

# Para el fichero
new_N, new_tiempo_d, new_vp, new_etam, new_sigma, new_dt, new_J, new_num_rasterplot, new_I0, \
new_start_estimulo, new_end_estimulo = N, tiempo_d, V_p, eta_mu, sigma, dt, J, num_raster_plot, I0, start_estimulo, end_estimulo

V_i = -2
V_init = np.ones(N) * V_i
lista_tjk = np.zeros([tamagnolistatjk])
lista_medias = np.zeros([len(lista_tjk)])
lista_medias_bin = np.array([])
lista_rate = np.zeros([len(lista_tjk)])
lista_rate_bin = np.array([])
matrix_raster = [[] for _ in range(len(random_indexes))]
tiempo_congelado_init = np.zeros(N)

start_time = time.time()

solucion_activa, tiempo_activo, lista_tjk, lista_medias, lista_medias_bin, lista_rate, \
lista_rate_bin, matrix_raster, tiempo_congelado_input = \
    simulacion.paso_edo(V_init, 0, T, lista_tjk, lista_medias, lista_medias_bin, lista_rate,
                        lista_rate_bin, random_indexes, matrix_raster, tiempo_congelado_init)

# Resolvemos las FRE para comparar con resultado bruto.
n_tot = int((d - c) / dt)
tiempo_total, yy = rk4(FRE, c, d, np.array((0, V_i)), n_tot - 1, args=(1, eta_mu, J, simulacion))
sol_fre_r = yy[:, 0]
sol_fre_v = yy[:, 1]
sol_fre_r2 = np.array([sol_fre_r[0]])
sol_fre_v2 = np.array([V_i])
tiempo_total_2 = np.array(tiempo_total[0])
for i in range(len(sol_fre_r)):
    if (i + 1) % tamagnolistatjk == 0:
        sol_fre_r2 = np.append(sol_fre_r2, sol_fre_r[i])
        sol_fre_v2 = np.append(sol_fre_v2, sol_fre_v[i])
        tiempo_total_2 = np.append(tiempo_total_2, tiempo_total[i])
sol_fre_r, sol_fre_v, tiempo_total = sol_fre_r2, sol_fre_v2, tiempo_total_2

tiempo_bin = np.linspace(0, T, len(lista_rate_bin))
dibu_pulso = []

for t in tiempo_total:
    dibu_pulso.append(simulacion.pulso_externo(t))

ax_pulso.plot(tiempo_total, dibu_pulso, color="black")

line_rate, = ax_rate.plot(tiempo_bin[0], lista_rate_bin[0], lw=1, color="salmon", zorder=0)
line_rate_fre, = ax_rate.plot(tiempo_total[0], sol_fre_r[0], color="black", zorder=1)
ax_rate.set_ylim([0, max(max(lista_rate_bin), max(sol_fre_r)) + 0.1])

line_v, = ax_vmedio.plot(tiempo_total[0], lista_medias_bin[0], lw=1, color="green", zorder=3)
line_v_fre, = ax_vmedio.plot(tiempo_total[0], sol_fre_v[0], color="black", zorder=2)

lineoffsets1 = np.linspace(0.5, num_raster_plot - 0.5, num_raster_plot)
linelengths1 = np.ones(num_raster_plot)
ax_vmedio.eventplot(matrix_raster, colors="black", lineoffsets=lineoffsets1, linelengths=linelengths1, zorder=0)
rectangulo_raster = patches.Rectangle((d, 0), -d, height=num_raster_plot - 0.5, fc='white', zorder=1)
ax_vmedio.add_patch(rectangulo_raster)

i_stamp = 0
results = {}
calculation = None
n_calc_totales = d / T
n_calc_actual = 1


def calculate(prev_sol, start_time, end_time, result):
    # Aqui result es un objeto mutable (ya que los Threads no pueden devolver)
    result['sols'], result['time'], result['tjk'], result['medias'], result['lista_medias_bin'], result['rate'], result['lista_rate_bin'], \
    result['matrix_raster'], result['tiempo_congelado'] = simulacion.paso_edo(prev_sol, start_time, end_time, lista_tjk,
                                                                              lista_medias, lista_medias_bin, lista_rate,
                                                                              lista_rate_bin, random_indexes,
                                                                              matrix_raster, tiempo_congelado_input)
    # print("Thread de calculo ha finalizado")


def animate(i):
    global cuentabin, tiempo_bin, ax_vmedio
    global results, calculation, i_stamp
    global solucion_activa, tiempo_activo, lista_tjk, lista_medias, lista_medias_bin, lista_rate, lista_rate_bin, matrix_raster
    global tiempo_congelado_input
    global n_calc_totales, n_calc_actual

    # Si i es mayor que el tamaño real de la animación simplemente ploteo la última solución.
    if i > int(rounder(n_tot / tamagnolistatjk)):
        line_rate.set_data(tiempo_bin, lista_rate_bin)
        line_rate_fre.set_data(tiempo_total, sol_fre_r)
        if not show_raster:
            rectangulo_raster.set_width(-d)
            ax_vmedio.set_ylim([min(min(sol_fre_v), min(lista_medias)), max(max(lista_medias), max(sol_fre_v))])

            line_v.set_data(tiempo_bin, lista_medias_bin)
            line_v_fre.set_data(tiempo_total, sol_fre_v)

    else:
        try:
            if i - i_stamp == 1 and calculation is None and n_calc_actual < n_calc_totales:
                print("Iniciamos thread de calculo")
                n_calc_actual += 1
                results = {}
                calculation = Thread(target=calculate,
                                     args=(solucion_activa[:, -1], tiempo_activo[-1], tiempo_activo[-1] + T, results))
                calculation.start()

            elif i - len(lista_medias_bin) == 0 and calculation is not None:
                print("Uniendo threads")
                calculation.join()
                solucion_activa = results['sols']
                tiempo_activo = np.append(tiempo_activo, results['time'])
                lista_tjk = results['tjk']
                lista_medias = results['medias']
                lista_medias_bin = results['lista_medias_bin']
                lista_rate = results['rate']
                lista_rate_bin = results['lista_rate_bin']
                matrix_raster = results['matrix_raster']
                tiempo_congelado_input = results['tiempo_congelado']
                tiempo_bin = np.linspace(0, tiempo_activo[-1], len(lista_rate_bin))
                ax_vmedio.eventplot(matrix_raster, colors="black", lineoffsets=lineoffsets1,
                                    linelengths=linelengths1, zorder=-2)
                calculation = None
                i_stamp = i

        except Exception as e:
            print(e)
            exit(1)

        line_rate.set_data(tiempo_bin[:i], lista_rate_bin[:i])
        line_rate_fre.set_data(tiempo_total[:i], sol_fre_r[:i])

        if not show_raster:
            rectangulo_raster.set_width(-d)
            ax_vmedio.set_ylim([min(min(sol_fre_v), min(lista_medias)), max(max(lista_medias), max(sol_fre_v))])

            line_v.set_data(tiempo_bin[:i], lista_medias_bin[:i])
            line_v_fre.set_data(tiempo_total[:i], sol_fre_v[:i])
        else:
            line_v.set_data(-1, -1)
            line_v_fre.set_data(-1, -1)

            rectangulo_raster.set_width(-d + (i * d) / (n_tot / tamagnolistatjk))

    return line_rate, line_rate_fre, line_v, line_v_fre, rectangulo_raster, ax_vmedio


anim = FuncAnimation(fig, animate, interval=1, blit=True, repeat=False)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
fig.canvas.manager.set_window_title('Simulación de un grupo de neuronas')
plt.show()
