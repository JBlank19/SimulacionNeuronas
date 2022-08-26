Applet del comportamiento de una población de neuronas por Josu Blanco y Elena del Campo. 2022.
Este applet simula la interacción de una población de N neuronas mediante un modelo “integrate and fire” de tipo cuadrático (QIF) y el sistema de EDOs (FREs) desarrollado en el paper de E. Montbrió, D. Pazó y A. Roxin (https://doi.org/10.48550/arXiv.1506.06581)

# Librerías requeridas

Python 3.10
Numpy, matplotlib y PyQt5

# Instalación

El archivo de descarga consiste en 5 .py Python3 y 2 .txt. Para hacer funcionar el código tan solo hay que ejecutar main.py. La descarga será:
Windows:
 1. Abre la terminal escribiendo cmd en la barra de búsqueda.
 2. Ve al directorio donde está main.py escribiendo cd C:...\main.py (cambiándolo esto según sea el camino correcto).
 3. Escribe python main.py
Linux:
 1. Abre la terminal
 2. Ve al directorio donde está main.py escribiendo cd C:...\main.py (cambiándolo esto según sea el camino correcto).
 3. Ejecuta el código con ./main.py

# Uso

El applet está dividido en cuatro partes.
Arriba a la izquierda tenemos el rate instantáneo, que muestra el número de neuronas que hacen spike frente al tiempo. El color salmón se corresponde al modelo QIF y el negro a la solución de las FREs.
En el medio a la izquierda tenemos el potencial de membrana medio de todas las neuronas. El color verde se corresponde al modelo QIF y el negro a la solución de las FREs. Se puede intercambiar por el raster plot, que es un gráfico que dibuja un punto cada vez que una neurona hace spike. El número de neuronas que se muestran en el mismo es uno de los parámetros a escoger por el usuario.
 
Abajo a la izquierda tenemos el estímulo externo en forma de función escalón. Es interesante verlo para observar cuando tienen mayor actividad las neuronas.
A la derecha tenemos las cajas y botones de control del programa. Ninguna es a tiempo real, todas necesitan que se vuelva a ejecutar el programa. De arriba abajo:
  - Seleccionador para escoger visualizar el raster plot o el potencial de membrana.
  - Número total de neuronas en la población.
  - Tiempo total de simulación.
  - Número de neuronas mostradas en el raster plot.
  - Constante de acoplamiento J aplicada en los modelos.
  - Intervalo de tiempo para la integración dt.
  - Potencial pico Vp para el modelo QIF (Vr se ha fijado como el valor negativo de Vp).
  - Media de la distribución lorentziana de las etas de las neuronas.
  - Desviación de la distribución lorentziana de las etas de las neuronas.
  - Amplitud del estímulo externo en forma de escalón.
  - Momento de inicio del estímulo externo.
  - Momento de fin del estímulo externo.
  - Cerrar: Cierra el programa.
  - Reset: Resetea el programa guardando las variables escritas en new_vars.txt.
