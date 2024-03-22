# OptimizadorKKT

## Problema 
Este proyecto implementa un optimizador basado en el método de Karush-Kuhn-Tucker (KKT) para resolver problemas de optimización con restricciones. Utiliza las bibliotecas numpy, scipy.optimize, y matplotlib.pyplot para la optimización numérica y la visualización de los resultados.

### Maximizar:
f(x,y)=x^2+y^2-8y+16

### Sujeto a:
y=x
2y=x

## Requisitos
Python 3.7 o superior
Numpy
Scipy
Matplotlib
## Instalación
Para instalar las dependencias necesarias, ejecute el siguiente comando:

pip install virtualenv
python -m venv venv
./venv/bin/activate
pip install numpy scipy matplotlib

## Uso
El script principal utiliza una función objetivo y un conjunto de restricciones para optimizar un problema de programación lineal
Este script define una función objetivo y un conjunto de restricciones, luego utiliza el optimizador para encontrar la solución óptima. Finalmente, visualiza la función objetivo y las restricciones junto con la solución óptima encontrada.
