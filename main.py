import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Funcion Objetivo
def objective(x):
    x1, x2 = x
    return x1**2 + x2**2 - 8*x2 + 16

#Restricciones
def constraint1(x):
    x1, x2 = x
    return x2 - x1

def constraint2(x):
    x1, x2 = x
    return 2*x2 - x1**2

#Lagrange
def lagrangian(x, l1, l2):
    return objective(x) + l1 * constraint1(x) + l2 * constraint2(x)

#Deriva de Lagrange
def lagrangian_gradient(x, l1, l2):
    x1, x2 = x
    grad = np.zeros(4)
    grad[0] = 2 * x1 + l1 - l2  
    grad[1] = 2 * x2 - 8 + l1 * 1 + l2 * 2  
    grad[2] = x2 - x1  
    grad[3] = 2 * x2 - x1
    return grad


x0 = np.array([0.0, 0.0])
l0 = np.array([0.0, 0.0])
bounds = ((None, None), (None, None))

res = minimize(lambda x: lagrangian(x, l0[0], l0[1]), x0, bounds=bounds, 
               constraints=({'type': 'ineq', 'fun': constraint1},
                            {'type': 'ineq', 'fun': constraint2}))

print("Resultados de la Optimización:")
print(res)

# Grafica
x1 = np.linspace(-20, 20, 400)
x2 = np.linspace(-20, 20, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + X2**2 - 8*X2 + 16

plt.figure()
plt.contour(X1, X2, Z, levels=np.arange(0, 200, 10))
plt.plot(x1, x1, label='x2 = x1', color='red', linestyle='--')
plt.plot(x1, x1/2, label='2x2 = x1', color='green', linestyle='--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Función de costo y restricciones')
plt.legend()
plt.scatter(res.x[0], res.x[1], color='red', marker='o', label='Punto óptimo')
plt.legend()
plt.grid(True)
plt.show()
