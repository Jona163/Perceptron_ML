# Autor: Jonathan Hernández
# Fecha: 08 Noviembre 2024
# Descripción: Código para una simulación de Perceptrón.
# GitHub: https://github.com/Jona163
#Importancion de liberias 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Función escalón unitario (unit step function)
def unit_step_func(x):
    """Función de activación: retorna 1 si x > 0, de lo contrario retorna 0."""
    return np.where(x > 0, 1, 0)
# Clase Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """Inicializa el perceptrón con tasa de aprendizaje y número de iteraciones."""
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
  def fit(self, X, y):
        """Entrena el perceptrón con los datos X e y."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Asegurar que las etiquetas sean binarias (0 o 1)
        y_ = np.where(y > 0, 1, 0)

        # Actualización de pesos y sesgo en cada iteración
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
