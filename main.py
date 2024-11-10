# Autor: Jonathan Hernández
# Fecha: 10 noviembre 2024
# Descripción: Código para una simulación de Perceptrón.
# GitHub: https://github.com/Jona163

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Función escalón unitario (unit step function)
def unit_step_func(x):
    """Función de activación: retorna 1 si x > 0, de lo contrario retorna 0."""
    return np.where(x > 0, 1, 0)
