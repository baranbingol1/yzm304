import numpy as np
import pandas as pd

def load_data(path: str):
    """Retursn X, y from data.csv"""
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def sigmoid(Z):
    """Computes the sigmoid activation function."""
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    """Computes the tanh activation function."""
    return np.tanh(Z)

def tanh_derivative(x):
    """Derivative of tanh function."""
    return 1 - np.power(tanh(x), 2)

def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)