import numpy as np
from numpy.linalg import inv


class LinearRegression:
    def __init__(self):
        self._coefficients = None

    def fit(self, input_values, output_values):
        x = np.array([[1] + input_value for input_value in input_values])
        self._coefficients = np.dot(np.dot(inv(np.dot(x.T, x)), x.T), output_values)

    def predict(self, input_values):
        x = np.array([[1] + input_value for input_value in input_values])
        return np.dot(x, self._coefficients).tolist()
