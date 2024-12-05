import numpy as np
def relu_derivative(x):
        return np.where(x > 0, 1, 0)
