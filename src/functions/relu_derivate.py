import numpy as np


def relu_derivate(x):
    return np.where(x > 0, 1, 0)
