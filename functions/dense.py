import numpy as np

def dense(input_vector:np.ndarray, weights:np.ndarray, bias:np.ndarray):
    return np.dot(input_vector, weights) + bias
