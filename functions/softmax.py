import numpy as np

def softmax(x:np.ndarray):
    # Using x, creates a vector of probs
    z = np.exp(x)
    output = z / (np.sum(z, axis=1, keepdims=True) + 1e-15)
    return output
