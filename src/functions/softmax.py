import numpy as np
def softmax(x:np.ndarray):
    """Compute the softmax of each row of the input x.
    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: An array where each row is the softmax of the corresponding row of x.
    """
    z = np.exp(x)
    output = z / (np.sum(z, axis=1, keepdims=True) + 1e-15)
    return output
