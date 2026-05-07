import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    output = []
    gradient = 1.0
    W_hh = np.linalg.norm(W_hh, ord=2)
    for t in range(T):
        output.append(float(gradient))
        gradient *= W_hh
    return output