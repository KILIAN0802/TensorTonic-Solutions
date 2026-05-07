import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    
    dims = tuple(i for i in range(x.ndim) if i !=1) if x.ndim > 2 else 0
    mean = np.mean(x, axis = dims, keepdims=True)
    var = np.var(x, axis=dims, keepdims=True)

    x = (x - mean )/ np.sqrt(var+eps)

    gamma_reshaped = gamma.reshape(mean.shape)
    beta_reshaped = beta.reshape(mean.shape)
    y = gamma_reshaped*x+beta_reshaped
    return y