import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size, T, input_dim = X.shape
    h_list = [] #All hiiden states
    h_current = h_0 #Curremt hidden state
    # X - (batch, T, input_dim)

    for t in range(T):
        x_t = X[: ,t ,:]
        h_current = np.tanh(x_t@W_xh.T + h_current@W_hh.T + b_h)
        h_list.append(h_current)

    hidden_states = np.stack(h_list, axis=1)
    return (hidden_states, h_current)