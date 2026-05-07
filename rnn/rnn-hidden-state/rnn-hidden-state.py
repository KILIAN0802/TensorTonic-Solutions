import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initialize the hidden state for an RNN.
    """
    hidden_state = np.zeros((batch_size, hidden_dim))
    return hidden_state