import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng is not None:
        random_val = rng.random(x.shape)
    else:
        random_val = np.random.random(x.shape)

    scale = 1 / (1 - p )

    dropout_pattern = scale * (random_val >=p)

    outputs = dropout_pattern * x
    return outputs, dropout_pattern