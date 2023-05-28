import numpy as np
#import time

def random_stochastic_matrix(n, m, absorb_rows):
    """
    Generates a stochastic matrix of random transition probabilities and adds rows representing absorbing states

    Parameters
    ----------
    n: number of rows/columns in the matrix
    m: number of absorbing states
    absorb_rows: rows which represent absorbing states

    Returns
    -------
    numpy ndarray
    
    """
    rsum = None
    while (np.any(rsum != 1)):
        mat = np.random.random((n,n))
        mat = mat / mat.sum(1)[:, np.newaxis]
        rsum = mat.sum(1)

    mat_temp = mat[0:(n-m)]
    mat = np.concatenate([mat_temp, absorb_rows], axis=0)
    return mat
