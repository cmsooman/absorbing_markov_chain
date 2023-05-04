import numpy as np

def random_stochastic_matrix(n, absorb_rows):
    """
    Generates a stochastic matrix of random transition probabilities and adds rows representing absorbing states

    Parameters
    ----------
    n: number of rows/columns in the matrix
    absorb_rows: rows which represent absorbing states

    Returns
    -------
    numpy ndarray
    
    """
    mat = np.random.random((n,n))
    rsum = None
    while (np.any(rsum != 1)):
        mat = mat / mat.sum(1)[:, np.newaxis]
        rsum = mat.sum(1)
        mat_temp = np.delete(mat, mat.shape[0]-1,0)
        mat = np.concatenate([mat_temp, absorb_rows], axis=0)
    return mat

a = random_stochastic_matrix(4, np.array([[0,0,0,1]]))

print(a)