import numpy as np

class AsorbingMarkovChain:
    ''' 
    This class is for discrete-time absorbing Markov chains with a finite discrete state space.
    '''

    def __init__(self, no_absorbing_states, tm_matrix, N):
        self.no_absorbing_states = no_absorbing_states
        self.tm_matrix = tm_matrix
        self.N = N
        if self.tm_matrix.shape[0] != self.tm_matrix.shape[1]:
            raise ValueError("Input transition matrix must be square")
        for row in range(0, self.tm_matrix.shape[0]):
            if np.sum(self.tm_matrix[row]) < 0 or np.sum(self.tm_matrix[row]) > 1:
                raise ValueError("Input transition matrix is not stochastic")
        if not (isinstance(self.no_absorbing_states, int)):
            raise TypeError("Got non-integer argument for no_absorbing_states")
        if not (isinstance(self.N, int)):
            raise TypeError("Got non-integer argument for N")

    def transient_matrix(self):
        '''
        Calculates the matrix of transient states from the base transition matrix

        Parameters
        ----------
        tm_matrix: numpy array
            The base transition matrix for the chain

        Returns
        -------
        numpy darray
            The matrix of transient states

        Examples
        --------
        >>> tm_matrix = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
            [[0.3 0.4]
            [0.1 0.7]]
        '''
        transient_temp = np.delete(self.tm_matrix,self.tm_matrix.shape[0]-1, axis=0)
        transient_matrix = np.delete(transient_temp, transient_temp.shape[1]-1, axis=1)
        return transient_matrix
    
    def nontransient_matrix(self):
        '''
        Calculates the matrix of non-transient states from the base transition matrix

        Parameters
        ----------
        tm_matrix: numpy array
            The base transition matrix for the chain

        Returns
        -------
        numpy darray
            The matrix of non-transient states

        Examples
        --------
        >>> tm_matrix = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
            [[0.3]
            [0.2]]
        '''
        nontransient_matrix = self.tm_matrix[0:self.tm_matrix.shape[0]-self.no_absorbing_states,self.tm_matrix.shape[1]-self.no_absorbing_states]
        nontransient_matrix.shape = (self.tm_matrix.shape[0]-self.no_absorbing_states, self.no_absorbing_states)
        return nontransient_matrix

    def id_matrix(self):
        '''
        Calculates the identity matrix for use in calculating the expected times to absorb

        Parameters
        ----------
        tm_matrix: numpy array
            The base transition matrix for the chain

        no_absorbing_states: int
            The total number of absorbing states in the state space

        Returns
        -------
        numpy darray
            Identity matrix of size equal to the number of transient states squared

        Examples
        --------
        >>> tm_matrix = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
            [[1. 0.]
            [0. 1.]]
        '''
        Id = np.identity(self.tm_matrix.shape[0] - self.no_absorbing_states)
        return Id

    def ones_vector(self):
        '''
        Calculates a vector with every element equal to 1

        Parameters
        ----------
        tm_matrix: numpy array
            The base transition matrix for the chain

        no_absorbing_states: int
            The total number of absorbing states in the state space

        Returns
        -------
        numpy darray
            Vector of 1s of length equal to the number of transient states

        Examples
        --------
        >>> tm_matrix = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
            [1. 1.] 
        '''
        ones = np.ones(self.tm_matrix.shape[0]  - self.no_absorbing_states)
        return ones

    def fundamental_matrix(self):
        '''
        Calculates the fundamental matrix whose entries represent the expected number of visits to a transient state j 
        starting from state i before being asorbed

        Parameters
        ----------
        id_matrix: numpy darray
            Inherited from id_matrix function
        transient_matrix: numpy darray
            Inherited from transient_matrix function

        Returns
        -------
        numpy darray
            The fundamental matrix

        Examples
        --------
        >>> id_matrix = np.array([[1,0],[0,1]])
        >>> transient_matrix = np.array([[0.3,0.4],[0.1,0.7]])
        [[1.76470588 2.35294118]
        [0.58823529 4.11764706]]
        '''
        F = np.linalg.inv(self.id_matrix() - self.transient_matrix())
        if np.linalg.det(F) == 0:
            raise ValueError("Fundamental matrix must be invertible")
        return F
    
    def absorb_times(self):
        ex_times_absorb = np.matmul(self.fundamental_matrix(),self.ones_vector())
        return ex_times_absorb

    def forward_matrix(self):
        projection = np.linalg.matrix_power(self.tm_matrix,self.N)
        return projection