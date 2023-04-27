import numpy as np

class AbsorbingMarkovChain:
    """
    This class is for discrete-time absorbing Markov chains with a finite discrete state space.

    ...

    Attributes
    ----------
    no_absorbing_states : int
        The number of absorbing states in the chain
    P_0 : numpy darray
        The base transition matrix at time 0
    N : int
        The number of time steps to project the chain
    
    Methods
    -------    
    transient_matrix()
        Calculates the matrix of transient states from the base transition matrix

    nontransient_matrix()
        Calculates the matrix of non-transient states from the base transition matrix

    id_matrix()
        Calculates the identity matrix for use in calculating the expected times to absorb

    ones_vector()
        Calculates a vector with every element equal to 1

    fundamental_matrix()
        Calculates the fundamental matrix whose entries represent the expected number of visits to a transient state j 
        starting from state i before being asorbed

    absorb_times()
        Calculates the expected number of steps before being absorbed in any absorbing state when
        starting in a transient state i
    
    forward_matrix()
        Calculates powers of the base transition matrix P_0 for N = 1, 2, 3,.....
  
    """
    def __init__(self, no_absorbing_states, P_0, N):
        self.no_absorbing_states = no_absorbing_states
        self.P_0 = P_0
        self.N = N
        if self.P_0.shape[0] != self.P_0.shape[1]:
            raise ValueError("Input transition matrix must be square")
        for row in range(0, self.P_0.shape[0]):
            if np.sum(self.P_0[row]) < 0 or np.sum(self.P_0[row]) > 1:
                raise ValueError("Input transition matrix is not stochastic")
        if not (isinstance(self.no_absorbing_states, int)):
            raise TypeError("Got non-integer argument for no_absorbing_states")
        if not (isinstance(self.N, int)):
            raise TypeError("Got non-integer argument for N")

    def transient_matrix(self):
        """
        Calculates the matrix of transient states from the base transition matrix

        Parameters
        ----------
        P_0: numpy array
            The base transition matrix for the chain

        Returns
        -------
        numpy darray
            The matrix of transient states

        Examples
        --------
        >>> P_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
            [[0.3 0.4]
            [0.1 0.7]]
        """
        Q_temp = np.delete(self.P_0,self.P_0.shape[0]-1, axis=0)
        Q = np.delete(Q_temp, Q_temp.shape[1]-1, axis=1)
        return Q
    
    def nontransient_matrix(self):
        """
        Calculates the matrix of non-transient states from the base transition matrix

        Parameters
        ----------
        P_0: numpy array
            The base transition matrix for the chain

        Returns
        -------
        numpy darray
            The matrix of non-transient states

        Examples
        --------
        >>> P_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
            [[0.3]
            [0.2]]
        """
        R = self.P_0[0:self.P_0.shape[0]-self.no_absorbing_states,self.P_0.shape[1]-self.no_absorbing_states]
        R.shape = (self.P_0.shape[0]-self.no_absorbing_states, self.no_absorbing_states)
        return R

    def id_matrix(self):
        """
        Calculates the identity matrix for use in calculating the expected times to absorb

        Parameters
        ----------
        P_0: numpy array
            The base transition matrix for the chain

        no_absorbing_states: int
            The total number of absorbing states in the state space

        Returns
        -------
        numpy darray
            Identity matrix of size equal to the number of transient states squared

        Examples
        --------
        >>> P_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
            [[1. 0.]
            [0. 1.]]
        """
        Id = np.identity(self.P_0.shape[0] - self.no_absorbing_states)
        return Id

    def ones_vector(self):
        """
        Calculates a vector with every element equal to 1

        Parameters
        ----------
        P_0: numpy array
            The base transition matrix for the chain

        no_absorbing_states: int
            The total number of absorbing states in the state space

        Returns
        -------
        numpy darray
            Vector of 1s of length equal to the number of transient states

        Examples
        --------
        >>> P_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
            [1. 1.] 
        """
        ones = np.ones(self.P_0.shape[0]  - self.no_absorbing_states)
        return ones

    def fundamental_matrix(self):
        """
        Calculates the fundamental matrix whose entries represent the expected number of visits to a transient state j 
        starting from state i before being asorbed

        Parameters
        ----------
        id_matrix(): numpy darray
            Inherited from id_matrix function
        transient_matrix(): numpy darray
            Inherited from transient_matrix function

        Returns
        -------
        numpy darray
            The fundamental matrix

        Examples
        --------
        >>> id_matrix() = np.array([[1,0],[0,1]])
        >>> transient_matrix() = np.array([[0.3,0.4],[0.1,0.7]])
        [[1.76470588 2.35294118]
        [0.58823529 4.11764706]]
        """
        F = np.linalg.inv(self.id_matrix() - self.transient_matrix())
        if np.linalg.det(F) == 0:
            raise ValueError("Fundamental matrix must be invertible")
        return F
    
    def absorb_times(self):
        """
        Calculates the expected number of steps before being absorbed in any absorbing state when
        starting in a transient state i

        Parameters
        ----------
        fundamental_matrix(): numpy darray
            Inherited from fundamental_matrix function
        ones(): numpy darray
            Inherited from ones function

        Returns
        -------
        numpy darray
            Vector of expected number of steps for eac transient state before being absorbed

        Examples
        --------
        >>> fundamental_matrix() = np.array([[1.76470588, 2.35294118], [0.58823529, 4.11764706]])
        >>> ones() = np.array([1,1])
        [4.11764706 4.70588235]
        """
        ex_times_absorb = np.matmul(self.fundamental_matrix(),self.ones_vector())
        return ex_times_absorb

    def forward_matrix(self):
        """
        Calculates powers of the base transition matrix P_0 for N = 1, 2, 3,.....

        Parameters
        ----------
        P_0: numpy darray
            The base transition matrix
        N: int
            Number of time steps for the projection

        Returns
        -------
        numpy darray
            Matrix containing the values of P^0 to the power N

        Examples
        --------
        >>> P_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> N = 5
        [[0.04347 0.20756 0.74897]
        [0.05189 0.25103 0.69708]
        [0.      0.      1.     ]]
        """
        P_N = np.linalg.matrix_power(self.P_0,self.N)
        return P_N
