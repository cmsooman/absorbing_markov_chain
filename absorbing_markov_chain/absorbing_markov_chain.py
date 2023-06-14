"""This modules provides the class AbsorbingMarkovChain"""

import numpy as np


class AbsorbingMarkovChain:
    """
    This class is for discrete-time absorbing Markov chains with a finite discrete state space.

    ...

    Attributes
    ----------
    no_absorbing_states : int
        The number of absorbing states in the chain
    p_0 : numpy ndarray
        The base transition matrix at time 0

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
        Calculates the fundamental matrix whose entries represent the expected number of visits to
        a transient state j starting from state i before being absorbed

    fundamental_matrix_var()
        Calculates the variance of the expected number of visits to a transient state j starting
        from transient state i before being absorbed

    absorb_times()
        Calculates the expected number of steps before being absorbed in any absorbing state when
        starting in a transient state i

    absorb_times_var(self)
        Calculates the variance on the number of steps before being absorbed when starting
        in transient state i

    absorb_probs()
        Calculates to probability of being absorbed when starting in transient state i

    forward_matrix()
        Calculates powers of the base transition matrix p_0 for n = 1, 2, 3,.....

    """

    def __init__(
        self, no_absorbing_states: int, p_0: np.ndarray  # , no_years_project: int
    ) -> None:
        self.no_absorbing_states = no_absorbing_states
        self.p_0 = p_0
        if self.p_0.shape[0] != self.p_0.shape[1]:
            raise ValueError("Input transition matrix must be square")
        for row in range(0, self.p_0.shape[0]):
            if np.sum(self.p_0[row]) < 0 or np.sum(self.p_0[row]) > 1:
                raise ValueError("Input transition matrix is not stochastic")
        if not isinstance(self.no_absorbing_states, int):
            raise TypeError("Got non-integer argument for no_absorbing_states")

    def transient_matrix(self) -> np.ndarray:
        """
        Calculates the matrix of transient states from the base transition matrix

        Returns
        -------
        numpy ndarray
            The matrix of transient states

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
            [[0.3 0.4]
            [0.1 0.7]]
        """
        q_matrix = self.p_0[
            0 : self.p_0.shape[0] - self.no_absorbing_states,
            0 : self.p_0.shape[1] - self.no_absorbing_states,
        ]
        return q_matrix

    def nontransient_matrix(self) -> np.ndarray:
        """
        Calculates the matrix of non-transient states from the base transition matrix

        Returns
        -------
        numpy ndarray
            The matrix of non-transient states

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
            [[0.3]
            [0.2]]
        """
        r_matrix = self.p_0[
            0 : (self.p_0.shape[0] - self.no_absorbing_states),
            (self.p_0.shape[1] - self.no_absorbing_states) : self.p_0.shape[1],
        ]
        return r_matrix

    def id_matrix(self) -> np.ndarray:
        """
        Calculates the identity matrix for use in calculating the expected times to absorb

        Returns
        -------
        numpy ndarray
            Identity matrix of size equal to the number of transient states squared

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
            [[1. 0.]
            [0. 1.]]
        """
        id_matrix = np.identity(self.p_0.shape[0] - self.no_absorbing_states)
        return id_matrix

    def ones_vector(self) -> np.ndarray:
        """
        Calculates a vector with every element equal to 1

        Returns
        -------
        numpy ndarray
            Vector of 1s of length equal to the number of transient states

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
            [1. 1.]
        """
        ones = np.ones(self.p_0.shape[0] - self.no_absorbing_states)
        return ones

    def fundamental_matrix(self) -> np.ndarray:
        """
        Calculates the fundamental matrix whose entries represent the expected number of visits to
        a transient state j starting from state i before being absorbed

        Returns
        -------
        numpy ndarray
            The fundamental matrix

        Examples
        --------
        >>> id_matrix() = np.array([[1,0],[0,1]])
        >>> transient_matrix() = np.array([[0.3,0.4],[0.1,0.7]])
        [[1.76470588 2.35294118]
        [0.58823529 4.11764706]]
        """
        f_matrix = np.linalg.inv(self.id_matrix() - self.transient_matrix())
        if np.linalg.det(f_matrix) == 0:
            raise ValueError("Fundamental matrix must be invertible")
        return f_matrix

    def fundamental_matrix_var(self) -> np.ndarray:
        """
        Calculates the variance of the fundamental matrix

        Returns
        -------
        numpy ndarray
            The variances of the fundamental matrix

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
        [[1.34948097, 11.48788927],
        [1.14186851, 12.83737024]]
        """
        f_var = np.matmul(
            self.fundamental_matrix(),
            (2 * (np.diag(np.diag(self.fundamental_matrix()))) - self.id_matrix()),
        ) - np.multiply(self.fundamental_matrix(), self.fundamental_matrix())
        return f_var

    def absorb_times(self) -> np.ndarray:
        """
        Calculates the expected number of steps before being absorbed in any absorbing state when
        starting in a transient state i

        Returns
        -------
        numpy ndarray
            Vector of expected number of steps for each transient state before being absorbed

        Examples
        --------
        >>> fundamental_matrix() = np.array([[1.76470588, 2.35294118], [0.58823529, 4.11764706]])
        >>> ones() = np.array([1,1])
        [4.11764706 4.70588235]
        """
        ex_times_absorb = np.matmul(self.fundamental_matrix(), self.ones_vector())
        return ex_times_absorb

    def absorb_times_var(self) -> np.ndarray:
        """
        Calculates the variance on the number of steps before being absorbed when starting
        in transient state i

        Returns
        -------
        numpy ndarray
            Matrix containing the variances for each expected time to absorb

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
        [15.60553633, 16.74740484]

        """
        ex_times_absorb_var = np.matmul(
            (2 * self.fundamental_matrix() - self.id_matrix()),
            self.absorb_times(),
        ) - np.multiply(self.absorb_times(), self.absorb_times())
        return ex_times_absorb_var

    def absorb_probs(self) -> np.ndarray:
        """
        Calculates the probability of being absorbed when starting in transient state i

        Returns
        -------
        numpy ndarray

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_absorbing_states = 1
        [1.],
        [1.]]

        """
        prob_of_absorb = np.matmul(
            self.fundamental_matrix(), self.nontransient_matrix()
        )
        return prob_of_absorb

    def forward_matrix(self, no_years_project) -> np.ndarray:
        """
        Calculates powers of the base transition matrix p_0 for n = 1, 2, 3,.....

        Parameters
        ----------
        no_years_project: int
            Number of time steps for the projection

        Returns
        -------
        numpy ndarray
            Matrix containing the values of P^0 to the power n

        Examples
        --------
        >>> p_0 = np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]])
        >>> no_years_project = 5
        [[0.04347 0.20756 0.74897]
        [0.05189 0.25103 0.69708]
        [0.      0.      1.     ]]
        """
        p_n = np.linalg.matrix_power(self.p_0, no_years_project)
        if not isinstance(no_years_project, int):
            raise TypeError("Got non-integer argument for no_years_project")
        return p_n
