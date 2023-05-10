import unittest
from absorbing_markov_chain import AbsorbingMarkovChain
from generate_test_matrices import random_stochastic_matrix
import numpy as np

mat1 = random_stochastic_matrix(4, 1, np.array([[0,0,0,1]]))
mat2 = random_stochastic_matrix(5, 2, np.array([[0,0,0,1,0], [0,0,0,0,1]]))
mat3 = random_stochastic_matrix(5, 1, np.array([[0,0,0,0,1]]))
mat4 = random_stochastic_matrix(4, 2, np.array([[0,0,1,0], [0,0,0,1]]))
mat5 = random_stochastic_matrix(7, 4, np.array([[0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]))

class AbsorbingMarkovChainTest(unittest.TestCase):
    
    def test_not_square(self):
        with self.assertRaises(ValueError):
            test1 = AbsorbingMarkovChain(1, np.array([[0.35, 0.47],[0.45, 0.34], [0.15, 0.80]]), 1)

    #def test_invertible(self):
    #    test1 = AbsorbingMarkovChain(1, np.array([[0,0,1],[0,1,0],[0,0,1]]), 1)
    #    with self.assertRaises(ValueError):
    #        test1.fundamental_matrix()    

    def test_row_sum_lt_zero(self):
        with self.assertRaises(ValueError):
            test1 = AbsorbingMarkovChain(1, np.array([[0.5, 0.5], [0.6, -0.65]]), 1)

    def test_row_sum_gt_zero(self):
        with self.assertRaises(ValueError):
            test1 = AbsorbingMarkovChain(1,np.array([[0.5, 0.55], [0.6, 0.35]]), 1)

    def test_absorbing_property(self):
        test1 = AbsorbingMarkovChain(1,np.array([[0.3, 0.4, 0.3],[0.1, 0.7, 0.2],[0, 0, 1]]),100000)
        np.testing.assert_array_almost_equal(test1.forward_matrix(), np.array([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]]))

    def test_mat1_identity(self):
        test1 = AbsorbingMarkovChain(1,mat1, 1)
        np.testing.assert_array_equal(test1.id_matrix(), np.array([[1,0,0], [0,1,0], [0,0,1]]))

    def test_mat2_identity(self):
        test1 = AbsorbingMarkovChain(2,mat2, 1)
        np.testing.assert_array_equal(test1.id_matrix(), np.array([[1,0,0], [0,1,0], [0,0,1]]))

    def test_mat3_identity(self):
        test1 = AbsorbingMarkovChain(1,mat3, 1)
        np.testing.assert_array_equal(test1.id_matrix(), np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]))

    def test_mat4_identity(self):
        test1 = AbsorbingMarkovChain(2,mat4, 1)
        np.testing.assert_array_equal(test1.id_matrix(), np.array([[1,0], [0,1]]))

    def test_mat5_identity(self):
        test1 = AbsorbingMarkovChain(4,mat5, 1)
        np.testing.assert_array_equal(test1.id_matrix(), np.array([[1,0,0], [0,1,0], [0,0,1]]))


if __name__ == '__main__':
    unittest.main()
