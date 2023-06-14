import unittest
from absorbing_markov_chain.absorbing_markov_chain import AbsorbingMarkovChain
from generate_test_matrices import random_stochastic_matrix
import numpy as np

mat1 = random_stochastic_matrix(4, 1, np.array([[0, 0, 0, 1]]))
mat2 = random_stochastic_matrix(5, 2, np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))
mat3 = random_stochastic_matrix(5, 1, np.array([[0, 0, 0, 0, 1]]))
mat4 = random_stochastic_matrix(4, 2, np.array([[0, 0, 1, 0], [0, 0, 0, 1]]))
mat5 = random_stochastic_matrix(
    7,
    4,
    np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    ),
)


class AbsorbingMarkovChainTest(unittest.TestCase):
    def test_not_square(self):
        with self.assertRaises(ValueError):
            test1 = AbsorbingMarkovChain(
                1, np.array([[0.35, 0.47], [0.45, 0.34], [0.15, 0.80]])
            )

    # def test_invertible(self):
    #    test1 = AbsorbingMarkovChain(1, np.array([[0,0,1],[0,1,0],[0,0,1]]), 1)
    #    with self.assertRaises(ValueError):
    #        test1.fundamental_matrix()

    def test_row_sum_lt_zero(self):
        with self.assertRaises(ValueError):
            test1 = AbsorbingMarkovChain(1, np.array([[0.5, 0.5], [0.6, -0.65]]))

    def test_row_sum_gt_zero(self):
        with self.assertRaises(ValueError):
            test1 = AbsorbingMarkovChain(1, np.array([[0.5, 0.55], [0.6, 0.35]]))

    def test_absorbing_property(self):
        test1 = AbsorbingMarkovChain(
            1, np.array([[0.3, 0.4, 0.3], [0.1, 0.7, 0.2], [0, 0, 1]])
        )
        np.testing.assert_array_almost_equal(
            test1.forward_matrix(100000),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        )

    def test_mat1_identity(self):
        test1 = AbsorbingMarkovChain(1, mat1)
        np.testing.assert_array_equal(
            test1.id_matrix(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )

    def test_mat2_identity(self):
        test1 = AbsorbingMarkovChain(2, mat2)
        np.testing.assert_array_equal(
            test1.id_matrix(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )

    def test_mat3_identity(self):
        test1 = AbsorbingMarkovChain(1, mat3)
        np.testing.assert_array_equal(
            test1.id_matrix(),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        )

    def test_mat4_identity(self):
        test1 = AbsorbingMarkovChain(2, mat4)
        np.testing.assert_array_equal(test1.id_matrix(), np.array([[1, 0], [0, 1]]))

    def test_mat5_identity(self):
        test1 = AbsorbingMarkovChain(4, mat5)
        np.testing.assert_array_equal(
            test1.id_matrix(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )

    def test_mat1_ones(self):
        test1 = AbsorbingMarkovChain(1, mat1)
        np.testing.assert_array_equal(test1.ones_vector(), np.array([1, 1, 1]))

    def test_mat2_ones(self):
        test1 = AbsorbingMarkovChain(2, mat2)
        np.testing.assert_array_equal(test1.ones_vector(), np.array([1, 1, 1]))

    def test_mat3_ones(self):
        test1 = AbsorbingMarkovChain(1, mat3)
        np.testing.assert_array_equal(test1.ones_vector(), np.array([1, 1, 1, 1]))

    def test_mat4_ones(self):
        test1 = AbsorbingMarkovChain(2, mat4)
        np.testing.assert_array_equal(test1.ones_vector(), np.array([1, 1]))

    def test_mat5_ones(self):
        test1 = AbsorbingMarkovChain(4, mat5)
        np.testing.assert_array_equal(test1.ones_vector(), np.array([1, 1, 1]))

    def test_mat1_proj_sum(self):
        test1 = AbsorbingMarkovChain(1, mat1)
        test1_row_sums = test1.forward_matrix(5).sum(1)
        np.testing.assert_array_almost_equal(test1_row_sums, np.array([1, 1, 1, 1]))

    def test_mat2_proj_sum(self):
        test1 = AbsorbingMarkovChain(2, mat2)
        test1_row_sums = test1.forward_matrix(31).sum(1)
        np.testing.assert_array_almost_equal(test1_row_sums, np.array([1, 1, 1, 1, 1]))

    def test_mat3_proj_sum(self):
        test1 = AbsorbingMarkovChain(1, mat3)
        test1_row_sums = test1.forward_matrix(100).sum(1)
        np.testing.assert_array_almost_equal(test1_row_sums, np.array([1, 1, 1, 1, 1]))

    def test_mat4_proj_sum(self):
        test1 = AbsorbingMarkovChain(2, mat4)
        test1_row_sums = test1.forward_matrix(173).sum(1)
        np.testing.assert_array_almost_equal(test1_row_sums, np.array([1, 1, 1, 1]))

    def test_mat5_proj_sum(self):
        test1 = AbsorbingMarkovChain(4, mat5)
        test1_row_sums = test1.forward_matrix(250).sum(1)
        np.testing.assert_array_almost_equal(
            test1_row_sums, np.array([1, 1, 1, 1, 1, 1, 1])
        )


if __name__ == "__main__":
    unittest.main()
