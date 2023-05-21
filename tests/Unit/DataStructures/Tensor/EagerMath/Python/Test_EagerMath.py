# Distributed under the MIT License.
# See LICENSE.txt for details.

import itertools
import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Jacobian, Scalar, tnsr
from spectre.DataStructures.Tensor.EagerMath import determinant, magnitude


def to_numpy(tensor):
    # Construct a Numpy array where the first 'rank' dimensions are the indices
    # and the last dimension is the number of grid points. This is an
    # inefficient way to store the data because it ignores symmetries, which is
    # why it currently isn't a public function.
    num_points = len(tensor[0])
    result = np.zeros(tensor.rank * (tensor.dim,) + (num_points,))
    for indices in itertools.product(*(tensor.rank * (range(tensor.dim),))):
        result[indices] = tensor.get(*indices)
    return result


class TestEagerMath(unittest.TestCase):
    def test_determinant(self):
        data = np.random.rand(9, 4)
        jacobian = Jacobian[DataVector, 3](data)
        det = determinant(jacobian)
        det_numpy = np.linalg.det(np.moveaxis(to_numpy(jacobian), -1, 0))
        npt.assert_allclose(np.array(det)[0], det_numpy)

    def test_magnitude(self):
        # Euclidean
        data = np.random.rand(3, 4)
        vector = tnsr.I[DataVector, 3](data)
        mag = magnitude(vector)
        mag_numpy = np.linalg.norm(data, axis=0)
        npt.assert_allclose(np.array(mag)[0], mag_numpy)
        # Non-Euclidean
        data_metric = np.random.rand(6, 4)
        metric = tnsr.ii[DataVector, 3](data_metric)
        mag = magnitude(vector, metric)
        mag_numpy = np.sqrt(
            np.einsum("i...,j...,ij...", data, data, to_numpy(metric))
        )
        npt.assert_allclose(np.array(mag)[0], mag_numpy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
