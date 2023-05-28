# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import InverseJacobian, Scalar
from spectre.NumericalAlgorithms.LinearOperators import partial_derivative
from spectre.Spectral import Basis, Mesh, Quadrature, collocation_points


def polynomial(x):
    return 2.0 * x**2 + 3.0 * x + 4.0


def deriv_polynomial(x):
    return 4.0 * x + 3.0


class TestPartialDerivatives(unittest.TestCase):
    def test_partial_derivative(self):
        mesh = Mesh[1](4, Basis.Legendre, Quadrature.GaussLobatto)
        x = collocation_points(mesh)
        inv_jacobian = InverseJacobian[DataVector, 1](num_points=4, fill=1.0)
        scalar = Scalar[DataVector]([polynomial(x)])
        deriv = partial_derivative(scalar, mesh, inv_jacobian)
        npt.assert_allclose(np.array(deriv), [deriv_polynomial(x)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
