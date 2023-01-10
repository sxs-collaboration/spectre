# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.NumericalAlgorithms.LinearOperators import definite_integral

import unittest
from spectre.DataStructures import DataVector
from spectre.Spectral import Mesh, Basis, Quadrature, collocation_points


# Integral [-1, 1] is 2
def polynomial(x):
    return 3. * x**2 + 2. * x


class TestDefiniteIntegral(unittest.TestCase):
    def test_definite_integral(self):
        mesh = Mesh[1](4, Basis.Legendre, Quadrature.GaussLobatto)
        x = collocation_points(mesh)
        integrand = polynomial(x)
        integral = definite_integral(integrand, mesh)
        self.assertAlmostEqual(integral, 2.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
