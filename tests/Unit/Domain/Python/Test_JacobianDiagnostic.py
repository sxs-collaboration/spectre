# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, Jacobian, tnsr
from spectre.Domain import jacobian_diagnostic
from spectre.Spectral import Basis, Mesh, Quadrature, collocation_points


def affine_map(x):
    return 2.0 * x


class TestJacobianDiagnostic(unittest.TestCase):
    def test_jacobian_diagnostic(self):
        mesh = Mesh[1](4, Basis.Legendre, Quadrature.GaussLobatto)
        x = collocation_points(mesh)
        mapped_coordinates = tnsr.I[DataVector, 1, Frame.Grid]([affine_map(x)])
        jac = Jacobian[DataVector, 1, Frame.Grid](num_points=4, fill=2.0)

        jac_diag = jacobian_diagnostic(jac, mapped_coordinates, mesh)
        expected_jac_diag = tnsr.I[DataVector, 1, Frame.ElementLogical](
            num_points=4, fill=0.0
        )
        npt.assert_allclose(jac_diag, expected_jac_diag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
