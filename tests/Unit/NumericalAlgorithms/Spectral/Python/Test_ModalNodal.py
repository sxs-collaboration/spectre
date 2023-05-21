# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
from numpy.polynomial import chebyshev, legendre

from spectre.DataStructures import DataVector
from spectre.Spectral import (
    Basis,
    Mesh1D,
    Quadrature,
    collocation_points,
    modal_to_nodal_matrix,
    nodal_to_modal_matrix,
)


class TestModalNodal(unittest.TestCase):
    def test_modal_to_nodal_matrix_legendre(self):
        for quadrature in [Quadrature.Gauss, Quadrature.GaussLobatto]:
            for num_coll_points in range(2, 13):
                mesh = Mesh1D(num_coll_points, Basis.Legendre, quadrature)
                coll_points = collocation_points(mesh)
                mtn_numpy = legendre.legvander(coll_points, num_coll_points - 1)
                mtn_spectre = modal_to_nodal_matrix(mesh)
                np.testing.assert_allclose(mtn_spectre, mtn_numpy, 1e-12, 1e-12)
                ntm_spectre = nodal_to_modal_matrix(mesh)
                np.testing.assert_allclose(
                    np.matmul(mtn_spectre, ntm_spectre),
                    np.identity(num_coll_points),
                    1e-12,
                    1e-12,
                )

    def test_modal_to_nodal_matrix_chebyshev(self):
        for quadrature in [Quadrature.Gauss, Quadrature.GaussLobatto]:
            for num_coll_points in range(2, 13):
                mesh = Mesh1D(num_coll_points, Basis.Chebyshev, quadrature)
                coll_points = collocation_points(mesh)
                mtn_numpy = chebyshev.chebvander(
                    coll_points, num_coll_points - 1
                )
                mtn_spectre = modal_to_nodal_matrix(mesh)
                np.testing.assert_allclose(mtn_spectre, mtn_numpy, 1e-12, 1e-12)
                ntm_spectre = nodal_to_modal_matrix(mesh)
                np.testing.assert_allclose(
                    np.matmul(mtn_spectre, ntm_spectre),
                    np.identity(num_coll_points),
                    1e-12,
                    1e-12,
                )


if __name__ == "__main__":
    unittest.main()
