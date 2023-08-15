# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt
from numpy.polynomial import chebyshev, legendre

from spectre.DataStructures import DataVector
from spectre.Spectral import (
    Basis,
    Mesh1D,
    Quadrature,
    collocation_points,
    differentiation_matrix,
    exponential_filter,
    interpolation_matrix,
    logical_coordinates,
    modal_to_nodal_matrix,
    nodal_to_modal_matrix,
    quadrature_weights,
    zero_lowest_modes,
)


class TestSpectral(unittest.TestCase):
    def test_legendre_gauss(self):
        for n in range(2, 13):
            xi, w = legendre.leggauss(n)
            mesh = Mesh1D(n, Basis.Legendre, Quadrature.Gauss)
            npt.assert_allclose(collocation_points(mesh), xi, 1e-12, 1e-12)
            npt.assert_allclose(quadrature_weights(mesh), w, 1e-12, 1e-12)

    def test_chebyshev_gauss(self):
        for num_coll_points in range(2, 13):
            coefs = np.zeros(num_coll_points + 1)
            coefs[-1] = 1.0
            coll_points_numpy = chebyshev.chebroots(coefs)
            mesh = Mesh1D(num_coll_points, Basis.Chebyshev, Quadrature.Gauss)
            coll_points_spectre = collocation_points(mesh)
            np.testing.assert_allclose(
                coll_points_numpy, coll_points_spectre, 1e-12, 1e-12
            )

    def test_legendre_gauss_lobatto(self):
        for num_coll_points in range(2, 13):
            coefs = np.zeros(num_coll_points)
            coefs[-1] = 1.0
            coll_points_numpy = np.concatenate(
                ([-1.0], legendre.Legendre(coefs).deriv().roots(), [1.0])
            )
            mesh = Mesh1D(
                num_coll_points, Basis.Legendre, Quadrature.GaussLobatto
            )
            coll_points_spectre = collocation_points(mesh)
            np.testing.assert_allclose(
                coll_points_numpy, coll_points_spectre, 1e-12, 1e-12
            )

    def test_chebyshev_gauss_lobatto(self):
        for num_coll_points in range(2, 13):
            coefs = np.zeros(num_coll_points)
            coefs[-1] = 1.0
            coll_points_numpy = np.concatenate(
                ([-1.0], chebyshev.Chebyshev(coefs).deriv().roots(), [1.0])
            )
            mesh = Mesh1D(
                num_coll_points, Basis.Chebyshev, Quadrature.GaussLobatto
            )
            coll_points_spectre = collocation_points(mesh)
            np.testing.assert_allclose(
                coll_points_numpy, coll_points_spectre, 1e-12, 1e-12
            )

    def test_differentiation_matrix(self):
        mesh = Mesh1D(3, Basis.Legendre, Quadrature.GaussLobatto)
        xi = np.asarray(logical_coordinates(mesh)[0])
        # The derivative of a quadratic polynomial should be exact
        u = 3.0 * xi**2 + 2.0 * xi + 1.0
        D = np.asarray(differentiation_matrix(mesh))
        du = D @ u
        npt.assert_allclose(du, 6.0 * xi + 2.0, 1e-12, 1e-12)

    def test_interpolation_matric(self):
        mesh = Mesh1D(3, Basis.Legendre, Quadrature.GaussLobatto)
        xi = np.asarray(logical_coordinates(mesh)[0])
        # Interpolation to midpoint should be exact because it's a collocation
        # point
        u = 3.0 * xi**2 + 2.0 * xi + 1.0
        I = np.asarray(interpolation_matrix(mesh, target_points=[0.0]))
        npt.assert_allclose(I @ u, 1.0, 1e-12, 1e-12)

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

    def test_exponential_filter(self):
        mesh = Mesh1D(4, Basis.Legendre, Quadrature.GaussLobatto)
        filter_matrix = exponential_filter(mesh, alpha=10.0, half_power=2)
        self.assertEqual(filter_matrix.shape, (4, 4))

    def test_zero_lowest_modes(self):
        mesh = Mesh1D(4, Basis.Legendre, Quadrature.GaussLobatto)
        x = np.asarray(logical_coordinates(mesh)[0])
        u = np.exp(x)
        filter_matrix = zero_lowest_modes(mesh, 2)
        u_filtered = filter_matrix @ u
        filtered_modes = nodal_to_modal_matrix(mesh) @ u_filtered
        npt.assert_allclose(filtered_modes[:2], 0.0, 1e-12, 1e-12)


if __name__ == "__main__":
    unittest.main()
