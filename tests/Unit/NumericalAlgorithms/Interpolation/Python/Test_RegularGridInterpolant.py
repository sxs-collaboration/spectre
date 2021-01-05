# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest
from spectre import Spectral
from spectre import DataStructures
from spectre import Interpolation
from numpy.polynomial.legendre import Legendre
import numpy as np


class TestRegularGrid(unittest.TestCase):

    # arbitrary polynomial functions of low order for exact interpolation
    def polynomial(self, coords):
        dim = len(coords)
        if dim == 1:
            x = coords[0]
            return x**2 + x + 1.

        elif dim == 2:
            x, y = coords
            return 2. * x**2 + y**2 + y + x + 2.

        elif dim == 3:
            x, y, z = coords
            return 3. * x**2 + 2. * y**2 + z**2 + 2. * y + z + x + 2.

        else:
            raise ValueError(
                "Coordinates must have shape (dim, N) where dim is 1, 2, or 3."
            )

    def generate_gauss_nodes(self, num_points):
        return Legendre.basis(num_points).roots()

    def generate_gauss_lobatto_nodes(self, num_points):
        nodes = Legendre.basis(num_points - 1).deriv().roots()
        return np.concatenate(([-1], nodes, [1]))

    def logical_coordinates(self, mesh):
        """
        creates a uniform mesh of shape (dim, num_points) with the
        requested quadrature
        """

        if mesh.quadrature()[0] == Spectral.Quadrature.Gauss:
            nodes = self.generate_gauss_nodes(mesh.extents(0))
        elif mesh.quadrature()[0] == Spectral.Quadrature.GaussLobatto:
            nodes = self.generate_gauss_lobatto_nodes(mesh.extents(0))
        else:
            raise ValueError(
                "Only Gauss or GaussLobatto are implemented quadratures")

        grid_points = np.meshgrid(*(mesh.dim * (nodes, )))
        return np.stack(grid_points, 0).reshape(mesh.dim, -1)

    def test_regular_grid(self):
        for dim in range(1, 4):
            Mesh = [Spectral.Mesh1D, Spectral.Mesh2D, Spectral.Mesh3D][dim - 1]
            RegularGrid = [
                Interpolation.RegularGrid1D, Interpolation.RegularGrid2D,
                Interpolation.RegularGrid3D
            ][dim - 1]
            for quadrature in [
                    Spectral.Quadrature.Gauss, Spectral.Quadrature.GaussLobatto
            ]:
                for num_points in range(3, 10):
                    source_mesh = Mesh(num_points, Spectral.Basis.Legendre,
                                       quadrature)
                    target_mesh = Mesh(num_points + 2, Spectral.Basis.Legendre,
                                       quadrature)
                    interpolant = RegularGrid(source_mesh, target_mesh)

                    source_coords = self.logical_coordinates(source_mesh)
                    target_coords = self.logical_coordinates(target_mesh)

                    initial_data = self.polynomial(source_coords)
                    target_data = self.polynomial(target_coords)

                    interpolated_data = interpolant.interpolate(
                        DataStructures.DataVector(initial_data))
                    self.assertTrue(
                        np.allclose(target_data, interpolated_data, 1e-14,
                                    1e-14))


if __name__ == '__main__':
    unittest.main()
