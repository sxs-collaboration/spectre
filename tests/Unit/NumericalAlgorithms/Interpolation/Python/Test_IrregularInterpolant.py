# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.random
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.Interpolation import Irregular
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates


class TestIrregular(unittest.TestCase):
    # arbitrary polynomial functions of low order for exact interpolation
    def polynomial(self, coords):
        dim = len(coords)
        if dim == 1:
            x = coords[0]
            return x**2 + x + 1.0
        elif dim == 2:
            x, y = coords
            return 2.0 * x**2 + y**2 + y + x + 2.0
        elif dim == 3:
            x, y, z = coords
            return 3.0 * x**2 + 2.0 * y**2 + z**2 + 2.0 * y + z + x + 2.0
        else:
            raise ValueError(
                "Coordinates must have shape (dim, N) where dim is 1, 2, or 3."
            )

    def test_irregular(self):
        for dim in [1, 2, 3]:
            for quadrature in [Quadrature.Gauss, Quadrature.GaussLobatto]:
                for num_points in range(3, 10):
                    source_mesh = Mesh[dim](
                        num_points, Basis.Legendre, quadrature
                    )
                    target_logical_coords = (
                        numpy.random.rand(dim, 3) * 2.0 - 1.0
                    )

                    interpolant = Irregular[dim](
                        source_mesh=source_mesh,
                        target_logical_coords=[
                            DataVector(xi) for xi in target_logical_coords
                        ],
                    )

                    source_logical_coords = np.array(
                        logical_coordinates(source_mesh)
                    )

                    source_data = self.polynomial(source_logical_coords)
                    target_data = self.polynomial(target_logical_coords)

                    interpolated_data = interpolant.interpolate(
                        DataVector(source_data)
                    )
                    npt.assert_allclose(interpolated_data, target_data, 1e-14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
