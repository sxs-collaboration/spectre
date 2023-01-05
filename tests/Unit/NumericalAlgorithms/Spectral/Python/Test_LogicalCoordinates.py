#Distributed under the MIT License.
#See LICENSE.txt for details.

from spectre.Spectral import (Mesh, Basis, Quadrature, collocation_points,
                              logical_coordinates)

import numpy as np
import numpy.testing as npt
import unittest
from random import randint
from spectre.DataStructures import DataVector


class TestLogicalCoordinates(unittest.TestCase):
    def test_logical_coordinates(self):
        bases = [Basis.Legendre, Basis.Chebyshev]
        quads = [Quadrature.Gauss, Quadrature.GaussLobatto]
        for dim in [1, 2, 3]:
            mesh = Mesh[dim](
                [randint(3, 12) for i in range(dim)],
                [bases[randint(0,
                               len(bases) - 1)] for i in range(dim)],
                [quads[randint(0,
                               len(quads) - 1)] for i in range(dim)])

            # Make sure we can convert to a Numpy array
            logical_coords = np.array(logical_coordinates(mesh))
            # Meshgrid is matrix-indexed, and flattened in Fortran order
            expected = np.stack([
                xyz.flatten(order="F") for xyz in np.meshgrid(*[
                    collocation_points(mesh_1d) for mesh_1d in mesh.slices()
                ],
                                                              indexing="ij")
            ])
            npt.assert_equal(logical_coords, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
