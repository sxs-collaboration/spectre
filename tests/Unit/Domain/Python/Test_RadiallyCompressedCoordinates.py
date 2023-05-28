# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr
from spectre.Domain import radially_compressed_coordinates
from spectre.Domain.CoordinateMaps import Distribution


class TestRadiallyCompressedCoordinates(unittest.TestCase):
    def test_radially_compressed_coordinates(self):
        inner_radius = 1.0
        outer_radius = 1.0e6
        expected_compressed_outer_radius = 6.0
        x = np.array([[inner_radius, outer_radius], [0.0, 0.0], [0.0, 0.0]])
        x_compressed = radially_compressed_coordinates(
            x,
            inner_radius=1.0,
            outer_radius=1.0e6,
            compression=Distribution.Logarithmic,
        )
        npt.assert_allclose(
            x_compressed[0], [inner_radius, expected_compressed_outer_radius]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
