# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures.Tensor import DataVector, Frame, tnsr
from spectre.Domain import ElementId, ElementMap
from spectre.Domain.CoordinateMaps import (
    CompositionMapElementLogicalBlockLogicalInertial3D,
)
from spectre.Domain.Creators import Sphere


class TestComposition(unittest.TestCase):
    def test_composition(self):
        domain = Sphere(
            inner_radius=1.0,
            outer_radius=3.0,
            initial_refinement=0,
            initial_number_of_grid_points=5,
            use_equiangular_map=True,
            excise=True,
        ).domain()
        element_id = ElementId[3](0)
        element_map = ElementMap(element_id, domain)
        self.assertIsInstance(
            element_map, CompositionMapElementLogicalBlockLogicalInertial3D
        )
        # ElementLogical -> Inertial
        npt.assert_almost_equal(
            element_map(
                tnsr.I[DataVector, 3, Frame.ElementLogical](
                    np.array([[0.0, 0.0, -1.0]]).T
                )
            ),
            np.array([[0.0, 0.0, 1.0]]).T,
        )
        # ElementLogical -> BlockLogical
        npt.assert_almost_equal(
            element_map.element_logical_to_block_logical(
                tnsr.I[DataVector, 3, Frame.ElementLogical](
                    np.array([[0.0, 0.0, -1.0]]).T
                )
            ),
            np.array([[0.0, 0.0, -1.0]]).T,
        )
        # BlockLogical -> Inertial
        npt.assert_almost_equal(
            element_map.block_logical_to_inertial(
                tnsr.I[DataVector, 3, Frame.BlockLogical](
                    np.array([[0.0, 0.0, -1.0]]).T
                )
            ),
            np.array([[0.0, 0.0, 1.0]]).T,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
