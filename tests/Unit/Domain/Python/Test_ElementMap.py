# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, tnsr
from spectre.Domain import (
    ElementId,
    ElementMap,
    deserialize_domain,
    deserialize_functions_of_time,
)
from spectre.Domain.CoordinateMaps import (
    CoordinateMapElementLogicalToInertial1D,
)
from spectre.Domain.Creators import Interval
from spectre.Informer import unit_test_src_path


class TestElementMap(unittest.TestCase):
    def test_element_map(self):
        # Full domain is [0, 2]
        domain = Interval(
            lower_x=[0.0],
            upper_x=[2.0],
            is_periodic_in_x=[False],
            initial_refinement_level_x=[1],
            initial_number_of_grid_points_in_x=[3],
        ).create_domain()
        # Element is [0, 1]
        element_id = ElementId[1]("[B0,(L1I0)]")
        element_map = ElementMap(element_id, domain)
        self.assertIsInstance(
            element_map, CoordinateMapElementLogicalToInertial1D
        )
        self.assertFalse(element_map.is_identity())
        # Logical coords are [-1, 1]
        xi_data = np.random.rand(1, 4) * 2.0 - 1.0
        xi = tnsr.I[DataVector, 1, Frame.ElementLogical](xi_data)
        npt.assert_allclose(element_map(xi), (np.array(xi) + 1.0) / 2.0)
        npt.assert_allclose(
            element_map.jacobian(xi).get(0, 0), np.ones(4) * 0.5
        )
        npt.assert_allclose(
            element_map.inv_jacobian(xi).get(0, 0), np.ones(4) * 2.0
        )
        # Test inverse
        target_point_inside = tnsr.I[float, 1, Frame.Inertial](fill=0.0)
        self.assertAlmostEqual(
            element_map.inverse(target_point_inside).get(0), -1.0
        )
        # Inverses aren't currently guaranteed to return None outside the source
        # interval, so here the inertial coordinate 1.5 is mapped to a logical
        # coordinate of 2. (that's halfway into the next element to the right).
        target_point_outside = tnsr.I[float, 1, Frame.Inertial](fill=1.5)
        self.assertAlmostEqual(
            element_map.inverse(target_point_outside).get(0), 2.0
        )

    def test_inertial_coordinates(self):
        volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
        )
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            domain = deserialize_domain[3](volfile.get_domain(obs_id))
            functions_of_time = deserialize_functions_of_time(
                volfile.get_functions_of_time(obs_id)
            )

        element_id = ElementId[3]("[B0,(L1I0,L0I0,L0I0)]")
        element_logical_coords = tnsr.I[DataVector, 3, Frame.ElementLogical](
            3 * [DataVector([-1.0, 1.0])]
        )

        element_map = ElementMap(element_id, domain)
        inertial_coords = np.asarray(
            element_map(
                element_logical_coords,
                time=0.0,
                functions_of_time=functions_of_time,
            )
        )

        # Domain is [0, 2 pi]^3, and split in two in dimension 0
        npt.assert_allclose(
            inertial_coords,
            np.array([[0.0, np.pi], [0.0, 2 * np.pi], [0.0, 2 * np.pi]]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
