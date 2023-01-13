#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.IO.H5.IterElements import iter_elements

import spectre.IO.H5 as spectre_h5
import numpy as np
import numpy.testing as npt
import os
import unittest
from spectre.Informer import unit_test_src_path
from spectre.Domain import ElementId, deserialize_functions_of_time
from spectre.Spectral import Mesh, Basis, Quadrature, logical_coordinates


class TestIterElements(unittest.TestCase):
    def setUp(self):
        self.volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5")
        self.subfile_name = "/element_data"

    def test_iter_elements(self):
        with spectre_h5.H5File(self.volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol(self.subfile_name)
            obs_id = volfile.list_observation_ids()[0]
            time = volfile.get_observation_value(obs_id)
            functions_of_time = deserialize_functions_of_time(
                volfile.get_functions_of_time(obs_id))

            elements = list(iter_elements(volfile, obs_id=obs_id))
            self.assertEqual(len(elements), 2)
            self.assertEqual(elements[0].id,
                             ElementId[3]("[B0,(L1I1,L0I0,L0I0)]"))
            self.assertEqual(
                elements[0].mesh, Mesh[3](4, Basis.Legendre,
                                          Quadrature.GaussLobatto))
            self.assertEqual(elements[1].id,
                             ElementId[3]("[B0,(L1I0,L0I0,L0I0)]"))
            self.assertEqual(
                elements[1].mesh, Mesh[3](4, Basis.Legendre,
                                          Quadrature.GaussLobatto))

            # Test fetching data, enumerating in a loop, determinism of
            # iteration order, list of volfiles
            tensor_components = [
                "InertialCoordinates_x",
                "InertialCoordinates_y",
                "InertialCoordinates_z",
                "Psi",
                "Error(Psi)",
            ]
            for i, (element, data) in enumerate(
                    iter_elements([volfile],
                                  obs_id=obs_id,
                                  tensor_components=tensor_components)):
                self.assertEqual(element.id, elements[i].id)
                self.assertEqual(element.mesh, elements[i].mesh)
                self.assertEqual(data.shape, (len(tensor_components), 4**3))
                # Test coordinates
                logical_coords = logical_coordinates(element.mesh)
                inertial_coords = element.map(logical_coords, time,
                                              functions_of_time)
                npt.assert_allclose(np.asarray(inertial_coords), data[:3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
