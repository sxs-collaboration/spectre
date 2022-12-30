#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.IO.H5.IterElements import iter_elements

import spectre.IO.H5 as spectre_h5
import numpy.testing as npt
import os
import unittest
from spectre.Informer import unit_test_src_path
from spectre.Domain import ElementId
from spectre.Spectral import Mesh, Basis, Quadrature


class TestIterElements(unittest.TestCase):
    def setUp(self):
        self.volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5")
        self.subfile_name = "/element_data"

    def test_iter_elements(self):
        with spectre_h5.H5File(self.volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol(self.subfile_name)
            obs_id = volfile.list_observation_ids()[0]

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
            for i, (element, data) in enumerate(
                    iter_elements([volfile],
                                  obs_id=obs_id,
                                  tensor_components=["Psi", "Error(Psi)"])):
                self.assertEqual(element.id, elements[i].id)
                self.assertEqual(element.mesh, elements[i].mesh)
                npt.assert_equal(element.inertial_coords,
                                 elements[i].inertial_coords)
                self.assertEqual(data.shape, (2, 4**3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
