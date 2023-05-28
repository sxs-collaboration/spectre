#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId, deserialize_functions_of_time
from spectre.Informer import unit_test_src_path
from spectre.IO.H5.IterElements import (
    include_element,
    iter_elements,
    stripped_element_name,
)
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates


class TestIterElements(unittest.TestCase):
    def setUp(self):
        self.volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
        )
        self.subfile_name = "/element_data"

    def test_stripped_element_name(self):
        self.assertEqual(
            stripped_element_name("[B0,(L0I0,L1I0)]"), "B0,(L0I0,L1I0)"
        )
        self.assertEqual(
            stripped_element_name("B0,(L0I0,L1I0)"), "B0,(L0I0,L1I0)"
        )
        self.assertEqual(stripped_element_name(ElementId[1](0)), "B0,(L0I0)")
        self.assertEqual(
            stripped_element_name(ElementId[2]("[B34,(L0I0,L1I0)]")),
            "B34,(L0I0,L1I0)",
        )
        self.assertEqual(
            stripped_element_name(ElementId[3]("[B12,(L2I1,L1I0,L0I0)]")),
            "B12,(L2I1,L1I0,L0I0)",
        )

    def test_include_element(self):
        self.assertTrue(include_element("[B0,(L0I0,L1I0)]", None))
        self.assertFalse(include_element("[B0,(L0I0,L1I0)]", []))
        self.assertTrue(include_element("[B0,(L0I0,L1I0)]", ["B*,(L0I0,L1I0)"]))
        self.assertTrue(include_element("[B0,(L0I0,L1I0)]", ["*"]))
        self.assertTrue(include_element("[B0,(L0I0,L1I0)]", ["B0,*"]))
        self.assertTrue(include_element("[B0,(L0I0,L1I0)]", ["B0,(L0I0,L1I0)"]))
        self.assertFalse(include_element("[B0,(L0I0,L1I0)]", ["B1,*"]))
        self.assertTrue(include_element("[B0,(L0I0,L1I0)]", ["B0,*", "B1,*"]))
        self.assertTrue(
            include_element(ElementId[2]("[B0,(L0I0,L1I0)]"), ["B0,*", "B1,*"])
        )

    def test_iter_elements(self):
        with spectre_h5.H5File(self.volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol(self.subfile_name)
            all_obs_ids = volfile.list_observation_ids()
            obs_id = all_obs_ids[0]
            time = volfile.get_observation_value(obs_id)
            functions_of_time = deserialize_functions_of_time(
                volfile.get_functions_of_time(obs_id)
            )

            elements = list(iter_elements(volfile, obs_id))
            self.assertEqual(len(elements), 2)
            for element in elements:
                self.assertEqual(element.dim, 3)
                self.assertIn(
                    element.id,
                    [
                        ElementId[3]("[B0,(L1I0,L0I0,L0I0)]"),
                        ElementId[3]("[B0,(L1I1,L0I0,L0I0)]"),
                    ],
                )
                self.assertEqual(
                    element.mesh,
                    Mesh[3](4, Basis.Legendre, Quadrature.GaussLobatto),
                )

            # Make sure we can't mutate properties, in particular the time
            with self.assertRaises(FrozenInstanceError):
                elements[0].time = 2.0

            # Test fetching data, enumerating in a loop, determinism of
            # iteration order, list of volfiles, all obs IDs
            tensor_components = [
                "InertialCoordinates_x",
                "InertialCoordinates_y",
                "InertialCoordinates_z",
                "Psi",
                "Error(Psi)",
            ]
            for i, (element, data) in enumerate(
                iter_elements(
                    [volfile],
                    obs_ids=all_obs_ids,
                    tensor_components=tensor_components,
                )
            ):
                self.assertEqual(element.id, elements[i].id)
                self.assertEqual(element.mesh, elements[i].mesh)
                self.assertEqual(data.shape, (len(tensor_components), 4**3))
                # Test coordinates
                npt.assert_allclose(element.inertial_coordinates, data[:3])
                # Test Jacobians. Domain is [0, 2 pi]^3 split in half along
                # first dimension, so elements have size (pi, 2 pi, 2 pi).
                # Logical size is 2, so Jacobian is diag(0.5, 1, 1) * pi.
                npt.assert_allclose(element.jacobian.get(0, 0) / np.pi, 0.5)
                npt.assert_allclose(element.jacobian.get(1, 1) / np.pi, 1.0)
                npt.assert_allclose(element.jacobian.get(2, 2) / np.pi, 1.0)
                npt.assert_allclose(element.inv_jacobian.get(0, 0) * np.pi, 2.0)
                npt.assert_allclose(element.inv_jacobian.get(1, 1) * np.pi, 1.0)
                npt.assert_allclose(element.inv_jacobian.get(2, 2) * np.pi, 1.0)
                npt.assert_allclose(
                    element.det_jacobian.get() / np.pi**3, 0.5
                )
                for j in range(3):
                    for k in range(j):
                        npt.assert_allclose(
                            element.jacobian.get(j, k), 0.0, atol=1e-14
                        )
                        npt.assert_allclose(
                            element.inv_jacobian.get(j, k), 0.0, atol=1e-14
                        )

            # Test filtering elements
            self.assertEqual(
                len(
                    list(iter_elements(volfile, obs_id, element_patterns=None))
                ),
                2,
            )
            self.assertFalse(
                list(iter_elements(volfile, obs_id, element_patterns=["B1,*"]))
            )
            self.assertEqual(
                len(
                    list(
                        iter_elements(
                            volfile, obs_id, element_patterns=["B0,*"]
                        )
                    )
                ),
                2,
            )
            self.assertEqual(
                len(
                    list(
                        iter_elements(
                            volfile, obs_id, element_patterns=["B0,(L1I1*)"]
                        )
                    )
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
