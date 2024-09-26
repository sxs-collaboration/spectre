# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.SphericalHarmonics import (
    AngularOrdering,
    Strahlkorper,
    cartesian_coords,
    read_surface_ylm,
    read_surface_ylm_single_time,
    write_sphere_of_points_to_text_file,
    ylm_legend_and_data,
)


class TestStrahlkorper(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "NumericalAlgorithms/Strahlkorper/Python",
        )
        self.filename = os.path.join(self.test_dir, "Strahlkorper.h5")
        self.text_filename = os.path.join(
            self.test_dir, "PyStrahlkorperCoords.txt"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_strahlkorper(self):
        strahlkorper = Strahlkorper(
            l_max=12, radius=1.0, center=[0.0, 0.0, 0.0]
        )
        self.assertEqual(strahlkorper.l_max, 12)
        self.assertEqual(strahlkorper.m_max, 12)
        self.assertEqual(strahlkorper.physical_extents, [13, 25])
        self.assertEqual(strahlkorper.expansion_center, [0.0, 0.0, 0.0])
        self.assertEqual(strahlkorper.physical_center, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(strahlkorper.average_radius, 1.0)
        self.assertAlmostEqual(strahlkorper.radius(0.0, 0.0), 1.0)
        self.assertTrue(strahlkorper.point_is_contained([0.5, 0.0, 0.0]))
        x = np.array(cartesian_coords(strahlkorper))
        r = np.linalg.norm(x, axis=0)
        npt.assert_allclose(r, 1.0)

        legend, ylm_data = ylm_legend_and_data(strahlkorper, 1.0, 12)
        self.assertEqual(len(legend), 174)
        self.assertEqual(
            legend[:7],
            [
                "Time",
                "InertialExpansionCenter_x",
                "InertialExpansionCenter_y",
                "InertialExpansionCenter_z",
                "Lmax",
                "coef(0,0)",
                "coef(1,-1)",
            ],
        )
        self.assertEqual(ylm_data[:5], [1.0, 0.0, 0.0, 0.0, 12.0])

        with spectre_h5.H5File(self.filename, "w") as h5file:
            datfile = h5file.insert_dat(
                "/Strahlkorper", legend=legend, version=0
            )
            datfile.append(ylm_data)

        self.assertEqual(
            read_surface_ylm(self.filename, "Strahlkorper", 1)[0], strahlkorper
        )
        self.assertEqual(
            read_surface_ylm_single_time(
                self.filename, "Strahlkorper", 1.0, 0.0, True
            ),
            strahlkorper,
        )

        l_max = 4
        print("hello")
        # First write with wrong l_max
        write_sphere_of_points_to_text_file(
            radius=1.2,
            l_max=l_max - 1,
            center=[-0.1, -0.2, -0.3],
            output_file_name=self.text_filename,
            ordering=AngularOrdering.Cce,
        )
        # Test that if overwrite_file = False (the default) an exception is
        # raised
        self.assertRaises(
            RuntimeError,
            write_sphere_of_points_to_text_file,
            radius=1.2,
            l_max=l_max - 1,
            center=[-0.1, -0.2, -0.3],
            output_file_name=self.text_filename,
            ordering=AngularOrdering.Cce,
        )
        # Finally write the correct l_max so we can check that overwrite_file
        # works
        write_sphere_of_points_to_text_file(
            radius=1.2,
            l_max=l_max,
            center=[-0.1, -0.2, -0.3],
            output_file_name=self.text_filename,
            ordering=AngularOrdering.Cce,
            overwrite_file=True,
        )

        with open(self.text_filename, "r") as text_file:
            # Physical size of ylm::Spherepack
            num_points = (l_max + 1) * (2 * l_max + 1)
            all_lines = text_file.readlines()
            self.assertEqual(num_points, len(all_lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
