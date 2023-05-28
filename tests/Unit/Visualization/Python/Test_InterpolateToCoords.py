# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Visualization.InterpolateToCoords import (
    interpolate_to_coords,
    interpolate_to_coords_command,
)


class TestInterpolateToCoords(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "InterpolateToCoords"
        )
        self.h5_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
        )
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_interpolate_to_coords(self):
        open_h5_file = spectre_h5.H5File(self.h5_filename, "r")
        volfile = open_h5_file.get_vol("/element_data")
        obs_id = volfile.list_observation_ids()[0]
        # Domain is [0, 2 pi]^3, so we check the endpoints
        coords = np.array([3 * [0.0], 3 * [2 * np.pi]])
        (psi,) = interpolate_to_coords(
            volfile,
            obs_id=obs_id,
            tensor_components=["Psi"],
            target_coords=coords,
        )
        npt.assert_allclose(psi, [-0.07059807, -0.06781784])

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            interpolate_to_coords_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "0",
                "-y",
                "Psi",
                "-p",
                "0,0,0",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("-0.0705980", result.output, result.output)

        coords_file = os.path.join(self.test_dir, "coords.txt")
        coords = np.array([3 * [0.0], 3 * [2 * np.pi]])
        np.savetxt(coords_file, coords)
        result_file = os.path.join(self.test_dir, "result.txt")
        result = runner.invoke(
            interpolate_to_coords_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "0",
                "-y",
                "Psi",
                "-t",
                coords_file,
                "-o",
                result_file,
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        result_data = np.loadtxt(result_file)
        npt.assert_allclose(
            result_data, np.hstack((coords, [[-0.07059807], [-0.06781784]]))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
