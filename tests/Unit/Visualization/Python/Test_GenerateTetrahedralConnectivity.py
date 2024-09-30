# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
import os
import shutil
import sys
import unittest

import h5py
import numpy as np
from click.testing import CliRunner

import spectre.Informer as spectre_informer
from spectre.support.Logging import configure_logging
from spectre.Visualization.GenerateTetrahedralConnectivity import (
    generate_tetrahedral_connectivity,
    generate_tetrahedral_connectivity_command,
)


class TestGenerateTetrahedralConnectivity(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
            spectre_informer.unit_test_src_path(), "Visualization/Python"
        )
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "Visualization/TetrahedralConnectivity",
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_tetrahedral_connectivity(self):
        original_file = os.path.join(self.data_dir, "VolTestData0.h5")
        data_file = os.path.join(self.test_dir, "VolTestData0.h5")
        shutil.copy2(original_file, data_file)

        self.assertTrue(
            "tetrahedral_connectivity"
            in h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]
        )

        generate_tetrahedral_connectivity(
            h5file=data_file,
            subfile_name="element_data",
            start_time=0.0,
            stop_time=1.0,
            stride=1,
            coordinates="InertialCoordinates",
            force=True,
        )

        self.assertTrue(
            "tetrahedral_connectivity"
            in h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]
        )

        # Wild card because some HDF5 configurations add "synchronously"
        with self.assertRaisesRegex(ValueError, "Unable to.*create dataset"):
            generate_tetrahedral_connectivity(
                h5file=data_file,
                subfile_name="element_data",
                start_time=0.0,
                stop_time=1.0,
                stride=1,
                coordinates="InertialCoordinates",
            )

        self.assertTrue(
            "tetrahedral_connectivity"
            in h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]
        )

        original_data = np.asarray(
            h5py.File(original_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]["tetrahedral_connectivity"]
        )
        new_data = np.asarray(
            h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]["tetrahedral_connectivity"]
        )

    def test_cli(self):
        original_file = os.path.join(self.data_dir, "VolTestData0.h5")
        data_file = os.path.join(self.test_dir, "VolTestData0.h5")
        shutil.copy2(original_file, data_file)
        data_files = glob.glob(os.path.join(self.test_dir, "VolTestData0.h5"))

        self.assertTrue(
            "tetrahedral_connectivity"
            in h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]
        )

        runner = CliRunner()
        result = runner.invoke(
            generate_tetrahedral_connectivity_command,
            [
                *data_files,
                "-d",
                "element_data",
                "--force",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        self.assertTrue(
            "tetrahedral_connectivity"
            in h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]
        )

        result = runner.invoke(
            generate_tetrahedral_connectivity_command,
            [
                *data_files,
                "-d",
                "element_data",
            ],
            catch_exceptions=True,
        )
        self.assertEqual(result.exit_code, 1)

        self.assertTrue(
            "tetrahedral_connectivity"
            in h5py.File(data_file, "r")["element_data.vol"][
                "ObservationId1090594013349131584"
            ]
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
