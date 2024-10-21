# Distributed under the MIT License.
# See LICENSE.txt for details.
# Unit test for eccentricity control

import logging
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import yaml
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Pipelines.Bbh.EccentricityControl import (
    eccentricity_control,
    eccentricity_control_command,
)
from spectre.support.Logging import configure_logging
from spectre.testing.Pipelines.MockBinaryData import write_mock_trajectory_data


class TestEccentricityControl(unittest.TestCase):
    # Set up and prepare test directory and file paths
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines/Bbh/EccentricityControl"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestEccentricityControlData.h5"
        )
        self.id_input_file_path = os.path.join(
            self.test_dir, "InitialData.yaml"
        )
        # Clean up any existing test directory and create new one
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)
        # Create HDF5 and YAML files for the test
        self.create_h5_file()
        self.create_yaml_file()

    # Clean up and remove test directory after tests are done
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_h5_file(self):
        self.initial_separation = 16.0
        self.angular_velocity = 0.015625
        self.eccentricity = 0.0
        write_mock_trajectory_data(
            self.h5_filename,
            t=np.arange(0, 2000, 2.0),
            separation=self.initial_separation,
        )

    def create_yaml_file(self):
        # Define YAML data and write it to the file
        data1 = {
            "Background": {
                "Binary": {
                    "AngularVelocity": self.angular_velocity,
                    "Expansion": -1e-6,
                    "XCoords": [
                        self.initial_separation / 2.0,
                        -self.initial_separation / 2.0,
                    ],
                    "ObjectLeft": {"KerrSchild": {"Mass": 0.5}},
                    "ObjectRight": {"KerrSchild": {"Mass": 0.5}},
                },
            }
        }
        with open(self.id_input_file_path, "w") as yaml_file:
            # Keep first dictionary in this list empty to match
            # the structure of the real file
            yaml.dump_all([{}, data1], yaml_file)

    # Test the eccentricity control function with the created files
    def test_eccentricity_control(self):
        eccentricity_control(
            h5_files=self.h5_filename,
            id_input_file_path=self.id_input_file_path,
        )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            eccentricity_control_command,
            [
                self.h5_filename,
                "-i",
                self.id_input_file_path,
                "-o",
                self.test_dir,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((Path(self.test_dir) / "FigureEccRemoval.pdf").exists())


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
