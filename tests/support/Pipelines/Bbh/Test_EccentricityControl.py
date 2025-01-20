# Distributed under the MIT License.
# See LICENSE.txt for details.

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
from spectre.testing.PostNewtonian import BinaryTrajectories


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
        binary_trajectories = BinaryTrajectories(initial_separation=16)
        self.angular_velocity = binary_trajectories.angular_velocity(0)
        self.initial_separation = binary_trajectories.separation(0)
        times = np.arange(0, 1500, 1.0)
        positions = np.array(binary_trajectories.positions(times))
        with spectre_h5.H5File(self.h5_filename, "w") as h5_file:
            for i, ab in enumerate("AB"):
                dataset = h5_file.insert_dat(
                    f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat",
                    legend=[
                        "Time",
                        "GridCenter_x",
                        "GridCenter_y",
                        "GridCenter_z",
                        "InertialCenter_x",
                        "InertialCenter_y",
                        "InertialCenter_z",
                    ],
                    version=0,
                )
                for t, coords in zip(times, positions[i].T):
                    dataset.append([t, *coords, *coords])
                h5_file.close_current_object()

    def create_yaml_file(self):
        # Define YAML data and write it to the file
        metadata = {
            "TargetParams": {
                "MassRatio": 1.0,
                "MassA": 0.5,
                "MassB": 0.5,
                "DimensionlessSpinA": [0.0, 0.0, 0.0],
                "DimensionlessSpinB": [0.0, 0.0, 0.0],
                "Eccentricity": 0.0,
            },
            "Next": {
                "With": {
                    "control": True,
                    "control_refinement_level": 1,
                    "control_polynomial_order": 10,
                }
            },
        }

        data1 = {
            "Background": {
                "Binary": {
                    "AngularVelocity": self.angular_velocity,
                    "Expansion": -1e-6,
                    "XCoords": [
                        -self.initial_separation / 2.0,
                        self.initial_separation / 2.0,
                    ],
                    "ObjectLeft": {"KerrSchild": {"Mass": 0.5}},
                    "ObjectRight": {"KerrSchild": {"Mass": 0.5}},
                },
            }
        }

        # Pass both metadata and data1 to the YAML file
        with open(self.id_input_file_path, "w") as yaml_file:
            yaml.dump_all([metadata, data1], yaml_file)

    # Test the eccentricity control function with the created files
    def test_eccentricity_control(self):
        eccentricity_control(
            h5_files=self.h5_filename,
            id_input_file_path=self.id_input_file_path,
            pipeline_dir=self.test_dir,
            plot_output_dir=self.test_dir,
            scheduler=None,
            submit=False,
        )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            eccentricity_control_command,
            [
                self.h5_filename,
                "-i",
                self.id_input_file_path,
                "-d",
                self.test_dir,
                "--plot-output-dir",
                self.test_dir,
                "--no-schedule",
                "--no-submit",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((Path(self.test_dir) / "FigureEccRemoval.pdf").exists())


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
