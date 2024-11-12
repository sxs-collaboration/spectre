# Distributed under the MIT License.
# See LICENSE.txt for details.
# Unit test for eccentricity control

import logging
import os
import shutil
import unittest

import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Pipelines.Bbh.EccentricityControl import eccentricity_control
from spectre.support.Logging import configure_logging
from spectre.testing.PostNewtonian import BinaryTrajectories


class TestEccentricityControl(unittest.TestCase):
    # Set up and prepare test directory and file paths
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines", "EccentricityControl"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestEccentricityControlData.h5"
        )
        self.id_input_file_path = os.path.join(self.test_dir, "Inspiral.yaml")
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
        data1 = {
            "Background": {
                "Binary": {"AngularVelocity": 0.01, "Expansion": 0.001}
            }
        }
        with open(self.id_input_file_path, "w") as yaml_file:
            # Keep first dictionary in this list empty to match
            # the structure of the real file
            yaml.dump_all([{}, data1], yaml_file)

    # Test the eccentricity control function with the created files
    def test_eccentricity_control(self):
        eccentricity_control(
            h5_file=self.h5_filename,
            id_input_file_path=self.id_input_file_path,
            tmin=0,
            tmax=10,
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
