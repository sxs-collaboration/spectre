# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.Pipelines.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params,
)
from spectre.support.Logging import configure_logging
from spectre.testing.PostNewtonian import BinaryTrajectories


class TestEccentricityControlParams(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines/EccentricityControl"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestEccentricityControlData.h5"
        )
        self.out_filename = os.path.join(
            self.test_dir, "TestEccentricityParams.yaml"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # Write mock trajectory data to an H5 file
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

        # Write a mock initial data input file
        self.id_input_file_path = os.path.join(
            self.test_dir, "InitialData.yaml"
        )
        with open(self.id_input_file_path, "w") as open_input_file:
            yaml.safe_dump_all(
                [
                    {
                        "TargetParams": {
                            "MassRatio": 1.0,
                            "MassA": 0.5,
                            "MassB": 0.5,
                            "DimensionlessSpinA": [0.0, 0.0, 0.0],
                            "DimensionlessSpinB": [0.0, 0.0, 0.0],
                            "Eccentricity": 0.0,
                        }
                    },
                    {
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
                    },
                ],
                open_input_file,
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ecc_control_params(self):
        ecc_params = eccentricity_control_params(
            h5_files=self.test_dir + "/TestEccentricity*.h5",
            id_input_file_path=self.id_input_file_path,
            tmin=0.0,
            tmax=1200.0,
            plot_output_dir=self.test_dir,
            ecc_params_output_file=self.out_filename,
        )
        self.assertAlmostEqual(ecc_params["Eccentricity"], 0.0, delta=1e-5)
        with open(self.out_filename, "r") as open_output_file:
            self.assertEqual(yaml.safe_load(open_output_file), ecc_params)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
