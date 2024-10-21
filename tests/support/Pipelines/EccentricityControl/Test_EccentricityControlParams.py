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

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params,
)
from spectre.support.Logging import configure_logging
from spectre.testing.Pipelines.MockBinaryData import write_mock_trajectory_data


class TestEccentricityControlParams(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines", "EccentricityControl"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestPlotTrajectoriesReductions.h5"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # Write mock trajectory data to an H5 file
        self.initial_separation = 16.0
        self.angular_velocity = 0.015625
        self.eccentricity = 0.0
        write_mock_trajectory_data(
            self.h5_filename,
            t=np.arange(0, 2000, 2.0),
            separation=self.initial_separation,
        )

        # Write a mock initial data input file
        self.id_input_file_path = os.path.join(
            self.test_dir, "InitialData.yaml"
        )
        with open(self.id_input_file_path, "w") as open_input_file:
            yaml.safe_dump_all(
                [
                    {},
                    {
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
                    },
                ],
                open_input_file,
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ecc_control_params(self):
        ecc, ecc_std_dev, param_updates = eccentricity_control_params(
            h5_files=[self.h5_filename],
            id_input_file_path=self.id_input_file_path,
            tmin=0.0,
            tmax=1200.0,
            plot_output_dir=self.test_dir,
        )
        self.assertAlmostEqual(ecc, self.eccentricity)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
