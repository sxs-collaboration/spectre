# Distributed under the MIT License.
# See LICENSE.txt for details.
# unit test for plot trajectories

import logging
import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.support.Logging import configure_logging
from spectre.testing.PostNewtonian import BinaryTrajectories
from spectre.Visualization.PlotTrajectories import plot_trajectories_command


class TestPlotTrajectories(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotTrajectories"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestPlotTrajectoriesReductions.h5"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)
        self.create_h5_file()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_h5_file(self):
        binary_trajectories = BinaryTrajectories(initial_separation=16)
        times = np.linspace(0, 200, 100)
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

    def test_cli(self):
        output_filename = os.path.join(self.test_dir, "output.pdf")
        runner = CliRunner()
        result = runner.invoke(
            plot_trajectories_command,
            [
                self.h5_filename,
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_filename))


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
