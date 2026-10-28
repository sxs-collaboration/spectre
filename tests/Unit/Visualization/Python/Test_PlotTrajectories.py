# Distributed under the MIT License.
# See LICENSE.txt for details.
# unit test for plot trajectories

import logging
import os
import shutil
import unittest

import click
import h5py
import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.support.Logging import configure_logging
from spectre.testing.PostNewtonian import BinaryTrajectories
from spectre.Visualization.PlotTrajectories import (
    import_A_and_B,
    plot_trajectories_command,
)


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

    def write_trajectories(self, times_a, times_b, filename=None):
        with h5py.File(filename or self.h5_filename, "w") as h5_file:
            for ab, times, sign in zip("AB", (times_a, times_b), (1.0, -1.0)):
                times = np.asarray(times)
                coords = sign * np.column_stack(
                    (
                        np.cos(0.1 * times),
                        np.sin(0.1 * times),
                        np.zeros_like(times),
                    )
                )
                h5_file.create_dataset(
                    f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat",
                    data=np.column_stack((times, coords, coords)),
                )

    def test_alignment(self):
        self.write_trajectories([20.0, 0.0, 5.0, 10.0, 15.0], [0.0, 10.0, 20.0])
        a, b = import_A_and_B([self.h5_filename])
        npt.assert_array_equal(a[:, 0], [0.0, 10.0, 20.0])
        npt.assert_array_equal(a[:, 0], b[:, 0])
        npt.assert_allclose(np.linalg.norm(a[:, 1:] - b[:, 1:], axis=1), 2.0)

    def test_roundoff(self):
        times = np.array([-1.0, 0.0, 1.0, 1.0e8])
        self.write_trajectories(times, np.nextafter(times, np.inf))
        a, b = import_A_and_B([self.h5_filename])
        npt.assert_array_equal(a[:, 0], times)
        npt.assert_array_equal(a[:, 0], b[:, 0])

    def test_restart_duplicates(self):
        self.write_trajectories([2.0, 0.0, 1.0], [0.0, 1.0, 2.0])
        next_segment = os.path.join(self.test_dir, "NextSegment.h5")
        self.write_trajectories([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], next_segment)
        with h5py.File(next_segment, "r+") as h5_file:
            for ab in "AB":
                h5_file[f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat"][
                    :, 4
                ] = 42.0
        a, b = import_A_and_B([self.h5_filename, next_segment])
        for trajectory in (a, b):
            npt.assert_array_equal(trajectory[:, 0], [0.0, 1.0, 2.0, 3.0])
            npt.assert_array_equal(trajectory[1:, 1], 42.0)

    def test_no_common_times(self):
        for times_a, times_b in (
            ([0.0, 1.0], [2.0, 3.0]),
            ([1.0e8, 1.0e8 + 1.0], [1.0e8 + 1.0e-4, 1.0e8 + 1.0001]),
        ):
            with self.subTest(times_a=times_a, times_b=times_b):
                self.write_trajectories(times_a, times_b)
                with self.assertRaisesRegex(
                    click.UsageError, "no common observation times"
                ):
                    import_A_and_B([self.h5_filename])

    def test_ambiguous_times(self):
        for times_a, times_b in (
            ([1.0], [1.0, np.nextafter(1.0, np.inf)]),
            ([1.0, np.nextafter(1.0, np.inf)], [1.0]),
            ([1.0, 1.0 + 1.0e-14], [1.0 + 2.0 * np.finfo(float).eps, 2.0]),
        ):
            with self.subTest(times_a=times_a, times_b=times_b):
                self.write_trajectories(times_a, times_b)
                with self.assertRaisesRegex(
                    click.UsageError, "Ambiguous trajectory times"
                ):
                    import_A_and_B([self.h5_filename])

    def test_empty_trajectories(self):
        with self.assertRaisesRegex(click.UsageError, "No trajectory files"):
            import_A_and_B([])
        self.write_trajectories([], [0.0])
        with self.assertRaisesRegex(click.UsageError, "must contain samples"):
            import_A_and_B([self.h5_filename])

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
