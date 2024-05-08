# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

from click.testing import CliRunner

from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Visualization.PlotSlice import plot_slice_command


class TestPlotSlice(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotSlice"
        )
        self.h5_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
        )
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cli(self):
        runner = CliRunner()
        common_args = [
            self.h5_filename,
            "-d",
            "element_data",
            "-y",
            "Psi",
            "-C",
            "1,1,1",
            "-X",
            "2",
            "2",
            "-n",
            "0,0,1",
            "-u",
            "0,1,0",
        ]
        # Single plot
        output_filename = os.path.join(self.test_dir, "output.pdf")
        result = runner.invoke(
            plot_slice_command,
            common_args
            + [
                "--step",
                "0",
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_filename))
        # Animation
        output_filename = os.path.join(self.test_dir, "output.gif")
        result = runner.invoke(
            plot_slice_command,
            common_args
            + [
                "--animate",
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_filename))


if __name__ == "__main__":
    unittest.main(verbosity=2)
