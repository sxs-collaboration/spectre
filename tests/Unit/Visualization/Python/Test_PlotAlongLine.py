# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

from click.testing import CliRunner

from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Visualization.PlotAlongLine import plot_along_line_command


class TestPlotAlongLine(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotAlongLine"
        )
        self.h5_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
        )
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cli(self):
        output_filename = os.path.join(self.test_dir, "output.pdf")
        runner = CliRunner()
        result = runner.invoke(
            plot_along_line_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "0",
                "-y",
                "Psi",
                "-A",
                "0,0,0",
                "-B",
                "1,1,1",
                "-o",
                output_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_filename))


if __name__ == "__main__":
    unittest.main(verbosity=2)
