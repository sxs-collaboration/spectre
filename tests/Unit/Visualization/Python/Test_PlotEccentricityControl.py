# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import yaml
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path
from spectre.Visualization.PlotEccentricityControl import (
    plot_eccentricity_control_command,
)


class TestPlotEccentricityControl(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotEccentricityControl"
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.ecc_filenames = []
        for i in range(3):
            ecc_filename = os.path.join(
                self.test_dir,
                f"EccentricityParams{i}.yaml",
            )
            with open(ecc_filename, "w") as open_file:
                yaml.safe_dump(
                    {
                        "Eccentricity": 10 ** (-i),
                        "Omega0": 0.015 + np.random.rand() * 1e-4,
                        "Adot0": 1e-4 + np.random.rand() * 1e-4,
                        "NewOmega0": 0.015 + np.random.rand() * 1e-4,
                        "NewAdot0": 1e-4 + np.random.rand() * 1e-4,
                    },
                    open_file,
                )
            self.ecc_filenames.append(ecc_filename)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cli(self):
        runner = CliRunner()
        output_filename = os.path.join(self.test_dir, "output.pdf")
        result = runner.invoke(
            plot_eccentricity_control_command,
            self.ecc_filenames + ["-o", output_filename],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.exists(output_filename))


if __name__ == "__main__":
    unittest.main(verbosity=2)
