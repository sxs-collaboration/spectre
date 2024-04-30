# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import unittest

from click.testing import CliRunner

from spectre.Informer import unit_test_src_path
from spectre.Pipelines.Bbh.FindHorizon import find_horizon_command


class TestFindHorizon(unittest.TestCase):
    def setUp(self):
        self.h5_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
        )
        self.output_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "Horizons.h5"
        )

    def test_cli(self):
        # Can't test more than that this code runs until it errors until we have
        # a way to generate data with an apparent horizon.
        runner = CliRunner()
        result = runner.invoke(
            find_horizon_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "0",
                "--l-max",
                "12",
                "--initial-radius",
                "0.5",
                "--center",
                "1.0",
                "1.0",
                "1.0",
                "-o",
                self.output_filename,
                "--output-coeffs-subfile",
                "HorizonCoeffs",
                "--output-coords-subfile",
                "HorizonCoords",
            ],
            catch_exceptions=True,
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn(
            "Failed to open dataset 'InverseSpatialMetric_xx'",
            str(result.exception),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
