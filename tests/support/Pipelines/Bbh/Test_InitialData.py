# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

from click.testing import CliRunner

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.Bbh.InitialData import generate_id_command, id_parameters


class TestInitialData(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/InitialData"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generate_id(self):
        params = id_parameters(
            mass_ratio=1.5,
            separation=20.0,
            orbital_angular_velocity=0.01,
            refinement_level=1,
            polynomial_order=5,
        )
        self.assertEqual(params["MassRight"], 0.6)
        self.assertEqual(params["MassLeft"], 0.4)
        self.assertEqual(params["XRight"], 8.0)
        self.assertEqual(params["XLeft"], -12.0)
        self.assertAlmostEqual(params["ExcisionRadiusRight"], 1.068)
        self.assertAlmostEqual(params["ExcisionRadiusLeft"], 0.712)
        self.assertEqual(params["OrbitalAngularVelocity"], 0.01)
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)
        # COM is zero
        self.assertAlmostEqual(
            params["MassRight"] * params["XRight"]
            + params["MassLeft"] * params["XLeft"],
            0.0,
        )

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            generate_id_command(
                [
                    "--mass-ratio",
                    "1.5",
                    "--separation",
                    "20",
                    "--orbital-angular-velocity",
                    "0.01",
                    "--refinement-level",
                    "1",
                    "--polynomial-order",
                    "5",
                    "-o",
                    str(self.test_dir),
                    "--no-submit",
                    "-e",
                    str(self.bin_dir / "SolveXcts"),
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue((self.test_dir / "InitialData.yaml").exists())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
