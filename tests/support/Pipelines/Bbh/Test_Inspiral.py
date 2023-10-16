# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.Bbh.Inspiral import (
    inspiral_parameters,
    start_inspiral_command,
)


class TestInspiral(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/Inspiral"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()
        generate_id(
            mass_ratio=1.5,
            separation=20.0,
            orbital_angular_velocity=0.01,
            refinement_level=1,
            polynomial_order=5,
            run_dir=self.test_dir / "ID",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "SolveXcts"),
        )
        self.id_dir = self.test_dir / "ID"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_inspiral_parameters(self):
        with open(self.id_dir / "InitialData.yaml") as open_input_file:
            _, id_input_file = yaml.safe_load_all(open_input_file)
        params = inspiral_parameters(
            id_input_file=id_input_file,
            id_run_dir=self.id_dir,
            refinement_level=1,
            polynomial_order=5,
        )
        self.assertEqual(
            params["IdFileGlob"],
            str((self.id_dir).resolve() / "BbhVolume*.h5"),
        )
        self.assertAlmostEqual(params["ExcisionRadiusA"], 1.068)
        self.assertAlmostEqual(params["ExcisionRadiusB"], 0.712)
        self.assertEqual(params["XCoordA"], 8.0)
        self.assertEqual(params["XCoordB"], -12.0)
        self.assertEqual(params["InitialAngularVelocity"], 0.01)
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            start_inspiral_command(
                [
                    str(self.id_dir / "InitialData.yaml"),
                    "--refinement-level",
                    "1",
                    "--polynomial-order",
                    "5",
                    "-O",
                    str(self.test_dir / "Inspiral"),
                    "--no-submit",
                    "-e",
                    str(self.bin_dir / "EvolveGhBinaryBlackHole"),
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue(
            (self.test_dir / "Inspiral/Segment_0000/Inspiral.yaml").exists()
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
