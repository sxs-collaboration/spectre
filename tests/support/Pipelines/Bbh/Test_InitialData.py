# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import numpy.testing as npt
import yaml
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.Bbh.InitialData import generate_id_command, id_parameters
from spectre.support.Logging import configure_logging


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
            conformal_mass_a=0.6,
            conformal_mass_b=0.4,
            horizon_rotation_a=[-0.04, -0.08, -0.1],
            horizon_rotation_b=[-0.3, -0.4, -0.4],
            center_of_mass_offset=[0.1, 0.2, 0.3],
            linear_velocity=[0.1, 0.2, 0.3],
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
            refinement_level=1,
            polynomial_order=5,
            negative_expansion_bc=True,
            target_params={
                "MassA": 0.6,
                "MassB": 0.4,
                "DimensionlessSpinA": [0.1, 0.2, 0.3],
                "DimensionlessSpinB": [0.4, 0.5, 0.6],
            },
        )
        self.assertEqual(params["ConformalMassRight"], 0.6)
        self.assertEqual(params["ConformalMassLeft"], 0.4)
        self.assertEqual(params["XRight"], 8.0 + 0.1)
        self.assertEqual(params["XLeft"], -12.0 + 0.1)
        self.assertEqual(
            [params[f"CenterOfMassOffset_{yz}"] for yz in "yz"],
            [0.2, 0.3],
        )
        self.assertEqual(
            [params[f"LinearVelocity_{xyz}"] for xyz in "xyz"],
            [0.1, 0.2, 0.3],
        )
        self.assertAlmostEqual(params["ExcisionRadiusRight"], 1.07546791205)
        self.assertAlmostEqual(params["ExcisionRadiusLeft"], 0.5504049327)
        self.assertEqual(params["OrbitalAngularVelocity"], 0.01)
        self.assertEqual(params["RadialExpansionVelocity"], -1.0e-5)
        self.assertEqual(
            [params[f"ConformalSpinRight_{xyz}"] for xyz in "xyz"],
            [0.1, 0.2, 0.3],
        )
        self.assertEqual(
            [params[f"ConformalSpinLeft_{xyz}"] for xyz in "xyz"],
            [0.4, 0.5, 0.6],
        )
        npt.assert_allclose(
            [params[f"HorizonRotationRight_{xyz}"] for xyz in "xyz"],
            [-0.04, -0.08, -0.1 + 0.01],
        )
        npt.assert_allclose(
            [params[f"HorizonRotationLeft_{xyz}"] for xyz in "xyz"],
            [-0.3, -0.4, -0.4 + 0.01],
        )
        self.assertAlmostEqual(params["FalloffWidthRight"], 6.479672589667676)
        self.assertAlmostEqual(params["FalloffWidthLeft"], 5.520327410332324)
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)
        # Newtonian center of mass (without offset) is zero
        self.assertAlmostEqual(
            params["ConformalMassRight"] * (params["XRight"] - 0.1)
            + params["ConformalMassLeft"] * (params["XLeft"] - 0.1),
            0.0,
        )

    def test_cli(self):
        common_args = [
            "--mass-ratio",
            "1.5",
            "--chi-A",
            "0.1",
            "0.2",
            "0.3",
            "--chi-B",
            "0.4",
            "0.5",
            "0.6",
            "--separation",
            "20",
            "--orbital-angular-velocity",
            "0.01",
            "--radial-expansion-velocity",
            "-1.0e-5",
            "--refinement-level",
            "1",
            "--polynomial-order",
            "5",
            "-E",
            str(self.bin_dir / "SolveXcts"),
            "--no-schedule",
        ]
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            generate_id_command(
                common_args
                + [
                    "-o",
                    str(self.test_dir),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue(
            (self.test_dir / "ControlParams_000/InitialData.yaml").exists()
        )
        # Test with pipeline directory
        try:
            generate_id_command(
                common_args
                + [
                    "-d",
                    str(self.test_dir / "Pipeline"),
                    "--evolve",
                    "--eccentricity-control",
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        with open(
            self.test_dir
            / "Pipeline/000_InitialData/ControlParams_000/InitialData.yaml",
            "r",
        ) as open_input_file:
            metadata = next(yaml.safe_load_all(open_input_file))
        self.assertEqual(
            metadata["TargetParams"],
            {
                "MassRatio": 1.5,
                "MassA": 0.6,
                "MassB": 0.4,
                "DimensionlessSpinA": [0.1, 0.2, 0.3],
                "DimensionlessSpinB": [0.4, 0.5, 0.6],
                "CenterOfMass": [0.0, 0.0, 0.0],
                "AdmLinearMomentum": [0.0, 0.0, 0.0],
                "Eccentricity": 0.0,
                "MeanAnomalyFraction": None,
                "NumOrbits": None,
                "TimeToMerger": None,
            },
        )
        self.assertEqual(
            metadata["Next"],
            {
                "Run": "spectre.Pipelines.Bbh.PostprocessId:postprocess_id",
                "With": {
                    "id_input_file_path": "__file__",
                    "id_run_dir": "./",
                    "pipeline_dir": str(self.test_dir.resolve() / "Pipeline"),
                    "horizon_l_max": 20,
                    "control": True,
                    "control_refinement_level": 1,
                    "control_polynomial_order": 5,
                    "control_params": [
                        "MassA",
                        "MassB",
                        "DimensionlessSpinA",
                        "DimensionlessSpinB",
                        "CenterOfMass",
                        "AdmLinearMomentum",
                    ],
                    "evolve": True,
                    "eccentricity_control": True,
                    "negative_expansion_bc": True,
                    "scheduler": "None",
                    "copy_executable": "None",
                    "submit_script_template": "None",
                    "submit": True,
                },
            },
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
