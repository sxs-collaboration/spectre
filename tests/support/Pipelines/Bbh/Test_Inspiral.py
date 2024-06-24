# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
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
            dimensionless_spin_a=[0.0, 0.0, 0.0],
            dimensionless_spin_b=[0.0, 0.0, 0.0],
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
            refinement_level=1,
            polynomial_order=5,
            run_dir=self.test_dir / "ID",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "SolveXcts"),
        )
        self.id_dir = self.test_dir / "ID"
        # Purposefully not in the ID directory
        self.horizons_filename = self.test_dir / "Horizons.h5"
        with spectre_h5.H5File(
            str(self.horizons_filename.resolve()), "a"
        ) as horizons_file:
            legend = ["Time", "ChristodoulouMass", "DimensionlessSpinMagnitude"]
            for subfile_name in ["AhA", "AhB"]:
                horizons_file.close_current_object()
                dat_file = horizons_file.try_insert_dat(subfile_name, legend, 0)
                dat_file.append([[0.0, 1.0, 0.3]])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_inspiral_parameters(self):
        with open(self.id_dir / "InitialData.yaml") as open_input_file:
            _, id_input_file = yaml.safe_load_all(open_input_file)
        params = inspiral_parameters(
            id_input_file=id_input_file,
            id_run_dir=self.id_dir,
            id_horizons_path=self.horizons_filename,
            refinement_level=1,
            polynomial_order=5,
        )
        self.assertEqual(
            params["IdFileGlob"],
            str((self.id_dir).resolve() / "BbhVolume*.h5"),
        )
        self.assertAlmostEqual(params["ExcisionRadiusA"], 1.08936)
        self.assertAlmostEqual(params["ExcisionRadiusB"], 0.72624)
        self.assertEqual(params["XCoordA"], 8.0)
        self.assertEqual(params["XCoordB"], -12.0)
        self.assertEqual(params["InitialAngularVelocity"], 0.01)
        self.assertEqual(params["RadialExpansionVelocity"], -1.0e-5)
        self.assertEqual(
            params["HorizonsFile"], str(self.horizons_filename.resolve())
        )
        self.assertEqual(params["AhASubfileName"], "AhA/Coefficients")
        self.assertEqual(params["AhBSubfileName"], "AhB/Coefficients")
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)
        # Control system
        self.assertEqual(params["MaxDampingTimescale"], 20.0)
        self.assertEqual(params["KinematicTimescale"], 0.4)
        self.assertAlmostEqual(params["SizeATimescale"], 0.04)
        self.assertAlmostEqual(params["SizeBTimescale"], 0.04)
        self.assertAlmostEqual(params["ShapeATimescale"], 2.0)
        self.assertAlmostEqual(params["ShapeBTimescale"], 2.0)
        self.assertEqual(params["SizeIncreaseThreshold"], 1e-3)
        self.assertEqual(params["DecreaseThreshold"], 1e-4)
        self.assertEqual(params["IncreaseThreshold"], 2.5e-5)
        self.assertEqual(params["SizeAMaxTimescale"], 20)
        self.assertEqual(params["SizeBMaxTimescale"], 20)
        # Constraint damping
        self.assertEqual(params["Gamma0Constant"], 5e-4)
        self.assertEqual(params["Gamma0LeftAmplitude"], 4.0)
        self.assertEqual(params["Gamma0LeftWidth"], 7.0)
        self.assertEqual(params["Gamma0RightAmplitude"], 4.0)
        self.assertEqual(params["Gamma0RightWidth"], 7.0)
        self.assertEqual(params["Gamma0OriginAmplitude"], 3.75e-2)
        self.assertEqual(params["Gamma0OriginWidth"], 50.0)
        self.assertEqual(params["Gamma1Width"], 200.0)

    def test_cli(self):
        common_args = [
            str(self.id_dir / "InitialData.yaml"),
            "--refinement-level",
            "1",
            "--polynomial-order",
            "5",
            "--id-horizons-path",
            str(self.horizons_filename),
            "-E",
            str(self.bin_dir / "EvolveGhBinaryBlackHole"),
        ]
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            start_inspiral_command(
                common_args
                + [
                    "-O",
                    str(self.test_dir / "Inspiral"),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue(
            (self.test_dir / "Inspiral/Segment_0000/Inspiral.yaml").exists()
        )
        # Test with pipeline directory
        try:
            start_inspiral_command(
                common_args
                + [
                    "-d",
                    str(self.test_dir / "Pipeline"),
                    "--continue-with-ringdown",
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        with open(
            self.test_dir / "Pipeline/002_Inspiral/Segment_0000/Inspiral.yaml",
            "r",
        ) as open_input_file:
            metadata = next(yaml.safe_load_all(open_input_file))
        self.assertEqual(
            metadata["Next"],
            {
                "Run": "spectre.Pipelines.Bbh.Ringdown:start_ringdown",
                "With": {
                    "inspiral_input_file_path": "__file__",
                    "inspiral_run_dir": "./",
                    "pipeline_dir": str(self.test_dir.resolve() / "Pipeline"),
                    "refinement_level": 1,
                    "polynomial_order": 5,
                    "scheduler": "None",
                    "copy_executable": "None",
                    "submit_script_template": "None",
                    "submit": True,
                },
            },
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
