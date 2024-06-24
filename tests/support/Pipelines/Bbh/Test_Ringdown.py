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
from spectre.Pipelines.Bbh.Inspiral import start_inspiral
from spectre.Pipelines.Bbh.Ringdown import (
    ringdown_parameters,
    start_ringdown_command,
)


class TestInitialData(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/Ringdown"
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
        self.horizons_filename = self.id_dir / "Horizons.h5"
        with spectre_h5.H5File(
            str(self.horizons_filename.resolve()), "a"
        ) as horizons_file:
            legend = ["Time", "ChristodoulouMass", "DimensionlessSpinMagnitude"]
            for subfile_name in ["AhA", "AhB"]:
                horizons_file.close_current_object()
                dat_file = horizons_file.try_insert_dat(subfile_name, legend, 0)
                dat_file.append([[0.0, 1.0, 0.3]])
        start_inspiral(
            id_input_file_path=self.test_dir / "ID" / "InitialData.yaml",
            refinement_level=1,
            polynomial_order=5,
            segments_dir=self.test_dir / "Inspiral",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "EvolveGhBinaryBlackHole"),
        )
        self.inspiral_dir = self.test_dir / "Inspiral" / "Segment_0000"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ringdown_parameters(self):
        with open(self.inspiral_dir / "Inspiral.yaml") as open_input_file:
            _, inspiral_input_file = yaml.safe_load_all(open_input_file)
        params = ringdown_parameters(
            inspiral_input_file,
            self.inspiral_dir,
            refinement_level=1,
            polynomial_order=5,
        )
        self.assertEqual(
            params["IdFileGlob"],
            str((self.inspiral_dir).resolve() / "BbhVolume*.h5"),
        )
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            start_ringdown_command(
                [
                    str(self.inspiral_dir / "Inspiral.yaml"),
                    "--refinement-level",
                    "1",
                    "--polynomial-order",
                    "5",
                    "-O",
                    str(self.test_dir / "Ringdown"),
                    "--no-submit",
                    "-E",
                    str(self.bin_dir / "EvolveGhSingleBlackHole"),
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue(
            (self.test_dir / "Ringdown/Segment_0000/Ringdown.yaml").exists()
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
