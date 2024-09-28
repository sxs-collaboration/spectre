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
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.Bbh.PostprocessId import postprocess_id_command
from spectre.support.Logging import configure_logging


class TestPostprocessId(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/PostprocessId"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()
        generate_id(
            mass_a=0.6,
            mass_b=0.4,
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

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        # Can only test that the command runs until this error until we have
        # some BBH initial data to postprocess.
        with self.assertRaisesRegex(ValueError, "Number of observations"):
            postprocess_id_command(
                [
                    str(self.id_dir / "InitialData.yaml"),
                ]
            )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
