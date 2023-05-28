# Distributed under the MIT License.
# See LICENSE.txt for details.

import shutil
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.tools.ValidateInputFile import (
    InvalidInputFileError,
    validate_input_file,
    validate_input_file_command,
)


class TestValidateInputFile(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(unit_test_build_path(), "tools/ValidateInputFile")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.executable = Path(
            unit_test_build_path(), "../../bin/SolvePoisson1D"
        )
        self.valid_input_file_path = Path(
            unit_test_src_path(),
            "../InputFiles/Poisson/ProductOfSinusoids1D.yaml",
        )
        self.invalid_input_file_path = self.test_dir / "InvalidInputFile.yaml"
        with open(self.valid_input_file_path, "r") as open_input_file:
            metadata, input_file = yaml.safe_load_all(open_input_file)
        del input_file["DomainCreator"]["Interval"]["LowerBound"]
        with open(self.invalid_input_file_path, "w") as open_input_file:
            yaml.safe_dump_all([metadata, input_file], open_input_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validate_input_file(self):
        validate_input_file(
            self.valid_input_file_path, executable=self.executable
        )
        with self.assertRaisesRegex(InvalidInputFileError, "LowerBound"):
            validate_input_file(
                self.invalid_input_file_path, executable=self.executable
            )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            validate_input_file_command,
            [str(self.valid_input_file_path), "-e", str(self.executable)],
        )
        self.assertEqual(result.exit_code, 0)
        result = runner.invoke(
            validate_input_file_command,
            [str(self.invalid_input_file_path), "-e", str(self.executable)],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("LowerBound", result.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
