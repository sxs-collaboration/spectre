# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import yaml
from click.testing import CliRunner

from spectre import Informer
from spectre.tools.CleanOutput import (
    MissingExpectedOutputError,
    clean_output,
    clean_output_command,
)


class TestCleanOutput(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            Informer.unit_test_build_path(), "tools", "CleanOutput"
        )
        self.input_file_path = os.path.join(self.test_dir, "Input.yaml")
        self.reduction_file_path = os.path.join(self.test_dir, "Reduction.h5")
        self.volume_file_path = os.path.join(self.test_dir, "Volume0.h5")
        os.makedirs(self.test_dir, exist_ok=True)
        with open(self.input_file_path, "w") as open_file:
            yaml.safe_dump_all(
                [{"ExpectedOutput": ["Reduction.h5", "Volume0.h5"]}, {}],
                open_file,
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_clean_output(self):
        with self.assertRaises(MissingExpectedOutputError):
            clean_output(
                input_file=self.input_file_path,
                output_dir=self.test_dir,
                force=False,
            )
        clean_output(
            input_file=self.input_file_path,
            output_dir=self.test_dir,
            force=True,
        )
        with open(self.reduction_file_path, "w"):
            pass
        with self.assertRaisesRegex(MissingExpectedOutputError, "Volume0.h5"):
            clean_output(
                input_file=self.input_file_path,
                output_dir=self.test_dir,
                force=False,
            )
        self.assertFalse(os.path.exists(self.reduction_file_path))
        with open(self.reduction_file_path, "w"):
            pass
        with open(self.volume_file_path, "w"):
            pass
        clean_output(
            input_file=self.input_file_path,
            output_dir=self.test_dir,
            force=False,
        )
        self.assertFalse(os.path.exists(self.reduction_file_path))
        self.assertFalse(os.path.exists(self.volume_file_path))

    def test_cli(self):
        runner = CliRunner()
        with open(self.reduction_file_path, "w"):
            pass
        with open(self.volume_file_path, "w"):
            pass
        result = runner.invoke(
            clean_output_command, ["-o", self.test_dir, self.input_file_path]
        )
        self.assertEqual(result.exit_code, 0)
        result = runner.invoke(
            clean_output_command,
            ["-o", self.test_dir, "-f", self.input_file_path],
        )
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
