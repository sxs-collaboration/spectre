# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

import spectre
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.support.Logging import configure_logging
from spectre.support.RunNext import run_next, run_next_command


class TestRunNext(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(unit_test_build_path(), "RunNext").resolve()
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.input_file_path = self.test_dir / "Input.yaml"
        self.next_entrypoint = {
            "Run": "spectre.__main__:cli",
            "With": {
                "args": ["--version"],
                "standalone_mode": False,
            },
        }
        with open(self.input_file_path, "w") as open_input_file:
            yaml.dump_all(
                [
                    {"Next": self.next_entrypoint},
                    {},
                ],
                open_input_file,
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_next(self):
        with open(self.input_file_path, "r") as open_input_file:
            metadata = next(yaml.safe_load_all(open_input_file))
        result = run_next(
            metadata["Next"],
            input_file_path=self.input_file_path,
            cwd=self.test_dir,
        )
        self.assertEqual(result, 0)

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            run_next_command,
            [str(self.input_file_path)],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(spectre.__version__, result.stdout)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
