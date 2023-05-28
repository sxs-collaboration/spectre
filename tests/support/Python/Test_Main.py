# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import yaml
from click.testing import CliRunner

from spectre.__main__ import cli
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.tools.CleanOutput import MissingExpectedOutputError


class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "support/Python/Main"
        )
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_config_file(self):
        # Collect some test data
        input_file = os.path.join(
            unit_test_src_path(),
            "../InputFiles/Poisson/ProductOfSinusoids1D.yaml",
        )
        # Create a config file
        config_file_path = os.path.join(self.test_dir, "config.yaml")
        with open(config_file_path, "w") as open_config_file:
            yaml.safe_dump({"clean-output": {"force": True}}, open_config_file)
        # Run. The 'clean-output' command should fail because there's no data to
        # clean up in the test dir, but with the 'force' option from the config
        # file it should pass.
        runner = CliRunner()
        with self.assertRaises(MissingExpectedOutputError):
            runner.invoke(
                cli,
                ["clean-output", "-o", self.test_dir, input_file],
                catch_exceptions=False,
            )
        result = runner.invoke(
            cli,
            ["clean-output", "-o", self.test_dir, input_file],
            catch_exceptions=False,
            env={"SPECTRE_CONFIG_FILE": config_file_path},
        )
        self.assertEqual(result.exit_code, 0, result.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
