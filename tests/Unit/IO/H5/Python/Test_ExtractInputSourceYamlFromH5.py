#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.ExtractInputSourceYamlFromH5 import (
    extract_input_source_from_h5_command,
)


class TestExtractInputSourceYAMLFromH5(unittest.TestCase):
    def test_cli(self):
        h5_path = os.path.join(
            spectre_informer.unit_test_src_path(),
            "Visualization/Python",
            "SurfaceTestData.h5",
        )
        output_path = os.path.join(
            spectre_informer.unit_test_build_path(),
            "IO/H5",
            "ExtractedInput.yaml",
        )

        runner = CliRunner()
        runner.invoke(
            extract_input_source_from_h5_command,
            [h5_path, output_path],
            catch_exceptions=False,
        )

        with open(output_path, "r") as open_file:
            extracted_input_source = open_file.read()
        with spectre_h5.H5File(h5_path, "r") as open_file:
            expected_input_source = open_file.input_source()
        self.assertEqual(extracted_input_source, expected_input_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
