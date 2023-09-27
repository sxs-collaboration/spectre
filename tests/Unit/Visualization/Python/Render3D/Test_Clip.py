#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.Visualization.Render3D.Clip import render_clip_command


class TestClip(unittest.TestCase):
    def setUp(self):
        self.vol_test_data = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData.xmf"
        )
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization/Render3D/Clip"
        )
        self.output_file = os.path.join(self.test_dir, "test_clip.png")
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_render_clip(self):
        runner = CliRunner()
        result = runner.invoke(
            render_clip_command,
            [self.vol_test_data, "-y", "Psi", "-o", self.output_file],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(os.path.exists(self.output_file), msg=result.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
