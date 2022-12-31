#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.IO.H5.DeleteSubfiles import delete_subfiles_command

import h5py
import os
import shutil
import unittest
from click.testing import CliRunner
from spectre.Informer import unit_test_src_path, unit_test_build_path


class TestDeleteSubfiles(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(unit_test_build_path(),
                                     "IO/H5/Python/DeleteSubfile")
        self.h5_filename = os.path.join(self.test_dir, "Test.h5")
        os.makedirs(self.test_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(unit_test_src_path(), "Visualization/Python",
                         "DatTestData.h5"), self.h5_filename)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_deletes_subfile(self):
        runner = CliRunner()
        result = runner.invoke(delete_subfiles_command,
                               [self.h5_filename, "-d", "TimeSteps2.dat"],
                               catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output, "")
        with h5py.File(self.h5_filename, "r") as open_h5_file:
            self.assertNotIn("TimeSteps2.dat", open_h5_file.keys())


if __name__ == '__main__':
    unittest.main(verbosity=2)
