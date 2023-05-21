#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.ExtractDatFromH5 import (
    extract_dat_command,
    extract_dat_files,
)


class TestExtractDatFromH5(unittest.TestCase):
    def setUp(self):
        self.unit_test_src_dir = spectre_informer.unit_test_src_path()
        self.unit_test_build_dir = spectre_informer.unit_test_build_path()
        self.h5_filename = os.path.join(
            self.unit_test_src_dir, "Visualization/Python", "DatTestData.h5"
        )
        self.test_dir = os.path.join(self.unit_test_build_dir, "IO/H5/Python")
        self.created_dir = os.path.join(self.test_dir, "extracted_DatTestData")

        os.makedirs(self.test_dir, exist_ok=True)
        if os.path.exists(self.created_dir):
            shutil.rmtree(self.created_dir)

    def tearDown(self):
        if os.path.exists(self.created_dir):
            shutil.rmtree(self.created_dir)

    def test_extract_dat_files(self):
        with spectre_h5.H5File(self.h5_filename, "r") as h5file:
            expected_memory_data = np.array(
                h5file.get_dat("/Group0/MemoryData").get_data()
            )
            h5file.close_current_object()
            expected_timestep_data = np.array(
                h5file.get_dat("/TimeSteps2").get_data()
            )

        # All defaults, data should be same as expected
        extract_dat_files(
            self.h5_filename,
            out_dir=self.created_dir,
            num_cores=1,
            precision=16,
        )

        self.assertTrue(os.path.exists(self.created_dir))

        memory_data_path = os.path.join(
            self.created_dir, "Group0", "MemoryData.dat"
        )
        timestep_data_path = os.path.join(self.created_dir, "TimeSteps2.dat")

        memory_data = np.loadtxt(memory_data_path)
        timestep_data = np.loadtxt(timestep_data_path)

        npt.assert_almost_equal(memory_data, expected_memory_data)
        npt.assert_almost_equal(timestep_data, expected_timestep_data)

        # Test that we get an error if we try to run again and 'force' flag is
        # False.
        self.failUnlessRaises(
            ValueError,
            extract_dat_files,
            self.h5_filename,
            out_dir=self.created_dir,
            num_cores=1,
            precision=16,
        )

        # Try with 'force' flag True this time
        extract_dat_files(
            self.h5_filename,
            out_dir=self.created_dir,
            num_cores=1,
            precision=16,
            force=True,
        )

        memory_data = np.loadtxt(memory_data_path)
        timestep_data = np.loadtxt(timestep_data_path)

        npt.assert_almost_equal(memory_data, expected_memory_data)
        npt.assert_almost_equal(timestep_data, expected_timestep_data)

        if os.path.exists(self.created_dir):
            shutil.rmtree(self.created_dir)

        # Parallelize. Use 2 cores (this is all we get on CI)
        extract_dat_files(
            self.h5_filename,
            out_dir=self.created_dir,
            num_cores=2,
            precision=16,
        )

        memory_data = np.loadtxt(memory_data_path)
        timestep_data = np.loadtxt(timestep_data_path)

        npt.assert_almost_equal(memory_data, expected_memory_data)
        npt.assert_almost_equal(timestep_data, expected_timestep_data)

        if os.path.exists(self.created_dir):
            shutil.rmtree(self.created_dir)

        # Extract single file to stdout
        extract_dat_files(
            self.h5_filename,
            out_dir=None,
            num_cores=1,
            precision=16,
            list=False,
            force=False,
            subfiles=["TimeSteps2.dat"],
        )

        # We shouldn't have created an out directory
        self.assertFalse(os.path.exists(self.created_dir))

        self.failUnlessRaises(
            ValueError,
            extract_dat_files,
            self.h5_filename,
            out_dir=None,
            num_cores=1,
            precision=16,
            list=False,
            force=False,
            subfiles=["TimeSteps2.dat", "Group0/MemoryData.dat"],
        )

        # We don't test the '--list' flag as this is effectively just
        # available_subfiles()

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            extract_dat_command,
            [self.h5_filename, self.created_dir],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
