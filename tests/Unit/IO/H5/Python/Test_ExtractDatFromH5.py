#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.IO.H5.ExtractDatFromH5 import *

import spectre.Informer as spectre_informer
import unittest
import os
import shutil


class TestExtractDatFromH5(unittest.TestCase):
    def setUp(self):
        self.unit_test_dir = spectre_informer.unit_test_src_path()
        self.h5_filename = os.path.join(self.unit_test_dir,
                                        "Visualization/Python",
                                        "DatTestData.h5")
        self.test_dir = os.path.join(self.unit_test_dir, "IO/H5/Python")
        self.created_dir_default = os.path.join(self.test_dir,
                                                "extracted_DatTestData")
        self.created_dir_input = os.path.join(self.test_dir, "WinterIsComing")

    def tearDown(self):
        if os.path.exists(self.created_dir_default):
            shutil.rmtree(self.created_dir_default)
        if os.path.exists(self.created_dir_input):
            shutil.rmtree(self.created_dir_input)

    def test_extract_dat_files(self):
        h5file = spectre_h5.H5File(self.h5_filename, "r")

        expected_memory_data = np.array(
            h5file.get_dat("/Group0/MemoryData").get_data())
        h5file.close()
        expected_timestep_data = np.array(
            h5file.get_dat("/TimeSteps2").get_data())
        h5file.close()

        # This will hold arguments. Since we aren't using an argument parser, we
        # just have to write them out by hand. Since there's only 3 this isn't
        # too bad
        kwargs_dict = {
            "force": False,
            "list": False,
            "output_directory": self.created_dir_default
        }

        # All defaults, data should be same as expected
        extract_dat_files(self.h5_filename, **kwargs_dict)

        self.assertTrue(os.path.exists(self.created_dir_default))

        memory_data_path = os.path.join(self.created_dir_default, "Group0",
                                        "MemoryData.dat")
        timestep_data_path = os.path.join(self.created_dir_default,
                                          "TimeSteps2.dat")

        memory_data = np.loadtxt(memory_data_path)
        timestep_data = np.loadtxt(timestep_data_path)

        self.assertEqual(memory_data.all(), expected_memory_data.all())
        self.assertEqual(timestep_data.all(), expected_timestep_data.all())

        # Test that we get an error if we try to run again and 'force' flag is
        # False.
        self.failUnlessRaises(ValueError, extract_dat_files, self.h5_filename,
                              **kwargs_dict)

        # Try with 'force' flag True this time
        kwargs_dict["force"] = True
        extract_dat_files(self.h5_filename, **kwargs_dict)

        memory_data = np.loadtxt(memory_data_path)
        timestep_data = np.loadtxt(timestep_data_path)

        self.assertEqual(memory_data.all(), expected_memory_data.all())
        self.assertEqual(timestep_data.all(), expected_timestep_data.all())

        # Try with new output directory
        kwargs_dict["force"] = False
        kwargs_dict["output_directory"] = self.created_dir_input
        extract_dat_files(self.h5_filename, **kwargs_dict)

        self.assertTrue(os.path.exists(self.created_dir_input))

        memory_data = np.loadtxt(
            os.path.join(self.created_dir_input, "Group0", "MemoryData.dat"))
        timestep_data = np.loadtxt(
            os.path.join(self.created_dir_input, "TimeSteps2.dat"))

        self.assertEqual(memory_data.all(), expected_memory_data.all())
        self.assertEqual(timestep_data.all(), expected_timestep_data.all())

        # We don't test the '--list' flag as this is effectively just
        # H5File.all_dat_files()


if __name__ == '__main__':
    unittest.main(verbosity=2)
