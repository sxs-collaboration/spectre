# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre import DataStructures
import spectre.IO.H5 as spectre_h5
import unittest
import numpy as np
import os
import numpy.testing as npt


class TestIOH5File(unittest.TestCase):
    # Test Fixtures
    def setUp(self):
        self.file_name = "pythontest.h5"
        self.data_1 = [1.0, 3.5]
        self.data_1_array = np.array(self.data_1)
        self.data_2 = [3.0, 10.3]
        self.data_2_array = np.array(self.data_2)
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    def tearDown(self):
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    # Test whether an H5 file is created correctly,
    def test_name(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        self.assertEqual(self.file_name, file_spec.name())
        file_spec.close()

    # Test whether a dat file can be added correctly
    def test_insert_dat(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)
        file_spec.close()
        datfile = file_spec.get_dat(path="/element_data")
        self.assertEqual(datfile.get_version(), 0)
        file_spec.close()

    # Test whether data can be added to the dat file correctly
    def test_append(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)
        file_spec.close()
        datfile = file_spec.get_dat(path="/element_data")
        datfile.append(self.data_1)
        outdata_array = np.asarray(datfile.get_data())
        npt.assert_array_equal(outdata_array[0], self.data_1_array)
        file_spec.close()

    # More complicated test case for getting data subsets and dimensions
    def test_get_data_subset(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)
        file_spec.close()
        datfile = file_spec.get_dat(path="/element_data")
        datfile.append(self.data_1)
        datfile.append(self.data_2)
        outdata_array = datfile.get_data_subset(columns=[1],
                                                first_row=0,
                                                num_rows=2)
        npt.assert_array_equal(outdata_array,
                               np.array([self.data_1[1:2], self.data_2[1:2]]))
        self.assertEqual(datfile.get_dimensions()[0], 2)
        file_spec.close()

    # Getting Attributes
    def test_get_legend(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)
        file_spec.close()
        datfile = file_spec.get_dat(path="/element_data")
        self.assertEqual(datfile.get_legend(), ["Time", "Value"])
        self.assertEqual(datfile.get_version(), 0)
        file_spec.close()

    # The header is not universal, just checking the part that is predictable
    def test_get_header(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)
        file_spec.close()
        datfile = file_spec.get_dat(path="/element_data")
        self.assertEqual(datfile.get_header()[0:16], "#\n# File created")
        file_spec.close()

    def test_all_files(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_vol(path="/root_vol_2", version=0)
        file_spec.close()
        file_spec.insert_vol(path="/root_vol_1", version=0)
        file_spec.close()
        file_spec.insert_vol(path="/group0/sub_vol", version=0)
        file_spec.close()
        legend = ["Fake"]
        file_spec.insert_dat(path="/root_dat", legend=legend, version=0)
        file_spec.close()
        file_spec.insert_dat(path="/group0/sub_dat_1",
                             legend=legend,
                             version=0)
        file_spec.close()
        file_spec.insert_dat(path="/group0/sub_dat_2",
                             legend=legend,
                             version=0)
        file_spec.close()

        expected_all_files = [
            "/group0/sub_dat_1.dat", "/group0/sub_dat_2.dat",
            "/group0/sub_vol.vol", "/root_dat.dat", "/root_vol_1.vol",
            "/root_vol_2.vol", "/src.tar.gz"
        ]
        expected_dat_files = [
            "/group0/sub_dat_1.dat", "/group0/sub_dat_2.dat", "/root_dat.dat"
        ]
        expected_vol_files = [
            "/group0/sub_vol.vol", "/root_vol_1.vol", "/root_vol_2.vol"
        ]

        all_files = file_spec.all_files()
        dat_files = file_spec.all_dat_files()
        vol_files = file_spec.all_vol_files()

        self.assertEqual(all_files, expected_all_files)
        self.assertEqual(dat_files, expected_dat_files)
        self.assertEqual(vol_files, expected_vol_files)

    def test_groups(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")
        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)
        file_spec.close()
        file_spec.insert_dat(path="/element_position",
                             legend=["x", "y", "z"],
                             version=0)
        file_spec.close()
        file_spec.insert_dat(path="/element_size",
                             legend=["Time", "Size"],
                             version=0)
        file_spec.close()
        groups_spec = [
            "element_data.dat", "element_position.dat", "element_size.dat",
            "src.tar.gz"
        ]
        for group_name in groups_spec:
            self.assertTrue(group_name in file_spec.groups())
        file_spec.close()

    def test_exceptions(self):
        file_spec = spectre_h5.H5File(file_name=self.file_name, mode="a")

        #create existing file
        with self.assertRaisesRegex(RuntimeError,
                                    "File 'pythontest.h5' already exists and"):
            spectre_h5.H5File(file_name=self.file_name, mode="w-")

        file_spec.insert_dat(path="/element_data",
                             legend=["Time", "Value"],
                             version=0)

        #insert existing data file
        with self.assertRaisesRegex(
                RuntimeError,
                "/element_data already open. Cannot insert object"):
            file_spec.insert_dat(path="/element_data",
                                 legend=["Time", "Value"],
                                 version=0)
        file_spec.close()

        # grab non-existing data file
        with self.assertRaisesRegex(
                RuntimeError,
                "Cannot open the object '/element_dat.vol' because it"):
            file_spec.get_vol("/element_dat")

        file_spec.close()
        file_spec.insert_vol(path="/volume_data", version=0)

        #insert existing volume data file
        with self.assertRaisesRegex(
                RuntimeError, "Object /volume_data already open. Cannot"):
            file_spec.insert_vol(path="/volume_data", version=0)
        file_spec.close()

        # grab non-existing volume data file
        with self.assertRaisesRegex(
                RuntimeError,
                "Cannot open the object '/volume_dat.vol' because it does"):
            file_spec.get_vol("/volume_dat")

        file_spec.close()

    def test_input_source(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            self.assertEqual(h5file.input_source(), "")

    def test_context_manager(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as f:
            self.assertEqual(f.name(), self.file_name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
