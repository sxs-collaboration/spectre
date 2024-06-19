# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre import DataStructures


class TestIOH5File(unittest.TestCase):
    # Test Fixtures
    def setUp(self):
        self.file_name = "pythontest.h5"
        self.data_1 = [1.0, 3.5]
        self.data_1_array = np.array(self.data_1)
        self.data_2 = [3.0, 10.3]
        self.data_2_array = np.array(self.data_2)
        self.l_max = 2
        self.bondi_variables = [
            "EthInertialRetardedTime",
            "News",
            "Psi0",
            "Psi1",
            "Psi2",
            "Psi3",
            "Psi4",
            "Strain",
        ]
        self.cce_data_1 = {
            name: np.array([i] * 19)
            for i, name in enumerate(self.bondi_variables)
        }
        self.cce_data_2 = {
            name: np.array([i + 1] * 19)
            for i, name in enumerate(self.bondi_variables)
        }
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    def tearDown(self):
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    # Test whether an H5 file is created correctly,
    def test_name(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            self.assertEqual(self.file_name, h5file.name())
        # Also test with pathlib.Path
        with spectre_h5.H5File(
            file_name=Path(self.file_name), mode="a"
        ) as h5file:
            self.assertEqual(self.file_name, h5file.name())

    # Test whether a dat file can be added correctly
    def test_insert_dat(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            datfile = h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            self.assertEqual(datfile.get_version(), 0)

    # Test whether a cce file can be added correctly
    def test_insert_cce(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            ccefile = h5file.insert_cce(path="/cce_data", l_max=2, version=0)
            self.assertEqual(ccefile.get_version(), 0)

    # Test whether data can be added to the dat file correctly
    def test_append_dat(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            datfile = h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            datfile.append(self.data_1)
            outdata_array = np.asarray(datfile.get_data())
            npt.assert_array_equal(outdata_array[0], self.data_1_array)

    # Test whether data can be added to the cce file correctly
    def test_append_cce(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            ccefile = h5file.insert_cce(
                path="/cce_data", l_max=self.l_max, version=0
            )
            ccefile.append(self.cce_data_1)
            outdata_dict = ccefile.get_data()
            for name in self.bondi_variables:
                self.assertIn(name, outdata_dict)
                npt.assert_array_equal(
                    np.asarray(outdata_dict[name])[0], self.cce_data_1[name]
                )
                outdata = np.asarray(ccefile.get_data(name))[0]
                npt.assert_array_equal(outdata, self.cce_data_1[name])

    # More complicated test case for getting data subsets and dimensions
    def test_get_data_subset_dat(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            datfile = h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            datfile.append(self.data_1)
            datfile.append(self.data_2)
            outdata_array = datfile.get_data_subset(
                columns=[1], first_row=0, num_rows=2
            )
            npt.assert_array_equal(
                outdata_array, np.array([self.data_1[1:2], self.data_2[1:2]])
            )
            self.assertEqual(datfile.get_dimensions()[0], 2)

    # More complicated test case for getting cce subsets and dimensions
    def test_get_data_subset_cce(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            ccefile = h5file.insert_cce(
                path="/cce_data", l_max=self.l_max, version=0
            )
            ccefile.append(self.cce_data_1)
            ccefile.append(self.cce_data_2)
            outdata_dict = ccefile.get_data_subset(
                ell=[0, 1], first_row=0, num_rows=2
            )
            for name in self.bondi_variables:
                self.assertIn(name, outdata_dict)
                expected = np.array(
                    [self.cce_data_1[name][:9], self.cce_data_2[name][:9]],
                )
                npt.assert_array_equal(outdata_dict[name], expected)
                outdata = np.asarray(
                    ccefile.get_data_subset(
                        name, ell=[0, 1], first_row=0, num_rows=2
                    )
                )
                npt.assert_array_equal(outdata, expected)
                self.assertEqual(ccefile.get_dimensions(name)[0], 2)

    # Getting Attributes
    def test_get_legend(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            datfile = h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            self.assertEqual(datfile.get_legend(), ["Time", "Value"])
            self.assertEqual(datfile.get_version(), 0)
            h5file.close_current_object()
            ccefile = h5file.insert_cce(
                path="/cce_data", l_max=self.l_max, version=0
            )
            self.assertEqual(
                ccefile.get_legend(),
                [
                    "time",
                    "Real Y_0,0",
                    "Imag Y_0,0",
                    "Real Y_1,-1",
                    "Imag Y_1,-1",
                    "Real Y_1,0",
                    "Imag Y_1,0",
                    "Real Y_1,1",
                    "Imag Y_1,1",
                    "Real Y_2,-2",
                    "Imag Y_2,-2",
                    "Real Y_2,-1",
                    "Imag Y_2,-1",
                    "Real Y_2,0",
                    "Imag Y_2,0",
                    "Real Y_2,1",
                    "Imag Y_2,1",
                    "Real Y_2,2",
                    "Imag Y_2,2",
                ],
            )
            self.assertEqual(ccefile.get_version(), 0)

    # The header is not universal, just checking the part that is predictable
    def test_get_header(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            datfile = h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            self.assertEqual(datfile.get_header()[0:16], "#\n# File created")
            h5file.close_current_object()
            ccefile = h5file.insert_cce(
                path="/cce_data", l_max=self.l_max, version=0
            )
            self.assertEqual(ccefile.get_header()[0:16], "#\n# File created")

    def test_all_files(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            h5file.insert_vol(path="/root_vol_2", version=0)
            h5file.close_current_object()
            h5file.insert_vol(path="/root_vol_1", version=0)
            h5file.close_current_object()
            h5file.insert_vol(path="/group0/sub_vol", version=0)
            h5file.close_current_object()
            legend = ["Fake"]
            h5file.insert_dat(path="/root_dat", legend=legend, version=0)
            h5file.close_current_object()
            h5file.insert_cce(path="/root_cce", l_max=self.l_max, version=0)
            h5file.close_current_object()
            h5file.insert_dat(
                path="/group0/sub_dat_1", legend=legend, version=0
            )
            h5file.close_current_object()
            h5file.insert_dat(
                path="/group0/sub_dat_2", legend=legend, version=0
            )
            h5file.close_current_object()
            h5file.insert_cce(
                path="/group0/sub_cce_1", l_max=self.l_max, version=0
            )
            h5file.close_current_object()
            h5file.insert_cce(
                path="/group0/sub_cce_2", l_max=self.l_max, version=0
            )
            h5file.close_current_object()

            expected_all_files = [
                "/group0/sub_cce_1.cce",
                "/group0/sub_cce_2.cce",
                "/group0/sub_dat_1.dat",
                "/group0/sub_dat_2.dat",
                "/group0/sub_vol.vol",
                "/root_cce.cce",
                "/root_dat.dat",
                "/root_vol_1.vol",
                "/root_vol_2.vol",
                "/src.tar.gz",
            ]
            expected_dat_files = [
                "/group0/sub_dat_1.dat",
                "/group0/sub_dat_2.dat",
                "/root_dat.dat",
            ]
            expected_cce_files = [
                "/group0/sub_cce_1.cce",
                "/group0/sub_cce_2.cce",
                "/root_cce.cce",
            ]
            expected_vol_files = [
                "/group0/sub_vol.vol",
                "/root_vol_1.vol",
                "/root_vol_2.vol",
            ]

            all_files = h5file.all_files()
            dat_files = h5file.all_dat_files()
            cce_files = h5file.all_cce_files()
            vol_files = h5file.all_vol_files()

            self.assertEqual(all_files, expected_all_files)
            self.assertEqual(dat_files, expected_dat_files)
            self.assertEqual(cce_files, expected_cce_files)
            self.assertEqual(vol_files, expected_vol_files)

    def test_groups(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            h5file.close_current_object()
            h5file.insert_dat(
                path="/element_position", legend=["x", "y", "z"], version=0
            )
            h5file.close_current_object()
            h5file.insert_dat(
                path="/element_size", legend=["Time", "Size"], version=0
            )
            h5file.close_current_object()
            h5file.insert_cce(path="/cce_1", l_max=self.l_max, version=0)
            h5file.close_current_object()
            h5file.insert_cce(path="/cce_2", l_max=self.l_max, version=0)
            h5file.close_current_object()
            h5file.insert_cce(path="/cce_3", l_max=self.l_max, version=0)
            h5file.close_current_object()
            groups_spec = [
                "cce_1.cce",
                "cce_2.cce",
                "cce_3.cce",
                "element_data.dat",
                "element_position.dat",
                "element_size.dat",
                "src.tar.gz",
            ]
            for group_name in groups_spec:
                self.assertTrue(group_name in h5file.groups())

    def test_exceptions(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            # create existing file
            with self.assertRaisesRegex(
                RuntimeError, "File 'pythontest.h5' already exists and"
            ):
                spectre_h5.H5File(file_name=self.file_name, mode="w-")

            h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )

            # insert existing data file
            with self.assertRaisesRegex(
                RuntimeError, "/element_data already open. Cannot insert object"
            ):
                h5file.insert_dat(
                    path="/element_data", legend=["Time", "Value"], version=0
                )
            h5file.close_current_object()

            # grab non-existing data file
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot open the object '/element_dat.vol' because it",
            ):
                h5file.get_vol("/element_dat")

            h5file.close_current_object()
            h5file.insert_cce(path="/cce_data", l_max=self.l_max, version=0)

            # insert existing cce file
            with self.assertRaisesRegex(
                RuntimeError, "/cce_data already open. Cannot insert object"
            ):
                h5file.insert_cce(path="/cce_data", l_max=self.l_max, version=0)
            h5file.close_current_object()

            # grab non-existing cce file
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot open the object '/cce.cce' because it",
            ):
                h5file.get_cce("/cce", self.l_max)

            h5file.close_current_object()
            h5file.insert_vol(path="/volume_data", version=0)

            # insert existing volume data file
            with self.assertRaisesRegex(
                RuntimeError, "Object /volume_data already open. Cannot"
            ):
                h5file.insert_vol(path="/volume_data", version=0)
            h5file.close_current_object()

            # grab non-existing volume data file
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot open the object '/volume_dat.vol' because it",
            ):
                h5file.get_vol("/volume_dat")

    def test_input_source(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            self.assertEqual(h5file.input_source(), "")

    def test_context_manager(self):
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as f:
            self.assertEqual(f.name(), self.file_name)

    def test_simultaneous_access(self):
        # Create the file and close it
        with spectre_h5.H5File(file_name=self.file_name, mode="a") as h5file:
            datfile = h5file.insert_dat(
                path="/element_data", legend=["Time", "Value"], version=0
            )
            datfile.append(self.data_1)

        # Read with bindings and h5py simultaneously
        h5file2 = spectre_h5.H5File(file_name=self.file_name, mode="r")
        datfile = h5file2.get_dat("/element_data")
        npt.assert_array_equal(np.array(datfile.get_data())[0], self.data_1)
        h5file3 = h5py.File(self.file_name, "r", locking=False)
        npt.assert_array_equal(
            np.array(h5file3["element_data.dat"])[0], self.data_1
        )
        h5file2.close()
        h5file3.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
