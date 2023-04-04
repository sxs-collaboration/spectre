# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import h5py
import pandas.testing as pdt
import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.Visualization.ReadH5 import available_subfiles, to_dataframe


class TestReadH5(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(spectre_informer.unit_test_src_path(),
                                     'Visualization/Python')

    def test_available_subfiles(self):
        with h5py.File(os.path.join(self.data_dir, "DatTestData.h5"),
                       "r") as open_file:
            self.assertEqual(available_subfiles(open_file, extension=".dat"),
                             ['Group0/MemoryData.dat', 'TimeSteps2.dat'])
            self.assertEqual(available_subfiles(open_file, extension=".vol"),
                             [])
        with h5py.File(os.path.join(self.data_dir, "VolTestData0.h5"),
                       "r") as open_file:
            self.assertEqual(available_subfiles(open_file, extension=".dat"),
                             [])
            self.assertEqual(available_subfiles(open_file, extension=".vol"),
                             ["element_data.vol"])

    def test_to_dataframe(self):
        # h5py subfile
        with h5py.File(os.path.join(self.data_dir, "DatTestData.h5"),
                       "r") as open_file:
            df = to_dataframe(open_file["TimeSteps2.dat"])
            self.assertEqual(df.columns[0], "Time")
        # SpECTRE subfile
        with spectre_h5.H5File(os.path.join(self.data_dir, "DatTestData.h5"),
                               "r") as open_file:
            df2 = to_dataframe(open_file.get_dat("/TimeSteps2"))
            pdt.assert_frame_equal(df2, df)


if __name__ == '__main__':
    unittest.main(verbosity=2)
