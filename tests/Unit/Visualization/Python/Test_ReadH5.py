# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.ReadH5 import available_subfiles

import h5py
import os
import spectre.Informer as spectre_informer
import unittest


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
