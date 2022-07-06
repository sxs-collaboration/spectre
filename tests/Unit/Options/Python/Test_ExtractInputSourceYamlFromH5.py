#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Options.ExtractInputSourceYamlFromH5 import *

import spectre.Informer as spectre_informer
import unittest
import os
import h5py

# For Py2 compatibility
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestExtractInputSourceYAMLFromH5(unittest.TestCase):
    def test_read_input_source_from_h5(self):
        h5_path = os.path.join(spectre_informer.unit_test_src_path(),
                               'Visualization/Python', 'SurfaceTestData0.h5')
        input_source = read_input_source_from_h5(h5_path)

        # Check extracted source input vs. extracting with h5py
        h5_file = h5py.File(h5_path, 'r')
        expected_input_source_h5py = h5_file.attrs['InputSource.yaml']
        h5_file.close()
        self.maxDiff = None
        self.assertEqual(input_source, expected_input_source_h5py)

        # Check extracted source input vs. the original .yaml file
        yaml_path = os.path.join(spectre_informer.unit_test_src_path(),
                                 'Options/Python', 'InputSource.yaml')
        yaml_file = open(yaml_path, 'r')
        expected_input_source_yaml = yaml_file.read()
        yaml_file.close()
        self.assertEqual(input_source, expected_input_source_yaml)


if __name__ == '__main__':
    unittest.main(verbosity=2)
