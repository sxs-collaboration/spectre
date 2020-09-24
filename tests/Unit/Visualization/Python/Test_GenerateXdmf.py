# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.GenerateXdmf import generate_xdmf

import spectre.Informer as spectre_informer
import unittest
import os

# For Py2 compatibility
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestGenerateXdmf(unittest.TestCase):
    def test_generate_xdmf(self):
        data_file_prefix = os.path.join(spectre_informer.unit_test_src_path(),
                                        'Visualization/Python', 'VolTestData')
        #write output file to same relative path in build directory
        output_filename = os.path.join(spectre_informer.unit_test_build_path(),
                                       'Visualization/Python',
                                       'Test_GenerateXdmf_output')
        if os.path.isfile(output_filename + '.xmf'):
            os.remove(output_filename + '.xmf')

        generate_xdmf(file_prefix=data_file_prefix,
                      output=output_filename,
                      subfile_name="element_data",
                      start_time=0.,
                      stop_time=1.,
                      stride=1,
                      coordinates='InertialCoordinates')

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising an error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + '.xmf'))
        os.remove(output_filename + '.xmf')

    def test_subfile_not_found(self):
        data_file_prefix = os.path.join(spectre_informer.unit_test_src_path(),
                                        'Visualization/Python', 'VolTestData')
        output_filename = 'Test_GenerateXdmf_subfile_not_found'
        if os.path.isfile(output_filename + '.xmf'):
            os.remove(output_filename + '.xmf')

        with self.assertRaisesRegex(ValueError, 'Could not open subfile'):
            generate_xdmf(file_prefix=data_file_prefix,
                          output=output_filename,
                          subfile_name="unknown_subfile",
                          start_time=0.,
                          stop_time=1.,
                          stride=1,
                          coordinates='InertialCoordinates')


if __name__ == '__main__':
    unittest.main(verbosity=2)
