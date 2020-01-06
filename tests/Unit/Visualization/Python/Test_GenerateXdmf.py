# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.GenerateXdmf import generate_xdmf

import spectre.Informer as spectre_informer
import unittest
import os


class TestGenerateXdmf(unittest.TestCase):
    def test_generate_xdmf(self):
        data_file_prefix = os.path.join(spectre_informer.unit_test_path(),
                                        'IO', 'VolTestData')
        output_filename = 'Test_GenerateXdmf_output'
        if os.path.isfile(output_filename + '.xmf'):
            os.remove(output_filename + '.xmf')

        generate_xdmf(file_prefix=data_file_prefix,
                      output_filename=output_filename,
                      start_time=0.,
                      stop_time=1.,
                      stride=1)

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising and error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + '.xmf'))
        os.remove(output_filename + '.xmf')


if __name__ == '__main__':
    unittest.main(verbosity=2)
