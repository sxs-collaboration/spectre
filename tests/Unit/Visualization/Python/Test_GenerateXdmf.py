# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.GenerateXdmf import (generate_xdmf,
                                                generate_xdmf_command)

import spectre.Informer as spectre_informer
import unittest
import logging
import os
import shutil
from click.testing import CliRunner


class TestGenerateXdmf(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(spectre_informer.unit_test_src_path(),
                                     'Visualization/Python')
        self.test_dir = os.path.join(spectre_informer.unit_test_build_path(),
                                     'Visualization/GenerateXdmf')
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_xdmf(self):
        data_file_prefix = os.path.join(self.data_dir, 'VolTestData')
        output_filename = os.path.join(self.test_dir,
                                       'Test_GenerateXdmf_output')
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

    def test_surface_generate_xdmf(self):
        data_file_prefix = os.path.join(self.data_dir, 'SurfaceTestData')
        output_filename = os.path.join(self.test_dir,
                                       'Test_SurfaceGenerateXdmf_output')
        generate_xdmf(file_prefix=data_file_prefix,
                      output=output_filename,
                      subfile_name="AhA",
                      start_time=0.,
                      stop_time=0.03,
                      stride=1,
                      coordinates='InertialCoordinates')

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising an error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + '.xmf'))

    def test_subfile_not_found(self):
        data_file_prefix = os.path.join(self.data_dir, 'VolTestData')
        output_filename = os.path.join(self.test_dir,
                                       'Test_GenerateXdmf_subfile_not_found')
        with self.assertRaisesRegex(ValueError, 'Could not open subfile'):
            generate_xdmf(file_prefix=data_file_prefix,
                          output=output_filename,
                          subfile_name="unknown_subfile",
                          start_time=0.,
                          stop_time=1.,
                          stride=1,
                          coordinates='InertialCoordinates')

    def test_cli(self):
        data_file_prefix = os.path.join(self.data_dir, 'VolTestData')
        output_filename = os.path.join(self.test_dir,
                                       'Test_GenerateXdmf_output')
        runner = CliRunner()
        result = runner.invoke(generate_xdmf_command, [
            "--file-prefix",
            data_file_prefix,
            "-o",
            output_filename,
            "-d",
            "element_data",
        ],
                               catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
