# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.IO.H5.ExtendConnectivityData import (
    extend_connectivity_data_command,
)


class TestExtendConnectivity(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "IO/H5/Python/ExtendConnectivityData"
        )

        self.filename = os.path.join(self.test_dir, "TestVolume.h5")
        os.makedirs(self.test_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(
                unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
            ),
            self.filename,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # Helper function to obtain the length of the connectivity in
    # the volume data file.
    def get_connectivity_length(self):
        with spectre_h5.H5File(self.filename, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            h5_connectivity = volfile.get_tensor_component(
                obs_id, "connectivity"
            ).data
        return len(h5_connectivity)

    def test_cli(self):
        runner = CliRunner()
        initial_connectivity = self.get_connectivity_length()
        result = runner.invoke(
            extend_connectivity_data_command,
            [self.filename, "-d", "element_data.vol"],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        final_connectivity = self.get_connectivity_length()

        assert initial_connectivity <= final_connectivity


if __name__ == "__main__":
    unittest.main(verbosity=2)
