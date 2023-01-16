# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.ComputePowerMonitors import compute_modal_coefs

import os
import unittest
import numpy as np
import logging
from spectre.Informer import unit_test_src_path


class TestComputePowerMonitors(unittest.TestCase):
    def test_sample_file(self):
        """
        tests the modal coefficients of a sample file with smooth data.
        The sample data was evolved on a 3D grid with [6, 7, 8] points
        and interpolated onto a grid with [9, 9, 9] points.
        We check that the first 6, 7, or 8 coefficients decrease
        exponentially and that the remaining coefficients are zero.
        """
        test_file_name = os.path.join(unit_test_src_path(),
                                      "Visualization/Python",
                                      "SmoothTestData0.h5")
        test_subfile_name = "/VolumeData"
        obs_key = 7641944464201565954
        modal_data = compute_modal_coefs(test_file_name, test_subfile_name,
                                         "Psi")[obs_key]
        for element_data in modal_data.values():
            x_data = element_data["x"]
            y_data = element_data["y"]
            z_data = element_data["z"]
            # check that the coefficients decrease exponentially
            for i in range(5):
                self.assertTrue(x_data[i + 1] < x_data[i] / 3.)
            for i in range(6):
                self.assertTrue(y_data[i + 1] < y_data[i] / 3.)
            for i in range(7):
                self.assertTrue(z_data[i + 1] < z_data[i] / 3.)
            # check that the last, interpolated coefficients are 0
            self.assertTrue(np.allclose(x_data[6:], np.zeros(3), atol=1e-15))
            self.assertTrue(np.allclose(x_data[7:], np.zeros(2), atol=1e-15))
            self.assertTrue(np.allclose(x_data[8:], np.zeros(1), atol=1e-15))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
