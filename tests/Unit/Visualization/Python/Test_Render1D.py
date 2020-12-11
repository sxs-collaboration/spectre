#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.Visualization.Render1D import (find_extrema_over_data_set,
                                            render_single_time)

import unittest
import os
import numpy as np
import matplotlib as mpl
mpl.use('agg')


class TestRender1D(unittest.TestCase):
    def test_find_extrema_over_data_set(self):
        test_array = np.array([1.1, 6.45, 0.34, 2.3])
        expected_vals = (0.34, 6.45)
        self.assertEqual(find_extrema_over_data_set(test_array), expected_vals)

    def test_render_single_time(self):
        var_name = "Variable Test"
        time_slice = 1
        output_prefix = "TestRenderSingleTime"
        time = [0.0, 0.1]
        coords = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        data = [[5.2, 4.5, 9.0, 2.0, 8.0], [1.1, 4.0, 6.0, 5.3, 3.0]]
        # test whether a pdf file is saved when run
        render_single_time(var_name, time_slice, output_prefix, time, coords,
                           data)
        self.assertTrue(os.path.isfile(output_prefix + '.pdf'))
        os.remove(output_prefix + '.pdf')


if __name__ == '__main__':
    unittest.main(verbosity=2)
