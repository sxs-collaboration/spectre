# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import SkewMapOptions


class TestSkewMap(unittest.TestCase):
    def test_construction(self):
        skew_map = SkewMapOptions([0.0, 40.0, 50.0], [0.0, 10.0, 0.0])
        self.assertNotEqual(skew_map, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
