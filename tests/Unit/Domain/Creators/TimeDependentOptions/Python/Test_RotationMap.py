# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import RotationMapOptions


class TestRotationMap(unittest.TestCase):
    def test_construction(self):
        rotation_map = RotationMapOptions([[0.0, 0.0, 0.0, 1.0]], 100.0)
        self.assertNotEqual(rotation_map, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
