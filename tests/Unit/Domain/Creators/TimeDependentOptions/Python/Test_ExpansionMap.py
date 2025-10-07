# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import ExpansionMapOptions


class TestExpansionMap(unittest.TestCase):
    def test_construction(self):
        expansion_map = ExpansionMapOptions([1.0, 1e-4, 0.0], 100.0, 1e-6)
        self.assertNotEqual(expansion_map, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
