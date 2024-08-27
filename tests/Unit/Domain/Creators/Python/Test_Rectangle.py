# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators import DomainCreator2D, Rectangle


class TestRectangle(unittest.TestCase):
    def test_construction(self):
        rectangle = Rectangle(
            lower_bounds=[1.0, 0.0],
            upper_bounds=[2.0, 1.0],
            is_periodic=[False, False],
            initial_refinement_levels=[1, 0],
            initial_num_points=[3, 4],
        )
        self.assertIsInstance(rectangle, DomainCreator2D)


if __name__ == "__main__":
    unittest.main(verbosity=2)
