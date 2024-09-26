# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain import Direction, Side


class TestDirection(unittest.TestCase):
    def test_construction(self):
        direction = Direction[1](dimension=0, side=Side.Lower)
        self.assertEqual(direction, Direction[1].lower_xi())
        self.assertNotEqual(direction, Direction[1].upper_xi())
        direction = Direction[2](dimension=1, side=Side.Upper)
        self.assertEqual(direction, Direction[2].upper_eta())
        self.assertNotEqual(direction, Direction[2].lower_xi())
        direction = Direction[3](dimension=2, side=Side.Upper)
        self.assertEqual(direction, Direction[3].upper_zeta())
        self.assertNotEqual(direction, Direction[3].lower_zeta())

    def test_repr(self):
        self.assertEqual(repr(Direction[1].lower_xi()), "-0")
        self.assertEqual(repr(Direction[2].upper_eta()), "+1")
        self.assertEqual(repr(Direction[3].upper_zeta()), "+2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
