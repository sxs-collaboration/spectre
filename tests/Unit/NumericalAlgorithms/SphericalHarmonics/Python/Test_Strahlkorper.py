# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.SphericalHarmonics import Strahlkorper, cartesian_coords


class TestStrahlkorper(unittest.TestCase):
    def test_strahlkorper(self):
        strahlkorper = Strahlkorper(
            l_max=12, m_max=12, radius=1.0, center=[0.0, 0.0, 0.0]
        )
        self.assertEqual(strahlkorper.l_max, 12)
        self.assertEqual(strahlkorper.m_max, 12)
        self.assertEqual(strahlkorper.physical_extents, [13, 25])
        self.assertEqual(strahlkorper.expansion_center, [0.0, 0.0, 0.0])
        self.assertEqual(strahlkorper.physical_center, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(strahlkorper.average_radius, 1.0)
        self.assertAlmostEqual(strahlkorper.radius(0.0, 0.0), 1.0)
        self.assertTrue(strahlkorper.point_is_contained([0.5, 0.0, 0.0]))
        x = np.array(cartesian_coords(strahlkorper))
        r = np.linalg.norm(x, axis=0)
        npt.assert_allclose(r, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
