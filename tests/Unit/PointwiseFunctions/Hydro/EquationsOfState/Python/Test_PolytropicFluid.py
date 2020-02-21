# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.PointwiseFunctions.Hydro.EquationsOfState as spectre_eos

import unittest


class TestPolytropicFluid(unittest.TestCase):
    def test_creation(self):
        # This class currently exposes no functionality, so we can't test
        # anything but its base class and that it can be instantiated.
        eos = spectre_eos.RelativisticPolytropicFluid(polytropic_constant=100.,
                                                      polytropic_exponent=2.)
        self.assertTrue(
            isinstance(eos, spectre_eos.RelativisticEquationOfState1D))


if __name__ == '__main__':
    unittest.main(verbosity=2)
