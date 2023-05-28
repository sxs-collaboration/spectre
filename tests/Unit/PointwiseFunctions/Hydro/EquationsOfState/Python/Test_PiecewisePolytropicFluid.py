# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import spectre.PointwiseFunctions.Hydro.EquationsOfState as spectre_eos


class TestPiecewisePolytropicFluid(unittest.TestCase):
    def test_creation(self):
        # This class currently exposes no functionality, so we can't test
        # anything but its base class and that it can be instantiated.
        eos = spectre_eos.RelativisticPiecewisePolytropicFluid(
            transition_density=10.0,
            polytropic_constant_lo=5.0,
            polytropic_exponent_lo=1.5,
            polytropic_exponent_hi=2.1,
        )
        self.assertTrue(
            isinstance(eos, spectre_eos.RelativisticEquationOfState1D)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
