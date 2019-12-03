# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity \
    as spectre_gr_solutions
import spectre.PointwiseFunctions.Hydro.EquationsOfState as spectre_eos

import unittest


class TestTov(unittest.TestCase):
    def test_creation(self):
        eos = spectre_eos.RelativisticPolytropicFluid(
            polytropic_constant=8, polytropic_exponent=2)
        tov = spectre_gr_solutions.Tov(
            equation_of_state=eos, central_mass_density=1e-3)
        # Just making sure we can call the member functions
        outer_radius = tov.outer_radius()
        self.assertAlmostEqual(outer_radius, 3.4685521362)
        self.assertAlmostEqual(tov.mass(outer_radius), 0.0531036941)
        self.assertAlmostEqual(
            tov.mass_over_radius(outer_radius) * outer_radius, 0.0531036941)
        self.assertAlmostEqual(tov.log_specific_enthalpy(outer_radius), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
