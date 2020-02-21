# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity \
    as spectre_gr_solutions
import spectre.PointwiseFunctions.Hydro.EquationsOfState as spectre_eos

import unittest
import numpy as np
import numpy.testing as npt


class TestTov(unittest.TestCase):
    def test_creation(self):
        eos = spectre_eos.RelativisticPolytropicFluid(polytropic_constant=8,
                                                      polytropic_exponent=2)
        tov = spectre_gr_solutions.Tov(equation_of_state=eos,
                                       central_mass_density=1e-3)
        # Just making sure we can call the member functions
        outer_radius = tov.outer_radius()
        self.assertAlmostEqual(outer_radius, 3.4685521362)
        expected_mass = 0.0531036941
        self.assertAlmostEqual(tov.mass(outer_radius), expected_mass)
        self.assertAlmostEqual(
            tov.mass_over_radius(outer_radius) * outer_radius, expected_mass)
        self.assertAlmostEqual(tov.log_specific_enthalpy(outer_radius), 0.)
        # Test vectorization of member functions
        radii = np.array([0., outer_radius])
        npt.assert_allclose(tov.mass(radii), np.array([0., expected_mass]))
        npt.assert_allclose(
            tov.mass_over_radius(radii) * radii, np.array([0., expected_mass]))
        # Testing `log_specific_enthalpy` only at outer radius because we
        # haven't wrapped any EOS functions yet, so it's not trivial to compute
        # the specific enthalpy at other points
        npt.assert_allclose(
            tov.log_specific_enthalpy(np.array([outer_radius, outer_radius])),
            np.array([0., 0.]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
