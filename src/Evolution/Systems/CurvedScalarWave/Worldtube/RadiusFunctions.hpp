// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace CurvedScalarWave::Worldtube {

namespace detail {

void check_alpha(double alpha);

void check_delta(double delta);
}  // namespace detail

/*!
 * \brief A smoothly broken power law that falls off to a constant value
 * for larger radii.
 *
 * \details The function is given by
 *
 * \begin{equation}
 * f(r) = A \left(\frac{r}{r_b}\right)^{\alpha}\left[ \frac{1}{2} \left( 1 +
 * \left(\frac{r}{r_b}\right)^{1 / \Delta}\right)\right]^{-\alpha\Delta}.
 * \end{equation}
 *
 * For radii $r \ll r_b$, the function obeys the power law $f(r) \propto
 * r^{\alpha}$. For radii $r \gg r_b$, the function is constant. The parameter
 * $\Delta$ determines the width of the transition region with a larger value of
 * $\Delta$ leading to a more gradual transition.
 *
 * This function is used to control the worldtube radius for more eccentric
 * orbits so the radius does not grow too large during the apoapsis passage
 * as this does not lead to performance gains and can cause problems with the
 * domain.
 */
double smooth_broken_power_law(double orbit_radius, double alpha, double amp,
                               double rb, double delta);
/*!
 * \brief Returns the analytical derivative of `smooth_broken_power_law`.
 */
double smooth_broken_power_law_derivative(double orbit_radius, double alpha,
                                          double amp, double rb, double delta);
}  // namespace CurvedScalarWave::Worldtube
