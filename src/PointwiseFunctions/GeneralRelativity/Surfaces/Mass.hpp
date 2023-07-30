// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace StrahlkorperGr {
/*!
 * \ingroup SurfacesGroup
 * \brief Irreducible mass of a 2D `Strahlkorper`.
 *
 * \details See Eqs. (15.38) \cite Hartle2003gravity. This function computes the
 * irreducible mass from the area of a horizon. Specifically, computes
 * \f$M_\mathrm{irr}=\sqrt{\frac{A}{16\pi}}\f$.
 */
double irreducible_mass(double area);

/*!
 * \ingroup SurfacesGroup
 * \brief Christodoulou Mass of a 2D `Strahlkorper`.
 *
 * \details See e.g. Eq. (1) of \cite Lovelace2016uwp.
 * This function computes the Christodoulou mass from the dimensionful
 * spin angular momentum \f$S\f$ and the irreducible mass \f$M_{irr}\f$
 * of a black hole horizon. Specifically, computes
 *\f$M=\sqrt{M_{irr}^2+\frac{S^2}{4M_{irr}^2}}\f$
 */
double christodoulou_mass(double dimensionful_spin_magnitude,
                          double irreducible_mass);
}  // namespace StrahlkorperGr
