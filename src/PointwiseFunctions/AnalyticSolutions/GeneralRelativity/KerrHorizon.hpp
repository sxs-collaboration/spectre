// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace gr {
namespace Solutions {

/*!
 * \brief Radius of Kerr horizon in Kerr-Schild coordinates.
 *
 * \details Computes the radius of a Kerr black hole as a function of
 * angles.  The input argument `theta_phi` is typically the output of
 * the `theta_phi_points()` method of a `YlmSpherepack` object; i.e.,
 * a std::array of two DataVectors containing the values of theta and
 * phi at each point on a Strahlkorper.
 *
 * \note If the spin is nearly extremal, this function has accuracy
 *       limited to roughly \f$10^{-8}\f$, because of roundoff amplification
 *       from computing \f$M + \sqrt{M^2-a^2}\f$.
 *
 * Derivation:
 *
 * Define spherical coordinates \f$(r,\theta,\phi)\f$ in the usual way
 * from the Cartesian Kerr-Schild coordinates \f$(x,y,z)\f$
 * (i.e. \f$x = r \sin\theta \cos\phi\f$ and so on).
 * Then the relationship between \f$r\f$ and the radial
 * Boyer-Lindquist coordinate \f$r_{BL}\f$ is
 * \f[
 * r_{BL}^2 = \frac{1}{2}(r^2 - a^2)
 *     + \left(\frac{1}{4}(r^2-a^2)^2 +
 *             r^2(\vec{a}\cdot \hat{x})^2\right)^{1/2},
 * \f]
 * where \f$\vec{a}\f$ is the Kerr spin vector (with units of mass),
 * \f$\hat{x}\f$ means \f$(x/r,y/r,z/r)\f$, and the dot product is
 * taken as in flat space.
 *
 * The horizon is a surface of constant \f$r_{BL}\f$. Therefore
 * we can solve the above equation for \f$r^2\f$ as a function of angles,
 * yielding
 * \f[
 *     r^2 = \frac{r_{BL}^2 (r_{BL}^2 + a^2)}
                  {r_{BL}^2+(\vec{a}\cdot \hat{x})^2},
 * \f]
 * where the angles are encoded in \f$\hat x\f$ and everything else on the
 * right-hand side is constant.
 *
 * `kerr_horizon_radius` evaluates \f$r\f$ using the above equation, and
 * using the standard expression for the Boyer-Lindquist radius of the
 * Kerr horizon:
 * \f[
 *   r_{BL} = r_+ = M + \sqrt{M^2-a^2}.
 * \f]
 *
 */
template <typename DataType>
Scalar<DataType> kerr_horizon_radius(
    const std::array<DataType, 2>& theta_phi, const double& mass,
    const std::array<double, 3>& dimensionless_spin) noexcept;

}  // namespace Solutions
}  // namespace gr
