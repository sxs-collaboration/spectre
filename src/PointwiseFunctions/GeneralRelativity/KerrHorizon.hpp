// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace gr::Solutions {

/*!
 * \brief The Kerr-Schild radius corresponding to a Boyer-Lindquist radius.
 *
 * \details Computes the radius of a surface of constant Boyer-Lindquist radius
 * as a function of angles.  The input argument `theta_phi` is typically the
 * output of the `theta_phi_points()` method of a `ylm::Spherepack` object;
 * i.e., a std::array of two DataVectors containing the values of theta and
 * phi at each point on a Strahlkorper.
 *
 *
 * Derivation:
 *
 * Define spherical coordinates $(r,\theta,\phi)$ in the usual way
 * from the Cartesian Kerr-Schild coordinates $(x,y,z)$
 * (i.e. $x = r \sin\theta \cos\phi$ and so on).
 * Then the relationship between $r$ and the radial
 * Boyer-Lindquist coordinate $r_{BL}$ is
 * \begin{equation}
 * r_{BL}^2 = \frac{1}{2}(r^2 - a^2)
 *     + \left(\frac{1}{4}(r^2-a^2)^2 +
 *             r^2(\vec{a}\cdot \hat{x})^2\right)^{1/2},
 * \end{equation}
 * where $\vec{a}$ is the Kerr spin vector (with units of mass),
 * $\hat{x}$ means $(x/r,y/r,z/r)$, and the dot product is
 * taken as in flat space.
 *
 * We solve the above equation for $r^2$ as a function of angles,
 * yielding
 * \begin{equation}
 *     r^2 = \frac{r_{BL}^2 (r_{BL}^2 + a^2)}
 *                {r_{BL}^2+(\vec{a}\cdot \hat{x})^2},
 * \end{equation}
 * where the angles are encoded in $\hat x$ and everything else on the
 * right-hand side is constant.
 *
 * Boosted black hole:
 *
 * \warning The spin is assumed to be given in the rest frame of the black
 * hole. Only the angles and the returned radius refer to the boosted frame.
 * The spin measured in the boosted frame differs, because the spin transforms
 * as an axial vector (see Eq. (68) of \cite Pfeiffer:2002xz).
 *
 * Given a `boost_velocity` $\vec{v}$ with Lorentz factor $\gamma$, the
 * angles and the returned radius are in the boosted Kerr-Schild coordinates at
 * $t=0$. Decompose $\hat{x}$ into components parallel and orthogonal
 * to the boost, $\hat{x}_\parallel = (\hat{x}\cdot\hat{v})\hat{v}$ and
 * $\hat{x}_\perp = \hat{x} - \hat{x}_\parallel$. The Lorentz
 * transformation at $t=0$ stretches only the parallel component, so the
 * rest-frame position is $r\vec{u}$ with
 * $\vec{u} = \gamma\hat{x}_\parallel + \hat{x}_\perp$. Applying the
 * rest-frame relation above along $\hat{u} = \vec{u}/|\vec{u}|$ to the
 * rest-frame radius $r|\vec{u}|$, and using
 * $\gamma^2 - 1 = \gamma^2 v^2$, simplifies to
 * \begin{equation}
 *     r^2 = \frac{r_{BL}^2 (r_{BL}^2 + a^2)}
 *                {r_{BL}^2 |\vec{u}|^2 + (\vec{a}\cdot\vec{u})^2},
 * \end{equation}
 * where
 * \begin{align}
 *     |\vec{u}|^2 &= 1 + \gamma^2 (\hat{x}\cdot\vec{v})^2, \\
 *     \vec{a}\cdot\vec{u} &= \vec{a}\cdot\hat{x}
 *       + \frac{\gamma^2}{1+\gamma}(\hat{x}\cdot\vec{v})(\vec{a}\cdot\vec{v}).
 * \end{align}
 * For $\vec{v}=0$ this reduces to the unboosted expression.
 */
template <typename DataType>
Scalar<DataType> kerr_schild_radius_from_boyer_lindquist(
    double boyer_lindquist_radius,
    const std::array<DataType, 2>& theta_phi, double mass,
    const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity = std::array<double, 3>{
        {0.0, 0.0, 0.0}});

/*!
 * \brief The Kerr-Schild radius corresponding to a Kerr horizon.
 *
 * \details `kerr_horizon_radius` evaluates $r$ using the above equation in
 * the documentation for `kerr_schild_radius_from_boyer_lindquist`, and
 * using the standard expression for the Boyer-Lindquist radius of the
 * Kerr horizon:
 * \begin{equation}
 *   r_{BL} = r_+ = M + \sqrt{M^2-a^2}.
 * \end{equation}
 *
 * \note If the spin is nearly extremal, this function has accuracy
 *       limited to roughly $10^{-8}$, because of roundoff amplification
 *       from computing $M + \sqrt{M^2-a^2}$.
 */
template <typename DataType>
Scalar<DataType> kerr_horizon_radius(
    const std::array<DataType, 2>& theta_phi, double mass,
    const std::array<double, 3>& dimensionless_spin);

}  // namespace gr::Solutions
