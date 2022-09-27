// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class UniformCylindricalFlatEndcap.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Map from 3D unit right cylinder to a 3D volume that connects
 *  a portion of a spherical surface with a disk.
 *
 * \image html UniCylFlatEndcap.svg "A cylinder maps to the shaded region."
 *
 * \details Consider a sphere with center \f$C_1\f$ and radius \f$R_1\f$,
 * and a disk in the \f$xy\f$
 * plane with center \f$C_2\f$ and radius \f$R_2\f$.
 * Let sphere 1 be intersected by a plane normal to the \f$z\f$ axis
 * and located at \f$z = z_{\mathrm{P}1}\f$,
 *
 * UniformCylindricalFlatEndcap maps a 3D unit right cylinder (with coordinates
 * \f$(\bar{x},\bar{y},\bar{z})\f$ such that \f$-1\leq\bar{z}\leq 1\f$
 * and \f$\bar{x}^2+\bar{y}^2 \leq 1\f$) to the shaded area
 * in the figure above (with coordinates \f$(x,y,z)\f$).  The "bottom"
 * of the cylinder \f$\bar{z}=-1\f$ is mapped to the portion of sphere
 * 1 that has \f$z \geq z_{\mathrm{P}1}\f$, and on this portion of the
 * sphere the angular coordinate \f$\theta_1 = \acos((z-C_1^z)/R_1)\f$
 * is uniform in \f$\bar{\rho} = \sqrt{\bar{x}^2+\bar{y}^2}\f$ and the angular
 * coordinate \f$\phi_1 = \atan((y-C_1^y)/(x-C_1^x))\f$ is the same as
 * \f$\phi = \atan(\bar{y}/\bar{x})\f$.
 * The "top" of the cylinder
 * \f$\bar{z}=+1\f$ is mapped to the disk,
 * and the radial polar coordinate coordinate \f$\sqrt{x^2+y^2}\f$
 * on this disk is equal to \f$R_2\bar\rho\f$ and the angular
 * coordinate \f$\phi_2 = \atan((y-C_2^y)/(x-C_2^x))\f$ is the same as
 * \f$\phi\f$.
 *
 * UniformCylindricalFlatEndcap is intended to be composed with
 * `Wedge<2>` maps to construct a portion of a cylindrical domain for
 * a binary system.
 *
 * UniformCylindricalFlatEndcap can be used to construct a domain that
 * is similar to, but not identical to, the one described briefly in
 * the Appendix of \cite Buchman:2012dw.  UniformCylindricalFlatEndcap
 * is used to construct the Blocks analogous to those labeled 'MA
 * wedge' and 'MB wedge' in Figure 20 of that paper.
 *
 * UniformCylindricalFlatEndcap provides the following functions:
 *
 * ### operator()
 *
 * `operator()` maps \f$(\bar{x},\bar{y},\bar{z})\f$ to \f$(x,y,z)\f$
 * according to
 *
 * \f{align}
 * x &= C_1^x+\lambda(C_2^x-C_1^x) +
 *        \cos\phi\left(R_1\sin\theta_1 +
 *        \lambda(R_2\bar\rho-R_1\sin\theta_1)\right), \label{eq:x0} \\
 * y &= C_1^y+\lambda(C_2^y-C_1^y) +
 *        \sin\phi\left(R_1\sin\theta_1 +
 *        \lambda(R_2\bar\rho-R_1\sin\theta_1)\right), \label{eq:x1} \\
 * z &= C_1^z+\lambda(C_2^z-C_1^z) +
 *        (1-\lambda)R_1\cos\theta_1 \label{eq:x2}.
 * \f}
 *
 * Here
 * \f{align}
 * \lambda  &= \frac{\bar{z}+1}{2},\label{eq:lambdafromzbar}\\
 * \theta_1 &= \bar{\rho} \theta_{1 \mathrm{max}},\label{eq:deftheta1}\\
 * \phi     &= \atan(\bar{y}/\bar{x})\label{eq:defphi},
 * \f}
 * where \f$\theta_{1 \mathrm{max}}\f$
 * is defined by
 * \f{align}
 *   \cos(\theta_{1\mathrm{max}}) &= (z_{\mathrm{P}1}-C_1^z)/R_1,\\
 * \f}
 * and
 * \f{align}
 * \bar{\rho} &= \sqrt{\bar{x}^2+\bar{y}^2}/\bar{R} \label{eq:defrhobar},
 * \f}
 * where \f$\bar{R}\f$ is the radius of the cylinder in barred
 * coordinates, which is always unity.
 *
 * ### inverse
 *
 * Given \f$(x,y,z)\f$ we want to find \f$(\bar{x},\bar{y},\bar{z})\f$.
 * From Eq. (\f$\ref{eq:x2}\f$) we can write \f$\lambda\f$ as a function
 * of \f$\bar\rho\f$:
 *
 * \f{align}
 * \lambda &= \frac{z - C_1^z - R_1\cos\theta_1}
 *                 {C_2^z-C_1^z - R_1\cos\theta_1}
 *                 \label{eq:lambda_from_rho}.
 * \f}
 *
 * Then by eliminating \f$\phi\f$ from Eqs. (\f$\ref{eq:x0}\f$) and
 * (\f$\ref{eq:x1}\f$) we find that \f$\bar{\rho}\f$ is the solution
 * of \f$Q(\bar{\rho})=0\f$, where
 *
 * \f{align}
 * Q(\bar{\rho}) &= \left(x-C_1^x-\lambda(C_2^x-C_1^x)\right)^2+
 *                  \left(y-C_1^y-\lambda(C_2^y-C_1^y)\right)^2-
 *                  \left((1-\lambda)R_1\sin\theta_1 +
 *                  \lambda \bar\rho R_2\right)^2.\label{eq:defQ}
 * \f}
 * Here \f$\lambda\f$ and \f$\theta_1\f$, are functions
 * of \f$\bar{\rho}\f$ through Eqs. (\f$\ref{eq:lambda_from_rho}\f$)
 * and (\f$\ref{eq:deftheta1}\f$).
 *
 * We solve \f$Q(\bar{\rho})=0\f$ numerically; it is a one-dimensional
 * root-finding problem.
 *
 * Once we have determined \f$\bar{\rho}\f$, we then obtain \f$\lambda\f$
 * from Eq. (\f$\ref{eq:lambda_from_rho}\f$), and we obtain \f$\phi\f$ from
 *
 * \f{align}
 *  \tan\phi &=
 *  \frac{y-C_1^y-\lambda(C_2^y-C_1^y)}{x-C_1^x-\lambda(C_2^x-C_1^x)}.
 * \f}
 *
 * Then \f$\bar{z}\f$ is obtained from Eq. (\f$\ref{eq:lambdafromzbar}\f$)
 * and \f$\bar{x}\f$ and \f$\bar{y}\f$ are obtained from
 *
 * \f{align}
 *   \bar{x} &= \bar{\rho}\bar{R}\cos\phi,\\
 *   \bar{y} &= \bar{\rho}\bar{R}\sin\phi.
 * \f}
 *
 * #### Considerations when root-finding.
 *
 * We solve \f$Q(\bar{\rho})=0\f$ numerically for \f$\bar{\rho}\f$,
 * where \f$Q(\bar{\rho})\f$ is given by Eq. (\f$\ref{eq:defQ}\f$).
 *
 * ##### min/max values of \f$\bar{\rho}\f$:
 *
 * Note that the root we care about must have
 * \f$0\leq\lambda\leq 1\f$; therefore from Eq. (\f$\ref{eq:lambda_from_rho}\f$)
 * we have
 *
 * \f{align}
 *   \bar{\rho}_{\mathrm{min}} &=
 *      \left\{\begin{array}{ll}
 *           0 & \text{for } z-C_1^z \geq R_1, \\
 *           \displaystyle \frac{1}{\theta_{1 \mathrm{max}}}
 *           \cos^{-1}\left(\frac{z-C_1^z}{R_1}\right) & \text{otherwise}
 *      \end{array}\right.\label{eq:rhobarmin}\\
 *   \bar{\rho}_{\mathrm{max}} &= 1. \label{eq:rhobarmax}
 * \f}
 *
 * so we look for a root only between \f$\bar{\rho}_{\mathrm{min}}\f$
 * and \f$\bar{\rho}_{\mathrm{max}}\f$.
 *
 * ##### Roots within roundoff of endpoints:
 *
 * Sometimes a root is within roundoff of \f$\bar{\rho}_{\mathrm{min}}\f$
 * This tends to happen at points on the
 * boundary of the mapped region. In this case, the root might
 * not be bracketed by
 * \f$[\bar{\rho}_{\mathrm{min}},\bar{\rho}_{\mathrm{max}}]\f$ if the root
 * is slightly outside that interval.  If we find that
 * \f$Q(\bar{\rho}_{\mathrm{min}})\f$ is near zero but has the wrong sign,
 * then we slightly expand the interval as follows:
 *
 * \f{align}
 *    \bar{\rho}_{\mathrm{min}} \to \bar{\rho}_{\mathrm{min}}
 *  - 2 \frac{Q(\bar{\rho}_{\mathrm{min}})}{Q'(\bar{\rho}_{\mathrm{min}})},
 * \f}
 *
 * where \f$Q'(\bar{\rho}_{\mathrm{min}})\f$ is the derivative of the function
 * in Eq. (\f$\ref{eq:defQ}\f$). Note that without the factor of 2, this is
 * a Newton-Raphson step; the factor of 2 is there to overcompensate so that
 * the new \f$\bar{\rho}_{\mathrm{min}}\f$ brackets the root.  Sometimes, if
 * the derivative is large enough so that the correction above amounts to
 * a value less than roundoff; in that case, we increase the correction to
 * a value larger than roundoff.
 *
 * Note that by differentiating Eqs. (\f$\ref{eq:defQ}\f$) and
 * (\f$\ref{eq:lambda_from_rho}\f$), one obtains
 *
 * \f{align}
 * Q'(\bar{\rho}) =& -2 \frac{d\lambda}{d\bar{\rho}}\left[
 *      \left(x-C_1^x-\lambda(C_2^x-C_1^x)\right)(C_2^x-C_1^x)+
 *      \left(y-C_1^y-\lambda(C_2^y-C_1^y)\right)(C_2^y-C_1^y)
 *      \right]\nonumber \\
 *     &
 *    -2 \left((1-\lambda)R_1\sin\theta_1 +
 *                \lambda \bar\rho R_2\right)
 *    \left[
 *    \frac{d\lambda}{d\bar{\rho}} (R_2\bar\rho - R_1\sin\theta_1)
 *    +(1-\lambda)R_1\theta_{1 \mathrm{max}}\cos\theta_1
 *    +\lambda R_2
 *    \right], \label{eq:defQderiv}
 * \f}
 *
 * where
 * \f{align}
 *  \frac{d\lambda}{d\bar{\rho}} &=
 *  \frac{(1-\lambda)R_1\theta_{1 \mathrm{max}}\sin\theta_1}
 *       {C_2^z-C_1^z -R_1\cos\theta_1}
 *  \label{eq:dlambda_drhobar}.
 * \f}
 *
 * ##### Roots within roundoff of \f$\bar{\rho}=0\f$ or \f$\bar{\rho}=1\f$:
 *
 * For some points on the boundary of the mapped domain, the root will
 * be within roundoff of \f$\bar{\rho}=0\f$ or \f$\bar{\rho}=1\f$.
 * Here it does not always make sense to expand the range of the map
 * if the root fails (by roundoff) to be bracketed, as is done above.
 * Furthermore, when \f$\bar{\rho}=0\f$ is a root it turns
 * out that both \f$Q(\bar{\rho})=0\f$ and \f$Q'(\bar{\rho})=0\f$ for
 * \f$\bar{\rho}=0\f$, so some root-finders (e.g. Newton-Raphson) have
 * difficulty converging.  Therefore the cases where the root is
 * within roundoff of \f$\bar{\rho}=0\f$ or \f$\bar{\rho}=1\f$ are
 * treated separately.
 *
 * These cases are detected by comparing terms in the first-order
 * power series of \f$Q(\bar{\rho})=0\f$ when expanded about
 * \f$\bar{\rho}=0\f$ or \f$\bar{\rho}=1\f$.  When one of these cases is
 * recognized, the root is returned as either \f$\bar{\rho}=0\f$ or
 * \f$\bar{\rho}=1\f$.
 *
 * #### Quick rejection of points out of range of the map.
 *
 * It is expected that `inverse()` will often be passed points
 * \f$(x,y,z)\f$ that are out of the range of the map; in this case
 * `inverse()` returns a `std::nullopt`. To avoid the difficulty and
 * expense of attempting to solve \f$Q(\bar{\rho})=0\f$ numerically
 * for such points (and then having this solution fail), it is useful
 * to quickly reject points \f$(x,y,z)\f$ that are outside the range
 * of the map.
 *
 * Any point in the range of the map must be below the disk
 * and it must be outside or on sphere 1, so the inverse map
 * can immediately return a `std::nullopt` for a point that does not
 * satisfy these conditions.
 *
 * Likewise, the inverse map can immediately reject any point with
 * \f$z < z_{\mathrm{P}1}\f$.
 *
 * Finally, consider the circle \f$S_1\f$ defining the intersection of sphere 1
 * and the plane \f$z = z_{\mathrm{P}1}\f$; this circle has radius
 * \f$r_1 = R_1 \sin\theta_{1\mathrm{max}}\f$.
 * Now consider the cone that passes through both \f$S_1\f$ and
 * the circle \f$S_2\f$ bounding the upper disk.
 * A point in the range of the map must be inside
 * or on this cone. The cone can be defined parametrically as
 *
 * \f{align}
 * x_c &= C_1^x + \tilde{\lambda}(C_2^x-C_1^x) +
 *      \cos\varphi (r_1 + \tilde{\lambda} (R_2 -r_1)),\\
 * y_c &= C_1^y + \tilde{\lambda}(C_2^y-C_1^y),+
 *      \sin\varphi (r_1 + \tilde{\lambda} (R_2 -r_1)),\\
 * z_c &= C_1^z + R_1 \cos\theta_{1\mathrm{max}} +
 *        \tilde{\lambda}(C_2^z -
 *        C_1^z - R_1 \cos\theta_{1\mathrm{max}}),
 * \f}
 *
 * where \f$(x_c,y_c,z_c)\f$ is a point on the cone, and the two
 * parameters defining a point on the cone are the angle \f$\varphi\f$
 * around the cone and the parameter \f$\tilde{\lambda}\f$, which is
 * defined to be zero on \f$S_1\f$ and unity on \f$S_2\f$.
 *
 * Given an arbitrary point \f$(x, y, z)\f$, we can determine whether
 * or not that point is inside the cone as follows.  First determine
 *
 * \f{align}
 *  \tilde{\lambda} &= \frac{z - C_1^z - R_1 \cos\theta_{1\mathrm{max}}}
 *   {C_2^z -
 *   C_1^z- R_1 \cos\theta_{1\mathrm{max}}}, \\
 *  \tilde{x} &= x - C_1^x - \tilde{\lambda} (C_2^x-C_1^x),\\
 *  \tilde{y} &= y - C_1^y - \tilde{\lambda} (C_2^y-C_1^y).\\
 * \f}
 *
 * Then the condition for the point to be inside or on the cone is
 * \f{align}
 * \sqrt{\tilde{x}^2+\tilde{y}^2} \le r_1 + (R_2-r_1)\tilde{\lambda}.
 * \f}
 *
 * The inverse map can therefore reject any points that do
 * not satisfy this criterion.
 *
 * ### jacobian
 *
 * One can rewrite Eqs.(\f$\ref{eq:x0}\f$) through (\f$\ref{eq:x2}\f$) as
 *
 * \f{align}
 * x &= \frac{1}{2}\left((1-\bar{z})C_1^x+ (1+\bar{z})C_2^x\right) +
 *        \frac{\bar{x}}{2}\left(
 * (1-\bar{z}) R_1 S(\bar{\rho},\theta_{1 \mathrm{max}}) +
 * (1+\bar{z}) R_2
 *     \right), \label{eq:x0alt} \\
 * y &= \frac{1}{2}\left((1-\bar{z})C_1^y + (1+\bar{z})C_2^y\right) +
 *        \frac{\bar{y}}{2}\left(
 * (1-\bar{z})R_1 S(\bar{\rho},\theta_{1 \mathrm{max}}) +
 * (1+\bar{z})R_2
 *     \right), \label{eq:x1alt} \\
 * z &= \frac{1}{2}\left((1-\bar{z})C_1^z + (1+\bar{z})C_2^z\right) +
 * \frac{1}{2} (1-\bar{z})R_1 \cos\theta_1, \label{eq:x2alt} \\
 * \f}
 *
 * where we have used Eq. (\f$\ref{eq:lambdafromzbar}\f$) to eliminate
 * \f$\lambda\f$ in favor of \f$\bar{z}\f$, and where we have defined the
 * function
 *
 * \f{align}
 *   S(\bar{\rho},a) = \frac{\sin(\bar{\rho} a)}{\bar{\rho}}. \label{eq:Sdef}
 * \f}
 *
 * Note that \f$S(\bar{\rho},a)\f$ is finite as \f$\bar{\rho}\f$
 * approaches zero, and in the code we must take care that everything
 * remains well-behaved in that limit.
 *
 * Then differentiating Eqs. (\f$\ref{eq:x0alt}\f$) and (\f$\ref{eq:x1alt}\f$)
 * with respect to \f$\bar{x}\f$ and \f$\bar{y}\f$, taking into account the
 * dependence of \f$\bar{\rho}\f$ on \f$\bar{x}\f$ and \f$\bar{y}\f$ from Eq.
 * (\f$\ref{eq:defrhobar}\f$), we find:
 *
 * \f{align}
 * \frac{\partial x^0}{\partial \bar{x}} &=
 * \frac{1}{2}\left(
 *       (1-\bar{z}) R_1 S(\bar{\rho},\theta_{1 \mathrm{max}}) +
 *       (1+\bar{z}) R_2
 *       \right) +
 * \frac{\bar{x}^2}{2\bar{\rho}}
 *           (1-\bar{z}) R_1 S'(\bar{\rho},\theta_{1 \mathrm{max}}), \\
 * \frac{\partial x^1}{\partial \bar{y}} &=
 * \frac{1}{2}\left(
 *       (1-\bar{z}) R_1 S(\bar{\rho},\theta_{1 \mathrm{max}}) +
 *       (1+\bar{z}) R_2
 *       \right) +
 * \frac{\bar{y}^2}{2\bar{\rho}}
 *           (1-\bar{z}) R_1 S'(\bar{\rho},\theta_{1 \mathrm{max}}), \\
 * \frac{\partial x^0}{\partial \bar{y}} &=
 * \frac{\bar{x}\bar{y}}{2\bar{\rho}}
 *           (1-\bar{z}) R_1 S'(\bar{\rho},\theta_{1 \mathrm{max}}), \\
 * \frac{\partial x^1}{\partial \bar{x}} &=
 * \frac{\partial x^0}{\partial \bar{y}},
 * \f}
 *
 * where \f$S'(\bar{\rho},a)\f$ means the derivative of \f$S(\bar{\rho},a)\f$
 * with respect to \f$\bar\rho\f$.  Note that \f$S'(\bar{\rho},a)/\bar{\rho}\f$
 * approaches a constant value as \f$\bar{\rho}\f$ approaches zero.
 *
 * Differentiating Eq. (\f$\ref{eq:x2alt}\f$) with respect to
 * \f$\bar{x}\f$ and \f$\bar{y}\f$ we find
 *
 * \f{align}
 * \frac{\partial z}{\partial \bar{x}} &=
 * - \frac{\bar{x}}{2}
 *        (1-\bar{z}) R_1 \theta_{1 \mathrm{max}}
 *                S(\bar{\rho},\theta_{1 \mathrm{max}}),\\
 * \frac{\partial z}{\partial \bar{y}} &=
 * - \frac{\bar{y}}{2}
 *        (1-\bar{z}) R_1 \theta_{1 \mathrm{max}}
 *                S(\bar{\rho},\theta_{1 \mathrm{max}}).
 * \f}
 *
 * Differentiating Eqs. (\f$\ref{eq:x0alt}\f$) through (\f$\ref{eq:x2alt}\f$)
 * with respect to \f$\bar{z}\f$ yields
 *
 * \f{align}
 * \frac{\partial x}{\partial \bar{z}} &=
 * \frac{1}{2}\left[
 *  C_2^x-C_1^x +
 *        \bar{x}\left(R_2 -
 *        R_1 S(\bar{\rho},\theta_{1 \mathrm{max}})\right)
 * \right],\\
 * \frac{\partial y}{\partial \bar{z}} &=
 * \frac{1}{2}\left[
 *  C_2^y-C_1^y +
 *        \bar{y}\left(R_2 -
 *        R_1 S(\bar{\rho},\theta_{1 \mathrm{max}})\right)
 * \right],\\
 * \frac{\partial z}{\partial \bar{z}} &=
 * \frac{1}{2}\left(
 *  C_2^z-C_1^z - R_1\cos\theta_1
 * \right).
 * \f}
 *
 * ### inv_jacobian
 *
 * The inverse Jacobian is computed by numerically inverting the
 * Jacobian.
 *
 * ### Restrictions on map parameters
 *
 * We demand that \f$C^2_1 + 1.05 R_1 \leq C^2_2 \leq C^2_1 + 5 R_1\f$.
 * It is possible to construct a valid
 * map without this assumption, but the assumption simplifies the
 * code, and the expected use cases obey this restriction.
 *
 * We also demand that the z plane in the above figure lies
 * above the center of the sphere and is not too close to the center
 * or edge of the sphere; specifically, we demand that
 * - \f$ 0.075\pi < \theta_{1 \mathrm{max}} < 0.35\pi\f$
 *
 * Here 0.075 and 0.35 are safety factors. These restrictions are not
 * strictly necessary but are made for simplicity and to ensure the
 * accuracy of the inverse map (the inverse map becomes less accurate if
 * the map parameters are extreme).
 *
 * Consider the line segment \f$L\f$ that connects a point on the
 * circle \f$S_1\f$ (the circle formed by the intersection of sphere 1
 * and the plane \f$z=z_{\mathrm{P}1}\f$) with the center of the
 * circle \f$S_1\f$.  Consider another line segment \f$L'\f$ that
 * connects the same point on the circle \f$S_1\f$ with the
 * corresponding point on the circle \f$S_2\f$ (the circle bounding
 * the disk with center \f$C_2\f$ and radius \f$R_2\f$).
 * Now consider the angle between \f$L\f$
 * and \f$L'\f$, as measured from the interior of sphere 1, and Let
 * \f$\alpha\f$ be the minimum value of this angle over the circle.
 * \f$\alpha\f$ is shown in the figure above. If
 * \f$\alpha < \theta_{1 \mathrm{max}}\f$, then the line segment \f$L'\f$
 * intersects the mapped portion of sphere 1 twice, so the map is
 * multi-valued. Therefore we demand that the map parameters are such that
 * - \f$\alpha > 1.1 \theta_{1 \mathrm{max}}\f$
 *
 * where 1.1 is a safety factor.
 *
 * The condition on \f$\alpha\f$ is guaranteed to provide an
 * invertible map if \f$C_1^x=C_2^x\f$ and \f$C_1^y=C_2^y\f$.
 * However, for \f$C_1^x \neq C_2^x\f$ or \f$C_1^y\neq C_2^y\f$, even
 * if the \f$\alpha\f$ condition is satisfied, it is possible for two
 * lines of constant \f$(\bar{x},\bar{y})\f$ (each line has different
 * values of \f$(\bar{x},\bar{y})\f$) to pass through the same point
 * \f$(x,y,z)\f$ if those lines are not coplanar.  This condition is
 * difficult to check analytically, so we check it numerically.  We
 * have found empirically that if \f$Q(\bar{\rho})\f$ from
 * Eq. (\f$\ref{eq:defQ}\f$) has only a single root between
 * \f$\bar{\rho}_{\mathrm{min}}\f$ and \f$\bar{\rho}_{\mathrm{max}}\f$
 * for all points \f$(x,y,z)\f$ on the surface of sphere 1 with
 * \f$z\geq z_{\mathrm{P}1}\f$ and with \f$(x-C_1^x)/(y-C_1^y) =
 * (C_1^x-C_2^x)/(C_1^y-C_2^y)\f$, then the map is single-valued
 * everywhere.  We cannot numerically check every point in this
 * one-parameter family of points, but we demand that this condition
 * is satisfied for a reasonably large number of points (currently 1000)
 * in this family.  This check is not very expensive since it is done only
 * once, in the constructor.
 *
 */
class UniformCylindricalFlatEndcap {
 public:
  static constexpr size_t dim = 3;
  UniformCylindricalFlatEndcap(const std::array<double, 3>& center_one,
                               const std::array<double, 3>& center_two,
                               double radius_one, double radius_two,
                               double z_plane_one);
  UniformCylindricalFlatEndcap() = default;
  ~UniformCylindricalFlatEndcap() = default;
  UniformCylindricalFlatEndcap(UniformCylindricalFlatEndcap&&) = default;
  UniformCylindricalFlatEndcap(const UniformCylindricalFlatEndcap&) = default;
  UniformCylindricalFlatEndcap& operator=(const UniformCylindricalFlatEndcap&) =
      default;
  UniformCylindricalFlatEndcap& operator=(UniformCylindricalFlatEndcap&&) =
      default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  static bool is_identity() { return false; }

 private:
  friend bool operator==(const UniformCylindricalFlatEndcap& lhs,
                         const UniformCylindricalFlatEndcap& rhs);
  std::array<double, 3> center_one_{};
  std::array<double, 3> center_two_{};
  double radius_one_{std::numeric_limits<double>::signaling_NaN()};
  double radius_two_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_one_{std::numeric_limits<double>::signaling_NaN()};
  double theta_max_one_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const UniformCylindricalFlatEndcap& lhs,
                const UniformCylindricalFlatEndcap& rhs);

/// Given parameters for UniformCylindricalFlatEndcap, returns whether
/// the map is invertible for target points on sphere_one.
///
/// `is_uniform_cylindrical_flat_endcap_invertible_on_sphere_one` is
/// publicly visible because it is useful for unit tests that need to
/// choose valid parameters to pass into the map.
bool is_uniform_cylindrical_flat_endcap_invertible_on_sphere_one(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two, const double radius_one,
    const double radius_two, const double theta_max_one);

}  // namespace domain::CoordinateMaps
