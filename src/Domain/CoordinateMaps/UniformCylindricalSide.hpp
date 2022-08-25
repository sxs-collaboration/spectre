// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class UniformCylindricalSide.

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
 * \brief Map from 3D unit right cylindrical shell to a volume that connects
 *  portions of two spherical surfaces.
 *
 * \image html UniformCylSide.svg "A hollow cylinder maps to the shaded region."
 *
 * \details Consider two spheres with centers \f$C_1\f$ and \f$C_2\f$,
 * and radii \f$R_1\f$ and \f$R_2\f$. Sphere 1 is assumed to be contained
 * inside Sphere 2.
 * Let sphere 1 be intersected by two
 * planes normal to the \f$z\f$ axis and located at
 * \f$z = z^{\pm}_{\mathrm{P}1}\f$,
 * and let sphere 2 be intersected by two planes normal to the \f$z\f$ axis and
 * located at \f$z = z^{\pm}_{\mathrm{P}2}\f$.  Here we assume that
 * \f$z^{-}_{\mathrm{P}2} \leq z^{-}_{\mathrm{P}1}<
 * z^{+}_{\mathrm{P}1} \leq z^{+}_{\mathrm{P}2}\f$.
 *
 * UniformCylindricalSide maps a 3D unit right cylindrical shell (with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$1 \leq \bar{x}^2+\bar{y}^2 \leq 4\f$, where
 * the values of 1 and 2 for the inner and outer cylindrical radii
 * are arbitrary choices but are required by UniformCylindricalSide)
 * to the shaded area in the figure above (with coordinates
 * \f$(x,y,z)\f$).  The "inner surface" of the cylindrical shell
 * \f$\bar{x}^2+\bar{y}^2=1\f$ is mapped to the portion of sphere 1
 * that has \f$z^{-}_{\mathrm{P}1} \leq z \leq z^{+}_{\mathrm{P}1} \f$,
 * and on this portion of the sphere the cosine of the polar angular coordinate
 * \f$\cos\theta_1 =(z-C_1^z)/R_1\f$ is uniform in \f$\bar{z}\f$,
 * and the angular coordinate \f$\phi_1 = \atan((y-C_1^y)/(x-C_1^x))\f$
 * is the same as \f$\phi = \atan(\bar{y}/\bar{x})\f$.
 * Likewise, the "outer surface" of the cylindrical shell
 * \f$\bar{x}^2+\bar{y}^2=4\f$ is mapped to the portion of sphere 2
 * that has \f$z^{-}_{\mathrm{P}2} \leq z \leq z^{+}_{\mathrm{P}2}
 * \f$, and on this portion of the sphere the cosine of the azimuthal
 * angular coordinate
 * \f$\cos\theta_2 = (z-C_2^z)/R_2\f$ is uniform in \f$\bar{z}\f$,
 * and the angular coordinate \f$\phi_2 =
 * \atan((y-C_2^y)/(x-C_2^x))\f$ is the same as \f$\phi\f$.
 *
 * UniformCylindricalSide is different from CylindricalSide
 * because of the distribution of points on the spheres, and because
 * for UniformCylindricalSide the mapped portion of both Sphere 1
 * and Sphere 2 are bounded by planes of constant \f$z\f$, whereas for
 * CylindricalSide only one of the mapped portions is bounded by a
 * plane (except for specially chosen map parameters).  Note that
 * UniformCylindricalSide can be used to construct maps that connect
 * an arbitrary number of nested spheres; this is not possible for
 * CylindricalSide for more than 3 nested spheres because of this
 * asymmetry between CylindricalSide's two spherical surfaces.
 *
 * Note that the entire region between Sphere 1 and Sphere 2 can be covered
 * by a single cylindrical shell (mapped using UniformCylindricalSide) and
 * two cylinders (each mapped by UniformCylindricalEndcap).
 *
 * UniformCylindricalSide is intended to be composed with `Wedge<2>` maps to
 * construct a portion of a cylindrical domain for a binary system.
 *
 * UniformCylindricalSide can be used to construct a domain that is similar
 * to, but not identical to, the one described briefly in the Appendix of
 * \cite Buchman:2012dw.
 * UniformCylindricalSide is used to construct the Blocks analogous to
 * those labeled 'CA cylinder', 'EA cylinder', 'CB cylinder', 'EE cylinder',
 * and 'EB cylinder' in Figure 20 of that paper.
 *
 * UniformCylindricalSide provides the following functions:
 *
 * ## operator()
 *
 * `operator()` maps \f$(\bar{x},\bar{y},\bar{z})\f$ to \f$(x,y,z)\f$
 * according to
 *
 * \f{align}
 * x &= C_1^x+\lambda(C_2^x-C_1^x) +
 *        \cos\phi\left(R_1\sin\theta_1 +
 *        \lambda(R_2\sin\theta_2-R_1\sin\theta_1)\right), \label{eq:x0} \\
 * y &= C_1^y+\lambda(C_2^y-C_1^y) +
 *        \sin\phi\left(R_1\sin\theta_1 +
 *        \lambda(R_2\sin\theta_2-R_1\sin\theta_1)\right), \label{eq:x1} \\
 * z &= C_1^z+\lambda(C_2^z-C_1^z) +
 *        R_1\cos\theta_1 +
 *        \lambda(R_2\cos\theta_2-R_1\cos\theta_1) \label{eq:x2}.
 * \f}
 *
 * Here
 * \f{align}
 * \lambda  &= \bar{\rho}-1,\label{eq:lambdafromrhobar}\\
 * \cos\theta_1 &= \cos\theta_{1 \mathrm{max}} +
 *         \left(\cos\theta_{1 \mathrm{min}}-\cos\theta_{1 \mathrm{max}}\right)
 *             \frac{\bar{z}+1}{2}\label{eq:deftheta1}\\
 * \cos\theta_2 &= \cos\theta_{2 \mathrm{max}} +
 *             \left(\cos\theta_{2 \mathrm{min}}-
 *                   \cos\theta_{2 \mathrm{max}}\right)
 *             \frac{\bar{z}+1}{2}\label{eq:deftheta2}\\
 * \phi     &= \atan(\bar{y}/\bar{x})\label{eq:defphi},
 * \f}
 * where \f$\theta_{1 \mathrm{min}}\f$, \f$\theta_{2 \mathrm{min}}\f$,
 * \f$\theta_{1 \mathrm{max}}\f$, and \f$\theta_{2 \mathrm{max}}\f$
 * are defined by
 * \f{align}
 *   \label{eq:deftheta1min}
 *   \cos(\theta_{1\mathrm{min}}) &= (z^{+}_{\mathrm{P}1}-C_1^z)/R_1,\\
 *   \cos(\theta_{1\mathrm{max}}) &= (z^{-}_{\mathrm{P}1}-C_1^z)/R_1,\\
 *   \cos(\theta_{2\mathrm{min}}) &= (z^{+}_{\mathrm{P}2}-C_2^z)/R_2,\\
 *   \label{eq:deftheta2max}
 *   \cos(\theta_{2\mathrm{max}}) &= (z^{-}_{\mathrm{P}2}-C_2^z)/R_2,
 * \f}
 * and
 * \f{align}
 * \bar{\rho} &= \sqrt{\bar{x}^2+\bar{y}^2}/\bar{R} \label{eq:defrhobar},
 * \f}
 * where \f$\bar{R}\f$ is the inner radius of the cylindrical shell in barred
 * coordinates, which is always unity.
 *
 * Note that \f$\theta_{1\mathrm{min}}<\theta_{1\mathrm{max}}\f$ but
 * \f$\cos\theta_{1\mathrm{min}}>\cos\theta_{1\mathrm{max}}\f$ (and same
 * for sphere 2).
 *
 * Also note that Eqs. (\f$\ref{eq:deftheta1}\f$) and
 * (\f$\ref{eq:deftheta2}\f$) can be simplified using Eqs.
 * (\f$\ref{eq:deftheta1min}\f$-\f$\ref{eq:deftheta2max}\f$):
 * \f{align}
 * R_1\cos\theta_1 &= z^{-}_{\mathrm{P}1}-C_1^z
 *        +(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})
 *             \frac{\bar{z}+1}{2}\label{eq:deftheta1alt}\\
 * R_2\cos\theta_2 &= z^{-}_{\mathrm{P}2}-C_2^z
 *        +(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2})
 *             \frac{\bar{z}+1}{2}\label{eq:deftheta2alt}\\
 * \f}
 *
 * ## inverse
 *
 * Given \f$(x,y,z)\f$ we want to find \f$(\bar{x},\bar{y},\bar{z})\f$.
 * From Eqs. (\f$\ref{eq:x2}\f$), (\f$\ref{eq:deftheta1alt}\f$), and
 * (\f$\ref{eq:deftheta2alt}\f$) we can write \f$\bar{z}\f$ as a function
 * of \f$\lambda\f$:
 *
 * \f{align}
 * \frac{1+\bar{z}}{2} &=
 *   \frac{z +
 *     \lambda (z^{-}_{\mathrm{P}1}-z^{-}_{\mathrm{P}2}) - z^{-}_{\mathrm{P}1}}
 *   {(1-\lambda)(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})
 *    + \lambda(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2})}
 *                 \label{eq:zbar_from_lambda},
 * \f}
 * Note that the denominator of
 * Eq. (\f$\ref{eq:zbar_from_lambda}\f$) is always positive because
 * \f$0\leq\lambda\leq 1\f$, \f$z^{+}_{\mathrm{P}1}>z^{-}_{\mathrm{P}1}\f$,
 * and \f$z^{+}_{\mathrm{P}2}>z^{-}_{\mathrm{P}2}\f$.
 *
 * By eliminating \f$\phi\f$ from Eqs. (\f$\ref{eq:x0}\f$) and
 * (\f$\ref{eq:x1}\f$) we find that \f$\lambda\f$ is the solution
 * of \f$Q(\lambda)=0\f$, where
 *
 * \f{align}
 * Q(\lambda) &= \left(x-C_1^x-\lambda(C_2^x-C_1^x)\right)^2+
 *               \left(y-C_1^y-\lambda(C_2^y-C_1^y)\right)^2-
 *               \left((1-\lambda)R_1\sin\theta_1 +
 *               \lambda R_2\sin\theta_2\right)^2.\label{eq:defQ}
 * \f}
 * Here \f$\theta_1\f$ and \f$\theta_2\f$ are functions
 * of \f$\bar{z}\f$ through Eqs. (\f$\ref{eq:deftheta1alt}\f$) and
 * (\f$\ref{eq:deftheta2alt}\f$), and \f$\bar{z}\f$ is a function of
 * \f$\lambda\f$ through Eq. (\f$\ref{eq:zbar_from_lambda}\f$).
 *
 * We solve \f$Q(\lambda)=0\f$ numerically; it is a one-dimensional
 * root-finding problem.
 *
 * Once we have determined \f$\lambda\f$, we then obtain \f$\bar{z}\f$
 * from Eq. (\f$\ref{eq:zbar_from_lambda}\f$), and we obtain \f$\phi\f$ from
 *
 * \f{align}
 *  \tan\phi &=
 *  \frac{y-C_1^y-\lambda(C_2^y-C_1^y)}{x-C_1^x-\lambda(C_2^x-C_1^x)}.
 * \f}
 *
 * Then \f$\bar{\rho}\f$ is obtained from Eq. (\f$\ref{eq:lambdafromrhobar}\f$)
 * and \f$\bar{x}\f$ and \f$\bar{y}\f$ are obtained from
 *
 * \f{align}
 *   \bar{x} &= \bar{\rho}\bar{R}\cos\phi,\\
 *   \bar{y} &= \bar{\rho}\bar{R}\sin\phi.
 * \f}
 *
 * ### Considerations when root-finding.
 *
 * We solve \f$Q(\lambda)=0\f$ numerically for \f$\lambda\f$,
 * where \f$Q(\lambda)\f$ is given by Eq. (\f$\ref{eq:defQ}\f$).
 *
 * ##### min/max values of \f$\lambda\f$:
 *
 * Note that the root we care about must have \f$-1\leq\bar{z}\leq 1\f$;
 * therefore from Eq. (\f$\ref{eq:zbar_from_lambda}\f$) we have
 *
 * \f{align}
 *   \lambda_{\mathrm{min}} &=
 *   \mathrm{max}\left\{0,
 *              \frac{z-z^{+}_{\mathrm{P}1}}
 *                 {(z^{+}_{\mathrm{P}2}-z^{+}_{\mathrm{P}1})},
 *              \frac{z^{-}_{\mathrm{P}1}-z}
 *                 {(z^{-}_{\mathrm{P}1}-z^{-}_{\mathrm{P}2})}
 *      \right\}\label{eq:lambdamin}
 * \f}
 * In the case where \f$z^{+}_{\mathrm{P}2}=z^{+}_{\mathrm{P}1}\f$
 * we treat the middle term in Eq.(\f$\ref{eq:lambdamin}\f$) as zero since
 * in that case \f$z-z^{+}_{\mathrm{P}1}\f$ can never be positive for
 * \f$x^2\f$ in the range of the map, and for
 * \f$z=z^{+}_{\mathrm{P}2}=z^{+}_{\mathrm{P}1}\f$
 * it turns out that
 * (\f$\ref{eq:zbar_from_lambda}\f$) places no restriction on
 * \f$\lambda_{\mathrm{min}}\f$.  For the same reason, if
 * \f$z^{-}_{\mathrm{P}2}=z^{-}_{\mathrm{P}1}\f$ we treat
 * the last term in Eq.(\f$\ref{eq:lambdamin}\f$) as zero.
 *
 * We look for a root only between \f$\lambda_{\mathrm{min}}\f$
 * and \f$\lambda_{\mathrm{max}}=1\f$.
 *
 * ##### Roots within roundoff of min or max \f$\lambda\f$
 *
 * Sometimes a root is within roundoff of \f$\lambda_{\mathrm{min}}\f$.
 * In this case, the root might not be bracketed by
 * \f$[\lambda_{\mathrm{min}},\lambda_{\mathrm{max}}]\f$ if the root
 * is slightly outside that interval by roundoff error.  If we find that
 * \f$Q(\lambda_{\mathrm{min}})\f$ is near zero but has the wrong sign,
 * then we slightly expand the interval as follows:
 *
 * \f{align}
 *    \lambda_{\mathrm{min}} \to \lambda_{\mathrm{min}}
 *  - 2 \frac{Q(\lambda_{\mathrm{min}})}{Q'(\lambda_{\mathrm{min}})},
 * \f}
 *
 * where \f$Q'(\lambda_{\mathrm{min}})\f$ is the derivative of the function
 * in Eq. (\f$\ref{eq:defQ}\f$). Note that without the factor of 2, this is
 * a Newton-Raphson step; the factor of 2 is there to overcompensate so that
 * the new \f$\lambda_{\mathrm{min}}\f$ brackets the root.
 *
 * Note that by differentiating Eqs. (\f$\ref{eq:defQ}\f$) and
 * (\f$\ref{eq:zbar_from_lambda}\f$), one obtains
 *
 * \f{align}
 * Q'(\lambda) =& -2 \left[
 *      \left(x-C_1^x-\lambda(C_2^x-C_1^x)\right)(C_2^x-C_1^x)+
 *      \left(y-C_1^y-\lambda(C_2^y-C_1^y)\right)(C_2^y-C_1^y)
 *      \right]\nonumber \\
 *     &
 *    -\left[
 *    2(R_2\sin\theta_2-R_1\sin\theta_1)
 *    -(1-\lambda)\cot\theta_1 (z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})
 *     \frac{d\bar{z}}{d\lambda} \right. \nonumber \\
 *     & \left.\qquad
 *    -\lambda \cot\theta_2 (z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2})
 *     \frac{d\bar{z}}{d\lambda}
 *    \right]
 *    \left((1-\lambda)R_1\sin\theta_1 +
 *                \lambda R_2\sin\theta_2\right), \label{eq:defQderiv}
 * \f}
 *
 * where
 * \f{align}
 *  \frac{d\bar{z}}{d\lambda} &=
 *  \frac{(1-\bar{z})(z^{-}_{\mathrm{P}1}-z^{-}_{\mathrm{P}2})
 *       -(1+\bar{z})(z^{+}_{\mathrm{P}2}-z^{+}_{\mathrm{P}1})}
 *        {(1-\lambda)(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})
 *       + \lambda(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2})}
 *  \label{eq:dzbar_dlambda}.
 * \f}
 *
 * A root within roundoff of \f$\lambda_{\mathrm{max}}\f$ is treated
 * similarly.
 *
 * #### Special cases:
 *
 * For some points on the boundary of the mapped domain,
 * \f$\lambda_{\mathrm{min}}\f$ will be within roundoff of
 * \f$\lambda=1\f$. We check explicitly for this case, and we
 * compute the root as exactly \f$\lambda=1\f$.
 *
 * ### Quick rejection of points out of range of the map.
 *
 * It is expected that `inverse()` will often be passed points
 * \f$(x,y,z)\f$ that are out of the range of the map; in this case
 * `inverse()` returns a `std::nullopt`. To avoid the difficulty and
 * expense of attempting to solve \f$Q(\lambda)=0\f$ numerically
 * for such points (and then having this solution fail), it is useful
 * to quickly reject points \f$(x,y,z)\f$ that are outside the range
 * of the map.
 *
 * Any point in the range of the map must be inside or on
 * sphere 2, and it must be outside or on sphere 1, so the inverse map
 * can immediately return a default-constructed
 * `std::optional<std::array<double, 3>>` for a point that does not
 * satisfy these conditions.
 *
 * Likewise, the inverse map can immediately reject any point with
 * \f$z < z^{-}_{\mathrm{P}2}\f$ or \f$z > z^{+}_{\mathrm{P}2}\f$.
 *
 * Finally, for \f$z^{+}_{\mathrm{P}2}\neq z^{+}_{\mathrm{P}1}\f$,
 * consider the circle \f$S^{+}_1\f$
 * defining the intersection of sphere 1
 * and the plane \f$z = z^{+}_{\mathrm{P}1}\f$; this circle has radius
 * \f$r_1 = R_1 \sin\theta_{1\mathrm{min}}\f$.  Similarly, the circle
 * \f$S^{+}_2\f$ defining the intersection of sphere 2 and the plane \f$z =
 * z^{+}_{\mathrm{P}2}\f$ has radius \f$r_2 = R_2
 * \sin\theta_{2\mathrm{min}}\f$.  Now consider the cone that passes
 * through these two circles.  A point in the range of the map must be outside
 * (where "outside" means farther from the \f$z\f$ axis) or on this cone.
 * The cone can be defined parametrically as
 *
 * \f{align}
 * x_c &= C_1^x + \tilde{\lambda}(C_2^x-C_1^x) +
 *      \cos\varphi (r_1 + \tilde{\lambda} (r_2 -r_1)),\\
 * y_c &= C_1^y + \tilde{\lambda}(C_2^y-C_1^y),+
 *      \sin\varphi (r_1 + \tilde{\lambda} (r_2 -r_1)),\\
 * z_c &= z^{+}_{\mathrm{P}1} +
 *        \tilde{\lambda}(z^{+}_{\mathrm{P}2}-z^{+}_{\mathrm{P}1}),
 * \f}
 *
 * where \f$(x_c,y_c,z_c)\f$ is a point on the cone, and the two
 * parameters defining a point on the cone are the angle \f$\varphi\f$
 * around the cone and the parameter \f$\tilde{\lambda}\f$, which is
 * defined to be zero on \f$S^{+}_1\f$ and unity on \f$S^{+}_2\f$.
 *
 * Given an arbitrary point \f$(x, y, z)\f$, we can determine whether
 * or not that point is inside the cone as follows.  First determine
 *
 * \f{align}
 *  \tilde{\lambda} &= \frac{z - z^{+}_{\mathrm{P}1}}
 *   {z^{+}_{\mathrm{P}2}-z^{+}_{\mathrm{P}1}}, \\
 *  \tilde{x} &= x - C_1^x - \tilde{\lambda} (C_2^x-C_1^x),\\
 *  \tilde{y} &= y - C_1^y - \tilde{\lambda} (C_2^y-C_1^y).\\
 * \f}
 *
 * Then the condition for the point to be outside or on the cone is
 * \f{align}
 * \sqrt{\tilde{x}^2+\tilde{y}^2} \ge r_1 + (r_2-r_1)\tilde{\lambda}.
 * \f}
 *
 * The inverse map therefore rejects any points that do
 * not satisfy this criterion. The cone criterion makes sense only
 * for points with \f$z\geq z^{+}_{\mathrm{P}1}\f$.
 *
 * For \f$z^{-}_{\mathrm{P}2} \neq z^{-}_{\mathrm{P}1}\f$,
 * a similar cone can be constructed for the southern hemisphere. That
 * cone passes through
 * the circle \f$S^{-}_1\f$
 * defining the intersection of sphere 1
 * and the plane \f$z = z^{-}_{\mathrm{P}1}\f$ and the circle
 * \f$S^{-}_2\f$ defining the intersection of sphere 2 and the plane \f$z =
 * z^{-}_{\mathrm{P}2}\f$.  The inverse map rejects any point that is inside
 * that cone as well, provided that the point has
 * \f$z\leq z^{-}_{\mathrm{P}1}\f$.  For points with
 * \f$z > z^{-}_{\mathrm{P}1}\f$ checking the cone criterion
 * does not make sense.
 *
 * ## jacobian
 *
 * From Eqs. (\f$\ref{eq:deftheta1alt}\f$) and (\f$\ref{eq:deftheta2alt}\f$)
 * we see that \f$\theta_1\f$ and \f$\theta_2\f$ depend on \f$\bar{z}\f$ and
 * are independent of \f$\bar{x}\f$ and \f$\bar{y}\f$, and that
 * \f{align}
 * \frac{\partial (R_1\cos\theta_1)}{\partial\bar{z}}
 * &= \frac{1}{2}(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1}),
 *     \label{eq:dcostheta1} \\
 * \frac{\partial (R_2\cos\theta_2)}{\partial\bar{z}}
 * &= \frac{1}{2}(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2}),
 *     \label{eq:dcostheta2} \\
 * \frac{\partial (R_1\sin\theta_1)}{\partial\bar{z}}
 * &= -\frac{1}{2}\cot\theta_1 (z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1}),
 *     \label{eq:dsintheta1} \\
 * \frac{\partial (R_2\sin\theta_2)}{\partial\bar{z}}
 * &= -\frac{1}{2}\cot\theta_2(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2}),
 *     \label{eq:dsintheta2}
 * \f}
 *
 * Also, from Eqs. (\f$\ref{eq:defphi}\f$) and (\f$\ref{eq:defrhobar}\f$)
 * we have
 * \f{align}
 * \frac{\partial\cos\phi}{\partial\bar{x}}
 *  &= \frac{\bar{y}^2}{\bar{R}^3\bar{\rho}^3},
 *  \label{eq:dcosphidxbar} \\
 * \frac{\partial\cos\phi}{\partial\bar{y}}
 *  &= -\frac{\bar{x}\bar{y}}{\bar{R}^3\bar{\rho}^3},
 *  \label{eq:dcosphidybar} \\
 * \frac{\partial\sin\phi}{\partial\bar{x}}
 *  &= -\frac{\bar{x}\bar{y}}{\bar{R}^3\bar{\rho}^3},
 *  \label{eq:dsinphidxbar} \\
 * \frac{\partial\sin\phi}{\partial\bar{y}}
 *  &= \frac{\bar{x}^2}{\bar{R}^3\bar{\rho}^3},
 *  \label{eq:dsinphidybar}
 * \f}
 * and we know that \f$\phi\f$ is independent of \f$\bar{z}\f$.
 *
 * Finally, from Eqs. (\f$\ref{eq:defrhobar}\f$) and
 * (\f$\ref{eq:lambdafromrhobar}\f$) we have
 *
 * \f{align}
 * \frac{\partial\lambda}{\partial\bar{x}}
 *  &= \frac{\bar{x}}{\bar{R}^2\bar{\rho}},
 *  \label{eq:dlambdadxbar} \\
 * \frac{\partial\lambda}{\partial\bar{y}}
 *  &= \frac{\bar{y}}{\bar{R}^2\bar{\rho}},
 *  \label{eq:dlambdadybar}
 * \f}
 * with no dependence on \f$\bar{z}\f$.
 *
 * Putting these results together yields
 * \f{align}
 * \frac{\partial x^0}{\partial \bar{x}} &=
 * \frac{\bar{y}^2}{\bar{\rho}^3\bar{R}^3}R_1\sin\theta_1 +
 *        (R_2\sin\theta_2-R_1\sin\theta_1)
 *        \frac{\lambda \bar{R}^2\bar\rho^2+\bar{x}^2}{\bar\rho^3\bar{R}^3}
 *  + \frac{\bar{x}}{\bar\rho\bar{R}^2}(C_2^x-C_1^x),\\
 * \frac{\partial x^0}{\partial \bar{y}} &=
 * \frac{\bar{x}\bar{y}}{\bar{\rho}^3\bar{R}^3}
 *        (R_2\sin\theta_2-2 R_1\sin\theta_1)
 *  + \frac{\bar{y}}{\bar\rho\bar{R}^2}(C_2^x-C_1^x),\\
 * \frac{\partial x^0}{\partial \bar{z}} &=
 * -\frac{1}{2}\frac{\bar{x}}{\bar\rho\bar{R}}\left[
 *   \cot\theta_1(1-\lambda)(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})+
 *   \cot\theta_2\lambda(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2})\right],\\
 * \frac{\partial x^1}{\partial \bar{x}} &=
 * \frac{\bar{x}\bar{y}}{\bar{\rho}^3\bar{R}^3}
 *        (R_2\sin\theta_2-2 R_1\sin\theta_1)
 *  + \frac{\bar{x}}{\bar\rho\bar{R}^2}(C_2^y-C_1^y),\\
 * \frac{\partial x^1}{\partial \bar{y}} &=
 * \frac{\bar{x}^2}{\bar{\rho}^3\bar{R}^3}R_1\sin\theta_1 +
 *        (R_2\sin\theta_2-R_1\sin\theta_1)
 *        \frac{\lambda \bar{R}^2\bar\rho^2+\bar{y}^2}{\bar\rho^3\bar{R}^3}
 *  + \frac{\bar{y}}{\bar\rho\bar{R}^2}(C_2^y-C_1^y),\\
 * \frac{\partial x^1}{\partial \bar{z}} &=
 * -\frac{1}{2}\frac{\bar{y}}{\bar\rho\bar{R}}\left[
 *   \cot\theta_1(1-\lambda)(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})+
 *   \cot\theta_2\lambda(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2})\right],\\
 * \frac{\partial x^2}{\partial \bar{x}} &=
 *  \frac{\bar{x}}{\bar\rho\bar{R}^2}\left(
 *        C_2^z-C_1^z + R_2\cos\theta_2-R_1\cos\theta_1\right),\\
 * \frac{\partial x^2}{\partial \bar{y}} &=
 *  \frac{\bar{y}}{\bar\rho\bar{R}^2}\left(
 *        C_2^z-C_1^z + R_2\cos\theta_2-R_1\cos\theta_1\right),\\
 * \frac{\partial x^2}{\partial \bar{z}} &=
 * \frac{1}{2}(1-\lambda)(z^{+}_{\mathrm{P}1}-z^{-}_{\mathrm{P}1})+
 * \frac{1}{2}\lambda(z^{+}_{\mathrm{P}2}-z^{-}_{\mathrm{P}2}).
 * \f}
 *
 * ## inv_jacobian
 *
 * The inverse Jacobian is computed by numerically inverting the
 * Jacobian.
 *
 * ## Restrictions on map parameters
 *
 * We demand that Sphere 1 is fully contained inside Sphere 2, and
 * that the two spheres have at least some small separation between
 * them. In particular, we demand that
 * \f{align}
 *  0.98 R_2 &\geq R_1 + |C_1-C_2|, \label{eq:spherecontained}
 * \f}
 * where 0.98 is a safety factor. It is possible to construct a valid
 * map without this assumption, but the assumption simplifies the
 * code, and the expected use cases obey this restriction.
 *
 * We also demand that \f$R_1 \geq 0.08 R_2\f$.  Again, this assumption
 * is made for accuracy purposes and might be relaxed.
 *
 * ### Invertibility condition
 *
 * Consider the line segment \f$L^+_1\f$ that connects a point on the
 * circle \f$S^+_1\f$ (the circle formed by the intersection of sphere 1
 * and the plane \f$z=z^+_{\mathrm{P}1}\f$) with the center of the
 * circle \f$S^+_1\f$.  Consider another line segment \f$L^+_2\f$ that
 * connects the same point on the circle \f$S^+_1\f$ with the
 * corresponding point on the circle \f$S^+_2\f$ (the circle formed by
 * the intersection of sphere 2 and the plane
 * \f$z=z^+_{\mathrm{P}2}\f$). Now consider the angle between \f$L^+_1\f$
 * and \f$L^+_2\f$, as measured from the interior of sphere 1, and Let
 * \f$\alpha^+\f$ be the minimum value of this angle over the circle.
 * \f$\alpha^+\f$ is shown in the figure above. If
 * \f$\alpha^+ < \theta_{1 \mathrm{min}}\f$, then the line segment \f$L^+_2\f$
 * twice intersects the unmapped portion of sphere 1 near the north pole,
 * so the map is ill-defined.
 * Similarly, if \f$\alpha^+ < \theta_{2 \mathrm{min}}\f$,
 * then the line segment \f$L^+_2\f$ twice intersects the mapped portion of
 * sphere 2 near the north pole, and again the map is poorly defined.
 * Therefore we demand that the map parameters satisfy
 * - \f$\alpha^+ > 1.1 \theta_{1 \mathrm{min}}\f$
 * - \f$\alpha^+ > 1.1 \theta_{2 \mathrm{min}}\f$
 *
 * where 1.1 is a safety factor.
 *
 * Similarly, one can define an angle \f$\alpha^-\f$ for the region
 * near the south pole, and we require similar restrictions on that angle.
 *
 * ### Restrictions on z-planes
 *
 * We also demand that either
 * \f$z^+_{\mathrm{P}1} = z^+_{\mathrm{P}2}\f$
 * or that \f$z^+_{\mathrm{P}1} <= z^+_{\mathrm{P}2} -0.03 R_2\f$.
 * Similarly, we demand that either \f$z^-_{\mathrm{P}1} = z^-_{\mathrm{P}2}\f$
 * or \f$z^-_{\mathrm{P}1} >= z^-_{\mathrm{P}2} + 0.03 R_2\f$.
 * These restrictions follow expected use cases and avoid extreme distortions.
 *
 * ### Restrictions for unequal z planes
 * For \f$z^+_{\mathrm{P}1} \neq z^+_{\mathrm{P}2}\f$ and
 * \f$z^-_{\mathrm{P}1} \neq z^-_{\mathrm{P}2}\f$, we assume the following
 * restrictions on other parameters:
 *
 * We prohibit a tiny Sphere 1 near the edge of Sphere 2 by demanding that
 * \f{align}
 *  C^z_1 - R_1 &\leq C^z_2 + R_2/5,\\
 *  C^z_1 + R_1 &\geq C^z_2 - R_2/5.
 * \f}
 * We also demand that the polar axis of Sphere 2 intersects Sphere 1
 * somewhere:
 * \f{align}
 * \sqrt{(C^x_1-C^x_2)^2 + (C^y_1-C^y_2)^2} &\leq R_1.
 * \f}
 * and we demand that Sphere 1 is not too close to the edge of Sphere 2
 * in the \f$x\f$ or \f$y\f$ directions:
 * \f{align}
 * \sqrt{(C^y_1-C^y_2)^2 + (C^y_1-C^y_2)^2} &\leq \mathrm{max}(0,0.95 R_2-R_1),
 * \f}
 * where the max avoids problems when \f$0.95 R_2-R_1\f$ is negative
 * (which, if it occurs, means that the \f$x\f$ and \f$y\f$ centers of the
 * two spheres are equal).
 *
 * We require that the z planes in the above figures lie above/below
 * the centers of the corresponding spheres and are not too close to
 * the centers or edges of those spheres; specificially, we demand
 * that
 * \f{align}
 *   \label{eq:theta_1_min_res}
 *   0.15\pi &< \theta_{1 \mathrm{min}} < 0.4\pi \\
 *   \label{eq:theta_1_max_res}
 *   0.6\pi &< \theta_{1 \mathrm{max}} < 0.85\pi \\
 *   \label{eq:theta_2_min_res}
 *   0.15\pi &< \theta_{2 \mathrm{min}} < 0.4\pi \\
 *   \label{eq:theta_2_max_res}
 *   0.6\pi &< \theta_{2 \mathrm{max}} < 0.85\pi .
 * \f}
 *
 * Here the numerical values are safety factors.
 * These restrictions are not strictly necessary but are made for simplicity.
 * Increasing the range will make the maps less accurate because the domain
 * is more distorted. These parameters
 * can be changed provided the unit tests are changed to test the
 * appropriate parameter ranges.
 *
 * ### Restrictions for equal z planes
 *
 * If \f$z^+_{\mathrm{P}1} = z^+_{\mathrm{P}2}\f$ or
 * \f$z^-_{\mathrm{P}1} = z^-_{\mathrm{P}2}\f$ we demand that
 * \f$C_1^x=C_2^x\f$ and \f$C_1^y=C_2^y\f$, which simplifies the cases
 * we need to test and agrees with our expected use cases.
 * We also demand
 * \f{align}
 *   z^+_{\mathrm{P}2} &\geq z^-_{\mathrm{P}2} + 0.18 R_2
 * \f}
 * This condition is necessary because for unequal z planes,
 * \f$\theta_{2 \mathrm{min}}\f$ and
 * \f$\theta_{2 \mathrm{max}}\f$ are no longer required
 * to be on opposite sides of the equator of sphere 2 (see the paragraph below).
 *
 * As in the case with unequal z planes, we require that the
 * z planes in the above figures lie above/below
 * the centers of the corresponding spheres and are not too close to
 * the centers or edges of those spheres, but some of the restrictions are
 * relaxed because of expected use cases.  The conditions are the
 * same as Eqs. (\f$\ref{eq:theta_1_min_res}\f$--\f$\ref{eq:theta_2_max_res}\f$)
 * except if \f$z^+_{\mathrm{P}1} = z^+_{\mathrm{P}2}\f$
 * we replace Eq. (\f$\ref{eq:theta_2_min_res}\f$) with
 * \f{align}
 *   0.15\pi &< \theta_{2 \mathrm{min}} < 0.75\pi ,
 * \f}
 * and if \f$z^-_{\mathrm{P}1} = z^-_{\mathrm{P}2}\f$ we replace
 * Eq. (\f$\ref{eq:theta_2_max_res}\f$) with
 * \f{align}
 *   0.25\pi &< \theta_{2 \mathrm{max}} < 0.85\pi .
 * \f}
 */
class UniformCylindricalSide {
 public:
  static constexpr size_t dim = 3;
  UniformCylindricalSide(const std::array<double, 3>& center_one,
                         const std::array<double, 3>& center_two,
                         double radius_one, double radius_two,
                         double z_plane_plus_one, double z_plane_minus_one,
                         double z_plane_plus_two, double z_plane_minus_two);
  UniformCylindricalSide() = default;
  ~UniformCylindricalSide() = default;
  UniformCylindricalSide(UniformCylindricalSide&&) = default;
  UniformCylindricalSide(const UniformCylindricalSide&) = default;
  UniformCylindricalSide& operator=(const UniformCylindricalSide&) = default;
  UniformCylindricalSide& operator=(UniformCylindricalSide&&) = default;

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
  friend bool operator==(const UniformCylindricalSide& lhs,
                         const UniformCylindricalSide& rhs);
  std::array<double, 3> center_one_{};
  std::array<double, 3> center_two_{};
  double radius_one_{std::numeric_limits<double>::signaling_NaN()};
  double radius_two_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_plus_one_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_minus_one_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_plus_two_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_minus_two_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const UniformCylindricalSide& lhs,
                const UniformCylindricalSide& rhs);
}  // namespace domain::CoordinateMaps
