// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class CylindricalEndcap.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

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
 * \brief Map from 3D unit right cylinder to a volume that connects
 *  portions of two spherical surfaces.
 *
 * \image html CylindricalEndcap.svg "A cylinder maps to the shaded region."
 *
 * \details Consider two spheres with centers \f$C_1\f$ and \f$C_2\f$,
 * and radii \f$R_1\f$ and \f$R_2\f$. Let sphere 1 be intersected by a
 * plane normal to the \f$z\f$ axis and located at \f$z = z_\mathrm{P}\f$.
 * Also let there be a projection point \f$P\f$.
 *
 * CylindricalEndcap maps a 3D unit right cylinder (with coordinates
 * \f$(\bar{x},\bar{y},\bar{z})\f$ such that \f$-1\leq\bar{z}\leq 1\f$
 * and \f$\bar{x}^2+\bar{y}^2+\bar{z}^2 \leq 1\f$) to the shaded area
 * in the figure above (with coordinates \f$(x,y,z)\f$).  The "bottom"
 * of the cylinder \f$\bar{z}=-1\f$ is mapped to the portion of sphere
 * 1 that has \f$z \geq z_\mathrm{P}\f$.  Curves of constant
 * \f$(\bar{x},\bar{y})\f$ are mapped to portions of lines that pass
 * through \f$P\f$. Along each of these curves, \f$\bar{z}=-1\f$ is
 * mapped to a point on sphere 1 and \f$\bar{z}=+1\f$ is mapped to a
 * point on sphere 2.
 *
 * CylindricalEndcap is intended to be composed with Wedge2D maps to
 * construct a portion of a cylindrical domain for a binary system.
 *
 * CylindricalEndcap is described briefly in the Appendix of
 * Phys. Rev. D 86, 084033 (2012), https://arxiv.org/abs/1206.3015.
 * CylindricalEndcap is used to construct the blocks labeled 'CA
 * wedge', 'EA wedge', 'CB wedge', 'EE wedge', and 'EB wedge' in
 * Figure 20 of that paper.
 *
 * ### More detail on how the map is constructed:
 *
 * Let \f$x_0^i\f$ be the coordinates \f$x^i\f$ that lie on
 * the two-dimensional image of \f$(\bar{x},\bar{y},\bar{z}=-1)\f$.
 * Thus \f$x_0^i\f$ lies on sphere 1 and has \f$x_0^2 \geq x_{\mathrm P}\f$.
 * \f$x_0^i\f$ is determined by
 *
 * \f{align}
 * x_0^0 &= R_1 \frac{\sin(\bar{\rho} \theta_\mathrm{max})
 *              \bar{x}}{\bar{\rho}} + C_1^0\\
 * x_0^1 &= R_1 \frac{\sin(\bar{\rho} \theta_\mathrm{max})
 *              \bar{y}}{\bar{\rho}} + C_1^1\\
 * x_0^2 &= R_1 \cos(\bar{\rho} \theta_\mathrm{max}) + C_1^2
 * \f}
 *
 * where \f$\bar{\rho}^2 \equiv \bar{x}^2+\bar{y}^2\f$, and where
 * \f$\theta_\mathrm{max}\f$ is defined by
 * \f$\cos(\theta_\mathrm{max}) = (z_\mathrm{P}-C_1^2)/R_1\f$.
 * Note that care must be taken to evaluate
 * \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$
 * near \f$\bar{\rho}=0\f$.
 *
 * Now let \f$x_1^i\f$ be the image of \f$(\bar{x},\bar{y},\bar{z}=+1)\f$.
 * This point lies on sphere 2, and is constructed so that \f$P\f$,
 * \f$x_0^i\f$, and \f$x_1^i\f$ are co-linear.  In particular, \f$x_1^i\f$
 * is determined by the equation
 *
 * \f{align} x_1^i = P^i + (x_0^i - P^i) \lambda,\f}
 *
 * where \f$\lambda\f$ is a scalar factor that depends on \f$x_0^i\f$ and
 * that can be computed by solving a quadratic equation.  This quadratic
 * equation is derived by demanding that \f$x_1^i\f$ lies on sphere2:
 *
 * \f{align}
 * |P^i + (x_0^i - P^i) \lambda - C_2^i |^2 - R_2^2 = 0.
 * \f}
 *
 * Finally, once \f$\lambda\f$ has been computed and \f$x_1^i\f$ has
 * been determined, the result of the CylindricalEndcap map is given
 * by mapping linearly in \f$\bar{z}\f$ between \f$x_0^i\f$ and
 * \f$x_1^i\f$ as follows:
 *
 * \f[x^i = x_0^i + (x_1^i - x_0^i) \frac{\bar{z}+1}{2}\f]
 *
 * ### Jacobian
 *
 * Because \f$x_0^i\f$ and \f$x_1^i\f$ are independent of \f$\bar{z}\f$,
 * a few of the Jacobian components are easy:
 *
 * \f[
 * \frac{\partial x^i}{\partial \bar{z}} = \frac{1}{2}(x_1^i - x_0^i).
 * \f]
 *
 * The other components of the Jacobian require more effort.
 *
 * Differentiating Eqs.(1--3) above yields
 *
 * \f{align*}
 * \frac{\partial x_0^2}{\partial \bar{x}} &=
 * - R_1 \theta_\mathrm{max}
 * \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\bar{x}\\
 * \frac{\partial x_0^2}{\partial \bar{y}} &=
 * - R_1 \theta_\mathrm{max}
 * \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\bar{y}\\
 * \frac{\partial x_0^0}{\partial \bar{x}} &=
 * R_1 \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}} +
 * R_1 \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}^2\\
 * \frac{\partial x_0^0}{\partial \bar{y}} &=
 * R_1 \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}\bar{y}\\
 * \frac{\partial x_0^1}{\partial \bar{x}} &=
 * R_1 \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}\bar{y}\\
 * \frac{\partial x_0^1}{\partial \bar{y}} &=
 * R_1 \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}} +
 * R_1 \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{y}^2,
 * \f}
 * where care must be taken to evaluate
 * \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$
 * and its derivatives near \f$\bar{\rho}=0\f$.
 *
 * Differentiating Eq.(5) above yields
 *
 * \f{align}
 * \frac{\partial\lambda}{\partial x_0^j} &=
 * \lambda^2 \frac{C_2^j - x_1^j}{|x_1^i - P^i|^2
 * + (x_1^i - P^i)(P_i - C_{2i})},
 * \f}
 *
 * and differentiating Eq.(4) above yields
 *
 * \f[
 * \frac{\partial x_1^i}{\partial x_0^j} = \lambda +
 *  (x_0^i - P^i) \frac{\partial \lambda}{\partial x_0^j}.
 * \f]
 *
 * Combining the above results, for \f$\bar{x}^j\f$ not equal to
 * \f$\bar{z}\f$ the Jacobian is evaluated via
 *
 * \f[
 * \frac{\partial x^i}{\partial \bar{x}^j} =
 * \frac{1+\bar{z}}{2} \frac{\partial x_1^i}{\partial x_0^k}
 * \frac{\partial x_0^k}{\partial \bar{x}^j} +
 * \frac{1-\bar{z}}{2}\frac{\partial x_0^i}{\partial \bar{x}^j}.
 * \f]
 *
 * ### Inverse map
 *
 * Given \f$x^i\f$, we wish to compute \f$\bar{x}\f$,
 * \f$\bar{y}\f$, and \f$\bar{z}\f$.
 *
 * We first find the coordinates \f$x_0^i\f$ that lie on sphere 1
 * and are defined such that \f$P\f$, \f$x_0^i\f$, and \f$x^i\f$ are co-linear.
 * \f$x_0^i\f$ is determined by the equation
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda},\f}
 *
 * where \f$\tilde{\lambda}\f$ is a scalar factor that depends on \f$x^i\f$ and
 * that can be computed by solving a quadratic equation.  This quadratic
 * equation is derived by demanding that \f$x_0^i\f$ lies on sphere 1:
 *
 * \f{align}
 * |P^i + (x^i - P^i) \tilde{\lambda} - C_1^i |^2 - R_1^2 = 0.
 * \f}
 *
 * Note that Eqs. (7) and (8) are similar to Eqs. (4) and (5) above.
 * In solving the quadratic, we choose the larger root if
 * \f$x^2>z_\mathrm{P}\f$ and the smaller root otherwise. We demand that
 * the root is greater than unity.  If there is no such root, this means
 * that the point \f$x^i\f$ is not in the range of the map and the
 * inverse function returns boost::none.
 * Also, if the point \f$x_0^i\f$ computed by Eq. (7) does not lie
 * on sphere 1, then likewise the point \f$x^i\f$ is not in the range
 * of the map and the inverse function returns boost::none.
 *
 * Now consider the coordinates \f$x_1^i\f$ that lie on sphere 2
 * and are defined such that \f$P\f$, \f$x_1^i\f$, and \f$x^i\f$ are co-linear.
 * \f$x_1^i\f$ is determined by the equation
 *
 * \f{align} x_1^i = P^i + (x^i - P^i) \bar{\lambda},\f}
 *
 * where \f$\bar{\lambda}\f$ is a scalar factor that depends on \f$x^i\f$ and
 * is the solution of a quadratic
 * that is derived by demanding that \f$x_1^i\f$ lies on sphere 2:
 *
 * \f{align}
 * |P^i + (x^i - P^i) \bar{\lambda} - C_2^i |^2 - R_2^2 = 0.
 * \f}
 *
 * Once we have found \f$\bar{\lambda}\f$, we don't actually need to compute
 * \f$x_1^i\f$. Instead, we use \f$\bar{\lambda}\f$ and \f$\tilde{\lambda}\f$
 * to determine \f$\bar{z}\f$ by the equation
 *
 * \f{align}
 * \bar{z} &= 2 \frac{\tilde{\lambda}-1}{\tilde{\lambda}-\bar{\lambda}} - 1,
 * \f}
 *
 * or by the equivalent equation
 *
 * \f{align}
 * \bar{z} &= 2 \frac{\bar{\lambda}-1}{\tilde{\lambda}-\bar{\lambda}} + 1,
 * \f}
 *
 * where to minimize roundoff error we choose the first
 * equation if \f$\tilde{\lambda}\sim 1\f$ and the second equation otherwise.
 *
 * We now have \f$\bar{z}\f$. To get \f$\bar{x}\f$ and \f$\bar{y}\f$,
 * we use our computed values of \f$x_0^i\f$ above and invert
 * Eqs (1--3).  Note that we have already demanded that
 * \f$x_0^i\f$ lies on sphere 1, so therefore the inverse of Eqs. (1--3)
 * depends on \f$x_0^0\f$ and \f$x_0^1\f$ but not on \f$x_0^2\f$.
 * To compute the inverse, we start by finding
 *
 * \f{align}
 * \sigma \equiv \sin(\bar{\rho}\theta_\mathrm{max})
 *        &= \sqrt{\frac{(x_0^0-C_1^0)^2+(x_0^1-C_1^1)^2}{R^2_1}}.
 * \f}
 *
 * Then we determine \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$
 * from
 *
 * \f{align*}
 * \frac{1}{\bar{\rho}}\sin(\bar{\rho}\theta_\mathrm{max})
 * &= \frac{\theta_\mathrm{max}\sigma}{\arcsin(\sigma)}\\
 * &\sim \theta_\mathrm{max}\left(1-\frac{\sigma^2}{6}\right)
 * \qquad \hbox{(for small $\bar{\rho}$)},
 * \f}
 *
 * where we use the Taylor approximation only for small \f$\bar{\rho}\f$.
 * Finally, we have
 *
 * \f{align}
 * \bar{x} &= \frac{x_0^0-C_1^0}{R_1}\left(\frac{1}{\bar{\rho}}
              \sin(\bar{\rho}\theta_\mathrm{max})\right)^{-1}\\
 * \bar{y} &= \frac{x_0^1-C_1^1}{R_1}\left(\frac{1}{\bar{\rho}}
              \sin(\bar{\rho}\theta_\mathrm{max})\right)^{-1}.
 * \f}
 *
 * Note that if \f$\bar{x}^2+\bar{y}^2 > 1\f$, the original point is outside
 * the range of the map so we return boost::none.
 *
 * #### Root polishing
 *
 * The inverse function described above will sometimes have errors that
 * are noticeably larger than roundoff.  Therefore we apply a single
 * Newton-Raphson iteration to refine the result of the inverse map:
 * Suppose we are given \f$x^i\f$, and we have computed \f$\bar{x}^i\f$
 * by the above procedure.  We then correct \f$\bar{x}^i\f$ by adding
 *
 * \f[
 * \delta \bar{x}^i = \left(x^j - x^j(\bar{x})\right)
 * \frac{\partial \bar{x}^i}{\partial x^j},
 * \f]
 *
 * where \f$x^j(\bar{x})\f$ is the result of applying the forward map
 * to \f$\bar{x}^i\f$ and \f$\partial \bar{x}^i/\partial x^j\f$ is the
 * inverse jacobian.
 *
 * ### Inverse jacobian
 *
 * We first consider components \f$\partial\bar{z}/\partial x^i\f$; we
 * will separately treat the other components below.
 * We note that \f$\bar{\lambda}=\tilde{\lambda}\lambda\f$,
 * so Eq. (11) and (12) are equivalent to
 *
 * \f{align}
 * \bar{z} &=
 * 2 \frac{\tilde{\lambda}-1}{\tilde{\lambda}-\lambda\tilde{\lambda}} - 1.
 * \f}
 *
 * Differentiating this expression yields
 *
 * \f{align}
 * \frac{\partial \bar{z}}{\partial x^i} &=
 * \frac{\partial \bar{z}}{\partial \lambda}
 * \frac{\partial \lambda}{\partial x_0^j}
 * \frac{\partial x_0^j}{\partial x^i}
 * + \frac{\partial \bar{z}}{\partial \tilde\lambda}
 *   \frac{\partial \tilde\lambda}{\partial x^i}.
 * \f}
 *
 * We now compute the factors on the right-hand side of Eq. (17).
 * By differentiating Eq. (8), we find that
 *
 * \f{align}
 * \frac{\partial\tilde{\lambda}}{\partial x^j} &=
 * \lambda^2 \frac{C_1^j - x_0^j}{|x_0^i - P^i|^2
 * + (x_0^i - P^i)(P_i - C_{1i})}.
 * \f}
 *
 * Differentiating Eq. (7) yields
 *
 * \f{align}
 * \frac{\partial x_0^j}{\partial x^i} &= \tilde{\lambda} \delta_i^j
 * + \frac{x_0^i-P^i}{\tilde{\lambda}}
 *   \frac{\partial\tilde{\lambda}}{\partial x^j}.
 * \f}
 *
 * We can now evaluate Eq. (17) using Eqs. (6), (16), (18), and (19).
 *
 * Now we turn to \f$\partial \bar{x}/\partial x^i\f$ and
 * \f$\partial \bar{y}/\partial x^i\f$.  We write
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial x^j} =
 * \frac{\partial x_0^k}{\partial x^j}
 * \frac{\partial \bar{x}^i}{\partial x_0^k}
 * \qquad (i\neq 2),
 * \f}
 *
 * The first factor on the right-hand side can be evaluated using
 * Eq. (19).  Now we turn to the second factor
 * \f$\partial \bar{x}^i/\partial x_0^k\f$.  Note that
 * \f$\bar{x}\f$ and \f$\bar{y}\f$ can be considered to depend only on
 * \f$x_0^0\f$ and \f$x_0^1\f$ but not on \f$x_0^2\f$, because the
 * point \f$x_0^i\f$ is constrained to lie on a sphere of radius \f$R_1\f$.
 * We will compute \f$\partial \bar{x}^i/\partial x_0^k\f$ by differentiating
 * Eqs. (14) and (15), but those equations involve \f$\bar{\rho}\f$, so
 * first we establish some relations involving \f$\bar{\rho}\f$.  For
 * ease of notation, we define
 *
 * \f[
 * q \equiv \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}.
 * \f]
 *
 * First observe that
 * \f[
 * \frac{dq}{d\sigma}
 * = \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},
 * \f]
 *
 * where \f$\sigma\f$ is the quantity defined by Eq. (13).  Therefore
 *
 * \f{align}
 * \frac{\partial q}{\partial x_0^0} &=
 * \frac{\bar{x}}{\bar{\rho}}\frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial q}{\partial x_0^1} &=
 * \frac{\bar{y}}{\bar{\rho}}\frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},
 * \f}
 *
 * where we have differentiated Eq. (13), and where we have
 * used Eqs. (14) and (15) to eliminate \f$x_0^0\f$ and
 * \f$x_0^1\f$ in favor of \f$\bar{x}\f$ and
 * \f$\bar{y}\f$ in the final result.
 *
 * By differentiating Eqs. (14) and (15), and using Eqs. (21) and (22), we
 * find
 *
 * \f{align}
 * \frac{\partial \bar{x}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{x}}{\partial x_0^0} &=
 * \frac{1}{R_1 q}
 * - \frac{\bar{x}^2}{R_1 q \bar{\rho}} \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial \bar{x}}{\partial x_0^1} &=
 * - \frac{\bar{x}\bar{y}}{R_1 q \bar{\rho}} \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial \bar{y}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{y}}{\partial x_0^0} &=
 * \frac{\partial \bar{x}}{\partial x_0^1},\\
 * \frac{\partial \bar{y}}{\partial x_0^1} &=
 * \frac{1}{R_1 q}
 * - \frac{\bar{y}^2}{R_1 q \bar{\rho}} \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1}.
 * \f}
 * Note that care must be taken to evaluate
 * \f$q = \sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$ and its
 * derivative near \f$\bar{\rho}=0\f$.
 *
 * Now we can evaluate Eq. (20) using Eq. (19) and
 * Eqs. (23--28).
 *
 * ### Restrictions on map parameters.
 *
 * We demand that Sphere 1 is fully contained inside Sphere 2. It is
 * possible to construct a valid map without this assumption, but the
 * assumption simplifies the code, and the expected use cases obey
 * this restriction.
 *
 * We also demand that \f$z_\mathrm{P} > C_1^2\f$, that is, the plane
 * in the above and below figures lies to the right of \f$C_1^2\f$.
 * This restriction not strictly necessary but is made for simplicity.
 *
 * The map is invertible only for some choices of the projection point
 * \f$P\f$.  Given the above restrictions, the allowed values of
 * \f$P\f$ are illustrated by the following diagram:
 *
 * \image html CylindricalEndcap_Allowed.svg "Allowed region for P." width=75%
 *
 * The plane \f$z=z_\mathrm{P}\f$ intersects sphere 1 on a circle. The
 * cone with apex \f$C_1\f$ that intersects that circle has opening
 * angle \f$2\theta\f$ as shown in the above figure. Construct another
 * cone, the "invertibility cone", with apex \f$S\f$ chosen such that
 * the two cones intersect at right angles on the circle; thus the
 * opening angle of the invertibility cone is \f$2(\pi-\theta)\f$.  A
 * necessary condition for invertibility is that the projection point
 * \f$P\f$ lies inside the invertibility cone, but not between \f$S\f$
 * and sphere 1.  Placing the projection point \f$P\f$ to the right of
 * \f$S\f$ (but inside the invertibility cone) is ok for
 * invertibility.
 *
 * In addition to invertibility and the two additional restrictions
 * already mentioned above, we demand a few more restrictions on the
 * map parameters to simplify the logic for the expected use cases, and
 * to ensure that jacobians do not get too large. We demand:
 *
 * - \f$P\f$ is not too close to the edge of the invertibility cone.
 * - \f$P\f$ is contained in sphere 2.
 * - \f$P\f$ is to the left of \f$z_\mathrm{P}\f$.
 * - \f$z_\mathrm{P}\f$ is not too close to the center or the edge of sphere 1.
 * - If a line segment is drawn between \f$P\f$ and any point on the
 *   intersection circle, the angle between the line segment and
 *   the x-axis is smaller than \f$\pi/3\f$.
 *
 */
class CylindricalEndcap {
 public:
  static constexpr size_t dim = 3;
  CylindricalEndcap(const std::array<double, 3>& center_one,
                    const std::array<double, 3>& center_two,
                    const std::array<double, 3>& proj_center, double radius_one,
                    double radius_two, double z_plane) noexcept;

  CylindricalEndcap() = default;
  ~CylindricalEndcap() = default;
  CylindricalEndcap(CylindricalEndcap&&) = default;
  CylindricalEndcap(const CylindricalEndcap&) = default;
  CylindricalEndcap& operator=(const CylindricalEndcap&) = default;
  CylindricalEndcap& operator=(CylindricalEndcap&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

 private:
  friend bool operator==(const CylindricalEndcap& lhs,
                         const CylindricalEndcap& rhs) noexcept;
  std::array<double, 3> center_one_{}, center_two_{}, proj_center_{};
  double radius_one_{std::numeric_limits<double>::signaling_NaN()};
  double radius_two_{std::numeric_limits<double>::signaling_NaN()};
  double z_plane_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const CylindricalEndcap& lhs,
                const CylindricalEndcap& rhs) noexcept;
}  // namespace domain::CoordinateMaps
