// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Contains FocallyLiftedInnerMaps
namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {
/*!
 * \brief A FocallyLiftedInnerMap that maps a 3D unit right cylinder
 *  to a volume that connects portions of two spherical surfaces.
 *
 * Because FocallyLiftedEndcap is a FocallyLiftedInnerMap, it is meant
 * to be a template parameter of FocallyLiftedMap, and its member functions
 * are meant to be used by FocallyLiftedMap. See FocallyLiftedMap for further
 * documentation.
 *
 * \image html FocallyLiftedEndcap.svg "Focally Lifted Endcap."
 *
 * \details The domain of the map is a 3D unit right cylinder with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$\bar{x}^2+\bar{y}^2 \leq
 * 1\f$.  The range of the map has coordinates \f$(x,y,z)\f$.
 *
 * Consider a sphere with center \f$C^i\f$ and radius \f$R\f$ that is
 * intersected by a plane normal to the \f$z\f$ axis and located at
 * \f$z = z_\mathrm{P}\f$.  In the figure above, every point
 * \f$\bar{x}^i\f$ in the blue region \f$\sigma=0\f$ maps to a point
 * \f$x_0^i\f$ on a portion of the surface of the sphere.
 *
 * `Endcap` provides the following functions:
 *
 * ### forward_map
 * `forward_map` maps \f$(\bar{x},\bar{y},\bar{z}=-1)\f$ to the portion of
 * the sphere with \f$z \geq z_\mathrm{P}\f$.  The arguments to `forward_map`
 * are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 * `forward_map` returns \f$x_0^i\f$,
 * the 3D coordinates on that sphere, which are given by
 *
 * \f{align}
 * x_0^0 &= R \frac{\sin(\bar{\rho} \theta_\mathrm{max})
 *              \bar{x}}{\bar{\rho}} + C^0,\\
 * x_0^1 &= R \frac{\sin(\bar{\rho} \theta_\mathrm{max})
 *              \bar{y}}{\bar{\rho}} + C^1,\\
 * x_0^2 &= R \cos(\bar{\rho} \theta_\mathrm{max}) + C^2.
 * \f}
 *
 * Here \f$\bar{\rho}^2 \equiv (\bar{x}^2+\bar{y}^2)/\bar{R}^2\f$, where
 * \f$\bar{R}\f$ is the radius of the cylinder in barred coordinates,
 * which is always unity,
 * and where
 * \f$\theta_\mathrm{max}\f$ is defined by
 * \f$\cos(\theta_\mathrm{max}) = (z_\mathrm{P}-C^2)/R\f$.
 * Note that when \f$\bar{\rho}=0\f$, we must evaluate
 * \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$
 * as \f$\theta_\mathrm{max}\f$.
 *
 * ### sigma
 *
 * \f$\sigma\f$ is a function that is zero at \f$\bar{z}=-1\f$
 * (which maps onto the sphere \f$x^i=x_0^i\f$) and
 * unity at \f$\bar{z}=+1\f$ (corresponding to the
 * upper surface of the FocallyLiftedMap). We define
 *
 * \f{align}
 *  \sigma &= \frac{\bar{z}+1}{2}.
 * \f}
 *
 * ### deriv_sigma
 *
 * `deriv_sigma` returns
 *
 * \f{align}
 *  \frac{\partial \sigma}{\partial \bar{x}^j} &= (0,0,1/2).
 * \f}
 *
 * ### jacobian
 *
 * `jacobian` returns \f$\partial x_0^k/\partial \bar{x}^j\f$.
 * The arguments to `jacobian`
 * are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 *
 * Differentiating Eqs.(1--3) above yields
 *
 * \f{align*}
 * \frac{\partial x_0^2}{\partial \bar{x}} &=
 * - R \theta_\mathrm{max}
 * \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\bar{x}\\
 * \frac{\partial x_0^2}{\partial \bar{y}} &=
 * - R \theta_\mathrm{max}
 * \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\bar{y}\\
 * \frac{\partial x_0^0}{\partial \bar{x}} &=
 * R \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}} +
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}^2\\
 * \frac{\partial x_0^0}{\partial \bar{y}} &=
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}\bar{y}\\
 * \frac{\partial x_0^1}{\partial \bar{x}} &=
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}\bar{y}\\
 * \frac{\partial x_0^1}{\partial \bar{y}} &=
 * R \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}} +
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{y}^2\\
 * \frac{\partial x_0^i}{\partial \bar{z}} &=0.
 * \f}
 *
 * ### Evaluating sinc functions
 *
 * Note that \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$ and
 * its derivative appear in the above equations.  We evaluate
 * \f$\sin(ax)/x\f$ in a straightforward way, except we are careful
 * to evaluate \f$\sin(ax)/x = a\f$ for the special case \f$x=0\f$.
 *
 * The derivative of the sync function is more complicated to evaluate
 * because of roundoff. Note that we can expand
 *
 * \f{align*}
 *  \frac{1}{x}\frac{d}{dx}\left(\frac{\sin(ax)}{x}\right) &=
 *  \frac{a}{x^2}\left(1 - 1 - \frac{2 (ax)^2}{3!} +
 *  \frac{4(ax)^4}{5!} - \frac{5(ax)^6}{7!} + \cdots \right),
 * \f}
 *
 * where we kept the "1 - 1" above as a reminder that when evaluating
 * this function directly as \f$(a \cos(ax)/x^2 - \sin(ax)/x^3)\f$, there
 * can be significant roundoff because of the "1" in each of the two
 * terms that are subtracted. The relative error in direct evaluation is
 * \f$3\epsilon/(ax)^2\f$, where \f$\epsilon\f$ is machine precision
 * (This expression comes from replacing "1 - 1" with \f$\epsilon\f$
 * and comparing the lowest-order contribution to the correct answer, i.e.
 * the \f$2(ax)^2/3!\f$ term, with \f$\epsilon\f$, the error contribution).
 * This means the error is 100% if \f$(ax)^2=3\epsilon\f$.
 *
 * To avoid roundoff, we evaluate the series if \f$ax\f$ is small
 * enough.  Suppose we keep terms up to and including the \f$(ax)^{2n}\f$
 * term in the series.  Then we evaluate the series if the
 * next term, the \f$(ax)^{2n+2}\f$ term, is roundoff,
 * i.e. if \f$(2n+2)(ax)^{2n+2}/(2n+3)! < \epsilon\f$.  In this case,
 * the direct evaluation has the maximum error if
 * \f$(2n+2)(ax)^{2n+2}/(2n+3)! = \epsilon\f$.  We showed above that the
 * relative error in direct evaluation is \f$3\epsilon/(ax)^2\f$,
 * which evaluates to \f$(\epsilon^n (2n+2)/(2n+3)!)^{1/(n+1)}\f$.
 *
 * \f{align*}
 *   n=1 \qquad& \mathrm{error}=3(\epsilon/30)^{1/2} &\qquad
 *               \sim \mathrm{5e-09}\\
 *   n=2 \qquad& \mathrm{error}=3(\epsilon^2/840)^{1/3} &\qquad
 *               \sim \mathrm{7e-12}\\
 *   n=3 \qquad& \mathrm{error}=3(\epsilon^3/45360)^{1/4} &\qquad
 *               \sim \mathrm{2e-13}\\
 *   n=4 \qquad& \mathrm{error}=3(\epsilon^4/3991680)^{1/5} &\qquad
 *               \sim \mathrm{2e-14}\\
 *   n=5 \qquad& \mathrm{error}=3(\epsilon^5/518918400)^{1/6} &\qquad
 *               \sim \mathrm{5e-15}\\
 *   n=6 \qquad& \mathrm{error}=3(\epsilon^6/93405312000)^{1/7} &\qquad
 *               \sim \mathrm{1e-15}
 * \f}
 * We gain less and less with each order, so we choose \f$n=3\f$.
 * Then the series above can be rewritten to this order in the form
 * \f{align*}
 *  \frac{1}{x}\frac{d}{dx}\left(\frac{\sin(ax)}{x}\right) &=
 *  -\frac{a^3}{3}\left(1 - \frac{3\cdot 4 (ax)^2}{5!} +
 *  \frac{3 \cdot 6(ax)^4}{7!}\right).
 * \f}
 *
 * ### inverse
 *
 * `inverse` takes \f$x_0^i\f$ and \f$\sigma\f$ as arguments, and
 * returns \f$(\bar{x},\bar{y},\bar{z})\f$, or boost::none if
 * \f$x_0^i\f$ or \f$\sigma\f$ are outside the range of the map.
 * For example, if \f$x_0^i\f$ does not lie on the sphere,
 * we return boost::none.
 *
 * The easiest to compute is \f$\bar{z}\f$, which is given by inverting
 * Eq. (4):
 *
 * \f{align}
 *  \bar{z} &= 2\sigma - 1.
 * \f}
 *
 * If \f$\bar{z}\f$ is outside the range \f$[-1,1]\f$ then we return
 * boost::none.
 *
 * To get \f$\bar{x}\f$ and \f$\bar{y}\f$,
 * we invert
 * Eqs (1--3).  If \f$x_0^0=x_0^1=0\f$, then \f$\bar{x}=\bar{y}=0\f$.
 * Otherwise, we compute
 *
 * \f{align}
 *   \bar{\rho} = \theta_\mathrm{max}^{-1}
 *   \tan^{-1}\left(\frac{\rho}{x_0^2-C^2}\right),
 * \f}
 *
 * where \f$\rho^2 = (x_0^0-C^0)^2+(x_0^1-C^1)^2\f$. Then
 *
 * \f{align}
 * \bar{x} &= (x_0^0-C^0)\frac{\bar{\rho}}{\rho},\\
 * \bar{y} &= (x_0^1-C^1)\frac{\bar{\rho}}{\rho}.
 * \f}
 *
 * Note that if \f$\bar{x}^2+\bar{y}^2 > 1\f$, the original point is outside
 * the range of the map so we return boost::none.
 *
 * ### lambda_tilde
 *
 * `lambda_tilde` takes as arguments a point \f$x^i\f$ and a projection point
 *  \f$P^i\f$, and computes \f$\tilde{\lambda}\f$, the solution to
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda}.\f}
 *
 * Since \f$x_0^i\f$ must lie on the sphere, \f$\tilde{\lambda}\f$ is the
 * solution of the quadratic equation
 *
 * \f{align}
 * |P^i + (x^i - P^i) \tilde{\lambda} - C^i |^2 - R^2 = 0.
 * \f}
 *
 * In solving the quadratic, we demand a root that is positive and
 * less than or equal to unity, since \f$x_0^i\f$ is always between
 * the projection point and \f$x^i\f$. If there are two suitable
 * roots, this means that the entire sphere lies between \f$P^i\f$ and
 * \f$x^i\f$; in this case if \f$x^2 \geq z_\mathrm{P}\f$ we choose the
 * larger root, otherwise we choose the smaller one: this gives
 * us the root with \f$x_0^2 \geq z_\mathrm{P}\f$, the portion of the sphere
 * that is the range of `Endcap`. If there is no suitable root,
 * this means that the point \f$x^i\f$ is not in the range of the map
 * so we return a default-constructed std::optional.
 *
 * ### deriv_lambda_tilde
 *
 * `deriv_lambda_tilde` takes as arguments \f$x_0^i\f$, a projection point
 *  \f$P^i\f$, and \f$\tilde{\lambda}\f$, and
 *  returns \f$\partial \tilde{\lambda}/\partial x^i\f$.
 * By differentiating Eq. (11), we find
 *
 * \f{align}
 * \frac{\partial\tilde{\lambda}}{\partial x^j} &=
 * \tilde{\lambda}^2 \frac{C^j - x_0^j}{|x_0^i - P^i|^2
 * + (x_0^i - P^i)(P_i - C_{i})}.
 * \f}
 *
 * ### inv_jacobian
 *
 * `inv_jacobian` returns \f$\partial \bar{x}^i/\partial x_0^k\f$,
 *  where \f$\sigma\f$ is held fixed.
 *  The arguments to `inv_jacobian`
 *  are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 *
 * Note that \f$\bar{x}\f$ and \f$\bar{y}\f$ can be considered to
 * depend only on \f$x_0^0\f$ and \f$x_0^1\f$ but not on \f$x_0^2\f$,
 * because the point \f$x_0^i\f$ is constrained to lie on a sphere of
 * radius \f$R\f$.  Note that there is an alternative way to compute
 * Eqs. (8) and (9) using only \f$x_0^0\f$ and \f$x_0^1\f$. To do
 * this, define
 *
 * \f{align}
 * \upsilon \equiv \sin(\bar{\rho}\theta_\mathrm{max})
 *        &= \sqrt{\frac{(x_0^0-C^0)^2+(x_0^1-C^1)^2}{R^2}}.
 * \f}
 *
 * Then we can write
 *
 * \f{align}
 * \frac{1}{\bar{\rho}}\sin(\bar{\rho}\theta_\mathrm{max})
 * &= \frac{\theta_\mathrm{max}\upsilon}{\arcsin(\upsilon)},
 * \f}
 *
 * so that
 *
 * \f{align}
 * \bar{x} &= \frac{x_0^0-C^0}{R}\left(\frac{1}{\bar{\rho}}
 *             \sin(\bar{\rho}\theta_\mathrm{max})\right)^{-1} \\
 * \bar{y} &= \frac{x_0^1-C^1}{R}\left(\frac{1}{\bar{\rho}}
 *             \sin(\bar{\rho}\theta_\mathrm{max})\right)^{-1}.
 * \f}
 *
 * We will compute \f$\partial \bar{x}^i/\partial
 * x_0^k\f$ by differentiating Eqs. (15) and (16).  Because those equations
 * involve \f$\bar{\rho}\f$, we first establish some relations
 * involving derivatives of \f$\bar{\rho}\f$.  For ease of notation, we define
 *
 * \f{align}
 * q \equiv \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}.
 * \f}
 *
 * First observe that
 * \f{align}
 * \frac{dq}{d\upsilon}
 * = \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},
 * \f}
 *
 * where \f$\upsilon\f$ is the quantity defined by Eq. (13).  Therefore
 *
 * \f{align}
 * \frac{\partial q}{\partial x_0^0} &=
 * \frac{\bar{x}}{\bar{\rho}R}\frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial q}{\partial x_0^1} &=
 * \frac{\bar{y}}{\bar{\rho}R}\frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},
 * \f}
 *
 * where we have differentiated Eq. (13), and where we have
 * used Eqs. (15) and (16) to eliminate \f$x_0^0\f$ and
 * \f$x_0^1\f$ in favor of \f$\bar{x}\f$ and
 * \f$\bar{y}\f$ in the final result.
 *
 * Let
 * \f{align}
 * \Sigma \equiv \frac{1}{\bar\rho} \frac{dq}{d\bar{\rho}},
 * \f}
 * since that combination will appear frequently in formulas below. Note that
 * \f$\Sigma\f$ has a finite limit as \f$\bar{\rho}\to 0\f$, and it is evaluated
 * according to the section on evaluating sinc functions above.
 *
 * By differentiating Eqs. (15) and (16), and using Eqs. (19) and (20), we
 * find
 *
 * \f{align}
 * \frac{\partial \bar{x}}{\partial x_0^0} &=
 * \frac{1}{R q}
 * - \frac{\bar{x}^2 \Sigma}{R q}
 * \left(\bar{\rho}^2 \Sigma + q\right)^{-1},\\
 * \frac{\partial \bar{x}}{\partial x_0^1} &=
 * - \frac{\bar{x}\bar{y} \Sigma}{R q}
 * \left(\bar{\rho}^2 \Sigma + q\right)^{-1},\\
 * \frac{\partial \bar{x}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{y}}{\partial x_0^0} &=
 * \frac{\partial \bar{x}}{\partial x_0^1},\\
 * \frac{\partial \bar{y}}{\partial x_0^1} &=
 * \frac{1}{R q}
 * - \frac{\bar{y}^2 \Sigma}{R q}
 * \left(\bar{\rho}^2 \Sigma + q\right)^{-1},\\
 * \frac{\partial \bar{y}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{z}}{\partial x_0^i} &= 0.
 * \f}
 * Note that care must be taken to evaluate
 * \f$q = \sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$ and its
 * derivative \f$\Sigma\f$ near \f$\bar{\rho}=0\f$; see the discussion above on
 * evaluating sinc functions.
 *
 * ### dxbar_dsigma
 *
 * `dxbar_dsigma` returns \f$\partial \bar{x}^i/\partial \sigma\f$,
 *  where \f$x_0^i\f$ is held fixed.
 *
 * From Eq. (6) we have
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial \sigma} &= (0,0,2).
 * \f}
 *
 */
class Endcap {
 public:
  Endcap(const std::array<double, 3>& center, double radius,
         double z_plane) noexcept;

  Endcap() = default;
  ~Endcap() = default;
  Endcap(Endcap&&) = default;
  Endcap(const Endcap&) = default;
  Endcap& operator=(const Endcap&) = default;
  Endcap& operator=(Endcap&&) = default;

  template <typename T>
  void forward_map(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          target_coords,
      const std::array<T, 3>& source_coords) const noexcept;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords,
      double sigma_in) const noexcept;

  template <typename T>
  void jacobian(const gsl::not_null<
                    tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>*>
                    jacobian_out,
                const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  void inv_jacobian(const gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3,
                                                 Frame::NoFrame>*>
                        inv_jacobian_out,
                    const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  void sigma(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> sigma_out,
             const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  void deriv_sigma(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          deriv_sigma_out,
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  void dxbar_dsigma(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          dxbar_dsigma_out,
      const std::array<T, 3>& source_coords) const noexcept;

  std::optional<double> lambda_tilde(
      const std::array<double, 3>& parent_mapped_target_coords,
      const std::array<double, 3>& projection_point,
      bool source_is_between_focus_and_target) const noexcept;

  template <typename T>
  void deriv_lambda_tilde(
      const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>
          deriv_lambda_tilde_out,
      const std::array<T, 3>& target_coords, const T& lambda_tilde,
      const std::array<double, 3>& projection_point) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

 private:
  friend bool operator==(const Endcap& lhs, const Endcap& rhs) noexcept;
  std::array<double, 3> center_{};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  double theta_max_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Endcap& lhs, const Endcap& rhs) noexcept;
}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
