// Distributed under the MIT License.
// See LICENSE.txt for details.

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

template <typename InnerMap>
class FocallyLiftedMap;

template <typename InnerMap>
bool operator==(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs);

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Map from \f$(\bar{x},\bar{y},\bar{z})\f$ to the volume
 * contained between a 2D surface and the surface of a 2-sphere.
 *
 *
 * \image html FocallyLifted.svg "2D representation of focally lifted map."
 *
 * \details We are given the radius \f$R\f$ and
 * center \f$C^i\f$ of a sphere, and a projection point \f$P^i\f$.  Also,
 * through the class defined by the `InnerMap` template parameter,
 * we are given the functions
 * \f$f^i(\bar{x},\bar{y},\bar{z})\f$ and
 * \f$\sigma(\bar{x},\bar{y},\bar{z})\f$ defined below; these
 * functions define the mapping from \f$(\bar{x},\bar{y},\bar{z})\f$
 * to the 2D surface.
 *
 * The above figure is a 2D representation of the focally lifted map,
 * where the shaded region in \f$\bar{x}^i\f$ coordinates on the left
 * is mapped to the shaded region in the \f$x^i\f$ coordinates on the
 * right.  Shown is an arbitrary point \f$\bar{x}^i\f$ that is mapped
 * to \f$x^i\f$; for that point, the corresponding \f$x_0^i\f$ and
 * \f$x_1^i\f$ (defined below) are shown on the right, as is the
 * projection point \f$P^i\f$.  Also shown are the level surfaces
 * \f$\sigma=0\f$ and \f$\sigma=1\f$.
 *
 * The input coordinates are labeled \f$(\bar{x},\bar{y},\bar{z})\f$.
 * Let \f$x_0^i = f^i(\bar{x},\bar{y},\bar{z})\f$ be the three coordinates
 * of the 2D surface in 3D space.
 *
 * Now let \f$x_1^i\f$ be a point on the surface of the sphere,
 * constructed so that \f$P^i\f$, \f$x_0^i\f$, and \f$x_1^i\f$ are
 * co-linear.  In particular, \f$x_1^i\f$ is determined by the
 * equation
 *
 * \f{align} x_1^i = P^i + (x_0^i - P^i) \lambda,\f}
 *
 * where \f$\lambda\f$ is a scalar factor that depends on \f$x_0^i\f$ and
 * that can be computed by solving a quadratic equation.  This quadratic
 * equation is derived by demanding that \f$x_1^i\f$ lies on the sphere:
 *
 * \f{align}
 * |P^i + (x_0^i - P^i) \lambda - C^i |^2 - R^2 = 0,
 * \f}
 *
 * where \f$|A^i|^2\f$ means \f$\delta_{ij} A^i A^j\f$.
 *
 * The quadratic equation, Eq. (2),
 * takes the usual form \f$a\lambda^2+b\lambda+c=0\f$,
 * with
 *
 * \f{align*}
 *  a &= |x_0^i-P^i|^2,\\
 *  b &= 2(x_0^i-P^i)(P^j-C^j)\delta_{ij},\\
 *  c &= |P^i-C^i|^2 - R^2.
 * \f}
 *
 * Now assume that \f$x_0^i\f$ lies on a level surface defined
 * by some scalar function \f$\sigma(\bar{x}^i)=0\f$,
 * where \f$\sigma\f$ is normalized so that \f$\sigma=1\f$ on the sphere.
 * Then, once \f$\lambda\f$ has been computed and \f$x_1^i\f$ has
 * been determined, the map is given by
 *
 * \f{align}x^i = x_0^i + (x_1^i - x_0^i) \sigma(\bar{x}^i).\f}
 *
 * Note that classes using FocallyLiftedMap will place restrictions on
 * \f$P^i\f$, \f$C^i\f$, \f$x_0^i\f$, and \f$R\f$.  For example, we
 * demand that \f$P^i\f$ does not lie on either the \f$\sigma=0\f$ or
 * \f$\sigma=1\f$ surfaces depicted in the right panel of the figure,
 * and we demand that the \f$\sigma=0\f$ and \f$\sigma=1\f$ surfaces do
 * not intersect; otherwise the map is singular.
 *
 * Also note that the quadratic Eq. (2) typically has more than one
 * root, corresponding to two intersections of the sphere.  The
 * boolean parameter `source_is_between_focus_and_target` that is
 * passed into the constructor of `FocallyLiftedMap` is used to choose the
 * appropriate root, or to error if a suitable
 * root is not found. `source_is_between_focus_and_target` should be
 * true if the source point lies between the projection point
 * \f$P^i\f$ and the sphere.  `source_is_between_focus_and_target` is
 * known only by each particular `CoordinateMap` that uses
 * `FocallyLiftedMap`.
 *
 * ### Jacobian
 *
 * Differentiating Eq. (1) above yields
 *
 * \f{align}
 * \frac{\partial x_1^i}{\partial x_0^j} = \lambda \delta^i_j +
 *  (x_0^i - P^i) \frac{\partial \lambda}{\partial x_0^j}.
 * \f}
 *
 * and differentiating Eq. (2) and then inserting Eq. (1) yields
 *
 * \f{align}
 * \frac{\partial\lambda}{\partial x_0^j} &=
 * \lambda^2 \frac{C_j - x_1^j}{|x_1^i - P^i|^2
 * + (x_1^i - P^i)(P_i - C_{i})}.
 * \f}
 *
 * The Jacobian can be found by differentiating Eq. (3) above and using the
 * chain rule, recognizing that \f$x_1^i\f$ depends on \f$\bar{x}^i\f$ only
 * through its dependence on \f$x_0^i\f$; this is because there is no explicit
 * dependence on \f$\bar{x}^i\f$ in Eq. (1) (which determines \f$x_1^i\f$)
 * or Eq. (2) (which determines \f$\lambda\f$).
 *
 * \f{align}
 * \frac{\partial x^i}{\partial \bar{x}^j} &=
 * \sigma \frac{\partial x_1^i}{\partial x_0^k}
 * \frac{\partial x_0^k}{\partial \bar{x}^j} +
 * (1-\sigma)\frac{\partial x_0^i}{\partial \bar{x}^j}
 * + \frac{\partial \sigma}{\partial \bar{x}^j} (x_1^i - x_0^i),
 * \nonumber \\
 * &= (1-\sigma+\lambda\sigma) \frac{\partial x_0^i}{\partial \bar{x}^j} +
 * \sigma (x_0^i - P^i) \frac{\partial \lambda}{\partial x_0^k}
 * \frac{\partial x_0^k}{\partial \bar{x}^j}
 * + \frac{\partial \sigma}{\partial \bar{x}^j} (x_1^i - x_0^i),
 * \f}
 * where in the last line we have substituted Eq. (4).
 *
 * The class defined by the `InnerMap` template parameter
 * provides the function `deriv_sigma`, which returns
 * \f$\partial \sigma/\partial \bar{x}^j\f$, and the function `jacobian`,
 * which returns \f$\partial x_0^k/\partial \bar{x}^j\f$.
 *
 * ### Inverse map.
 *
 * Given \f$x^i\f$, we wish to compute \f$\bar{x}^i\f$.
 *
 * We first find the coordinates \f$x_0^i\f$ that lie on the 2-surface
 * and are defined such that \f$P^i\f$, \f$x_0^i\f$,
 * and \f$x^i\f$ are co-linear.
 * See the right panel of the above figure.
 * \f$x_0^i\f$ is determined by the equation
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda},\f}
 *
 * where \f$\tilde{\lambda}\f$ is a scalar factor that depends on
 * \f$x^i\f$ and is determined by the class defined by the `InnerMap` template
 * parameter.  `InnerMap`
 * provides a function `lambda_tilde` that takes \f$x^i\f$ and
 * \f$P^i\f$ as arguments and returns \f$\tilde{\lambda}\f$ (or
 * a default-constructed `std::optional` if the appropriate
 * \f$\tilde{\lambda}\f$ cannot be
 * found; this default-constructed value indicates that the point \f$x^i\f$ is
 * outside the range of the map).
 *
 * Now consider the coordinates \f$x_1^i\f$ that lie on the sphere and
 * are defined such that \f$P^i\f$, \f$x_1^i\f$, and \f$x^i\f$ are
 * co-linear.  See the right panel of the figure.
 * \f$x_1^i\f$ is determined by the equation
 *
 * \f{align} x_1^i = P^i + (x^i - P^i) \bar{\lambda},\f}
 *
 * where \f$\bar{\lambda}\f$ is a scalar factor that depends on \f$x^i\f$ and
 * is the solution of a quadratic
 * that is derived by demanding that \f$x_1^i\f$ lies on the sphere:
 *
 * \f{align}
 * |P^i + (x^i - P^i) \bar{\lambda} - C^i |^2 - R^2 = 0.
 * \f}
 *
 * Eq. (9) is a quadratic equation that
 * takes the usual form \f$a\bar{\lambda}^2+b\bar{\lambda}+c=0\f$,
 * with
 *
 * \f{align*}
 *  a &= |x^i-P^i|^2,\\
 *  b &= 2(x^i-P^i)(P^j-C^j)\delta_{ij},\\
 *  c &= |P^i-C^i|^2 - R^2.
 * \f}
 *
 * Note that we don't actually need to compute \f$x_1^i\f$. Instead, we
 * can determine \f$\sigma\f$ by the relation (obtained by solving Eq. (3)
 * for \f$\sigma\f$ and then inserting Eqs. (7) and (8) to eliminate
 * \f$x_1^i\f$ and \f$x_0^i\f$)
 *
 * \f{align}
 * \sigma = \frac{\tilde{\lambda}-1}{\tilde{\lambda}-\bar{\lambda}}.
 * \f}
 *
 * The denominator of Eq. (10) is nonzero for nonsingular maps:
 * From Eqs. (7) and (8), \f$\bar{\lambda}=\tilde{\lambda}\f$ means
 * that \f$x_1^i=x_0^i\f$, which means that
 * the \f$\sigma=0\f$ and \f$\sigma=1\f$ surfaces intersect, i.e.
 * the map is singular.
 *
 * Once we have \f$x_0^i\f$ and \f$\sigma\f$, the point
 * \f$(\bar{x},\bar{y},\bar{z})\f$ is uniquely determined by `InnerMap`.
 * The `inverse` function of `InnerMap` takes \f$x_0^i\f$ and \f$\sigma\f$
 * as arguments, and returns
 * \f$(\bar{x},\bar{y},\bar{z})\f$, or a default-constructed `std::optional`
 * if \f$x_0^i\f$ or
 * \f$\sigma\f$ are outside the range of the map.
 *
 * #### Root polishing
 *
 * The inverse function described above will sometimes have errors that
 * are noticeably larger than roundoff.  Therefore we apply a single
 * Newton-Raphson iteration to refine the result of the inverse map:
 * Suppose we are given \f$x^i\f$, and we have computed \f$\bar{x}^i\f$
 * by the above procedure.  We then correct \f$\bar{x}^i\f$ by adding
 *
 * \f{align}
 * \delta \bar{x}^i = \left(x^j - x^j(\bar{x})\right)
 * \frac{\partial \bar{x}^i}{\partial x^j},
 * \f}
 *
 * where \f$x^j(\bar{x})\f$ is the result of applying the forward map
 * to \f$\bar{x}^i\f$ and \f$\partial \bar{x}^i/\partial x^j\f$ is the
 * inverse jacobian.
 *
 * ### Inverse jacobian
 *
 * We write the inverse Jacobian as
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial x^j} =
 * \frac{\partial \bar{x}^i}{\partial x_0^k}
 * \frac{\partial x_0^k}{\partial x^j}
 * + \frac{\partial \bar{x}^i}{\partial \sigma}
 *   \frac{\partial \sigma}{\partial x^j},
 * \f}
 *
 * where we have recognized that \f$\bar{x}^i\f$ depends both on
 * \f$x_0^k\f$ (the corresponding point on the 2-surface) and on
 * \f$\sigma\f$ (encoding the distance away from the 2-surface).
 *
 * We now evaluate Eq. (12). The `InnerMap` class provides a function
 * `inv_jacobian` that returns \f$\partial \bar{x}^i/\partial x_0^k\f$
 * (where \f$\sigma\f$ is held fixed), and a function `dxbar_dsigma`
 * that returns \f$\partial \bar{x}^i/\partial \sigma\f$ (where
 * \f$x_0^i\f$ is held fixed).  The factor \f$\partial x_0^j/\partial
 * x^i\f$ can be computed by differentiating Eq. (7), which yields
 *
 * \f{align}
 * \frac{\partial x_0^j}{\partial x^i} &= \tilde{\lambda} \delta_i^j
 * + \frac{x_0^j-P^j}{\tilde{\lambda}}
 *   \frac{\partial\tilde{\lambda}}{\partial x^i},
 * \f}
 *
 * where \f$\partial \tilde{\lambda}/\partial x^i\f$ is provided by
 * the `deriv_lambda_tilde` function of `InnerMap`.  Note that for
 * nonsingular maps there is no worry that \f$\tilde{\lambda}\f$ is
 * zero in the denominator of the second term of Eq. (13); if
 * \f$\tilde{\lambda}=0\f$ then \f$x_0^i=P^i\f$ by Eq. (7), and therefore
 * the map is singular.
 *
 * To evaluate the remaining unknown factor in Eq. (12),
 * \f$\partial \sigma/\partial x^j\f$,
 * note that [combining Eqs. (1), (7), and (8)]
 * \f$\bar{\lambda}=\tilde{\lambda}\lambda\f$.
 * Therefore Eq. (10) is equivalent to
 *
 * \f{align}
 * \sigma &= \frac{\tilde{\lambda}-1}{\tilde{\lambda}(1-\lambda)}.
 * \f}
 *
 * Differentiating this expression yields
 *
 * \f{align}
 * \frac{\partial \sigma}{\partial x^i} &=
 * \frac{\partial \sigma}{\partial \lambda}
 * \frac{\partial \lambda}{\partial x_0^j}
 * \frac{\partial x_0^j}{\partial x^i}
 * + \frac{\partial \sigma}{\partial \tilde\lambda}
 *   \frac{\partial \tilde\lambda}{\partial x^i}\\
 * &=
 * \frac{\sigma}{1-\lambda}
 * \frac{\partial \lambda}{\partial x_0^j}
 * \frac{\partial x_0^j}{\partial x^i}
 * +
 * \frac{1}{\tilde{\lambda}^2(1-\lambda)}
 * \frac{\partial \tilde\lambda}{\partial x^i},
 * \f}
 *
 * where the second factor in the first term can be evaluated using
 * Eq. (5), the third factor in the first term can be evaluated using
 * Eq. (13), and the second factor in the second term is provided by
 * `InnerMap`s function `deriv_lambda_tilde`.
 *
 */
template <typename InnerMap>
class FocallyLiftedMap {
 public:
  static constexpr size_t dim = 3;
  FocallyLiftedMap(const std::array<double, 3>& center,
                   const std::array<double, 3>& proj_center, double radius,
                   bool source_is_between_focus_and_target, InnerMap inner_map);

  FocallyLiftedMap() = default;
  ~FocallyLiftedMap() = default;
  FocallyLiftedMap(FocallyLiftedMap&&) = default;
  FocallyLiftedMap(const FocallyLiftedMap&) = default;
  FocallyLiftedMap& operator=(const FocallyLiftedMap&) = default;
  FocallyLiftedMap& operator=(FocallyLiftedMap&&) = default;

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
  friend bool operator==<InnerMap>(const FocallyLiftedMap<InnerMap>& lhs,
                                   const FocallyLiftedMap<InnerMap>& rhs);
  std::array<double, 3> center_{}, proj_center_{};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  bool source_is_between_focus_and_target_;
  InnerMap inner_map_;
};
template <typename InnerMap>
bool operator!=(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs);
}  // namespace domain::CoordinateMaps
