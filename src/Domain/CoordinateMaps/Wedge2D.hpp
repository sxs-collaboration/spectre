// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Wedge2D.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Two dimensional map from the logical square to a wedge, which is used
 * by domains which use disks or annuli.
 *
 * \details The coordinate map is constructed by
 * linearly interpolating between a bulged arc which is circumscribed by a
 * circular arc of radius `radius_outer` and a bulged arc which is
 * circumscribed by a circular arc of radius `radius_inner`. These arcs can be
 * made to be straight or circular, based on the value of `circularity_inner`
 * or `circularity_outer`, which can take on values between 0 (for straight)
 * and 1 (for circular). These arcs extend \f$\pi/2\f$ in angle, and can be
 * oriented along the +/- x or y axis. The choice of using either equiangular
 * or equidistant coordinates along the arcs is specifiable with
 * `with_equiangular_map`. The default logical coordinate that points in the
 * angular direction is \f$\eta\f$. We introduce the auxiliary variable
 * \f$\mathrm{H}\f$ which is a function of \f$\eta\f$. If we are using
 * equiangular coordinates, we have:
 *
 * \f[\mathrm{H}(\eta) = \textrm{tan}(\eta\pi/4)\f]
 *
 * With derivative:
 *
 * \f[\mathrm{H}'(\eta) = \frac{\pi}{4}(1+\mathrm{H}^2)\f]
 *
 * If we are using equidistant coordinates, we have:
 *
 * \f[\mathrm{H}(\eta) = \eta\f]
 *
 * with derivative:
 *
 * <center>\f$\mathrm{H}'(\eta) = 1\f$</center>
 *
 * We also define the variable \f$\rho\f$, given by:
 *
 * \f[\rho = \sqrt{1+\mathrm{H}^2}\f]
 *
 * In terms of the the circularity \f$c\f$ and the radius \f$R\f$,
 * the mapping is:
 *
 * \f[\vec{x}(\xi,\eta) =
 * \frac{1}{2}\left\{(1-\xi)\Big[(1-c_{inner})\frac{R_{inner}}{\sqrt 2}
 * + c_{inner}\frac{R_{inner}}{\rho}\Big] +
 * (1+\xi)\Big[(1-c_{outer})\frac{R_{outer}}{\sqrt 2} +c_{outer}
 * \frac{R_{outer}}{\rho}\Big] \right\}\begin{bmatrix}
 * 1\\
 * \mathrm{H}\\
 * \end{bmatrix}\f]
 *
 * We will define the variables \f$T(\xi)\f$ and \f$A(\xi)\f$, the trapezoid
 * and annulus factors: \f[T(\xi) = T_0 + T_1\xi\f] \f[A(\xi) = A_0 +
 * A_1\xi\f]
 * Where \f{align*}T_0 &= \frac{1}{2} \big\{ (1-c_{outer})R_{outer} +
 * (1-c_{inner})R_{inner}\big\}\\
 * T_1 &= \partial_{\xi} T = \frac{1}{2} \big\{ (1-c_{outer})R_{outer} -
 * (1-c_{inner})R_{inner}\big\}\\
 * A_0 &= \frac{1}{2} \big\{ c_{outer}R_{outer} + c_{inner}R_{inner}\big\}\\
 * A_1 &= \partial_{\xi} A = \frac{1}{2} \big\{ c_{outer}R_{outer} -
 * c_{inner}R_{inner}\big\}\f}
 *
 * The map can then be rewritten as:
 * \f[\vec{x}(\xi,\eta) = \left\{\frac{T(\xi)}{\sqrt 2} +
 * \frac{A(\xi)}{\rho}\right\}\begin{bmatrix}
 * 1\\
 * \mathrm{H}\\
 * \end{bmatrix}\f]
 *
 *
 * The Jacobian is: \f[J =
 * \begin{bmatrix}
 * \frac{T_1}{\sqrt 2} + \frac{A_1}{\rho} &
 * \mathrm{H}\mathrm{H}'\frac{A(\xi)}{\rho^3} \\
 * \mathrm{H}\partial_{\xi}x &
 * \mathrm{H}\partial_{\eta}x + \mathrm{H}'x\\
 * \end{bmatrix}
 * \f]
 *
 * The inverse Jacobian is: \f[J^{-1} =
 * \frac{1}{x}\begin{bmatrix}
 * \frac{1}{\partial_{\xi}x}\Big\{
 * \frac{T(\xi)}{\sqrt 2}+\frac{A(\xi)}{\rho^3}
 * \Big\} & \mathrm{H}\frac{1}{\partial_{\xi}x}\frac{A(\xi)}{\rho^3}\\
 * -\mathrm{H}\mathrm{H}'^{-1} & \mathrm{H}'^{-1}\\
 * \end{bmatrix}
 * \f]
 *
 * For a more detailed discussion, see the
 * documentation for Wedge3D, where the default logical coordinate that points
 * in the radial direction is \f$\zeta\f$ (In Wedge2D the logical coordinate
 * that points in the radial direction is \f$\xi\f$), and one may set either
 * of the two other logical coordinates to zero to obtain an equivalent
 * Wedge2D map.
 */

class Wedge2D {
 public:
  static constexpr size_t dim = 2;

  Wedge2D(double radius_inner, double radius_outer, double circularity_inner,
          double circularity_outer, OrientationMap<2> orientation_of_wedge,
          bool with_equiangular_map) noexcept;

  Wedge2D() = default;
  ~Wedge2D() = default;
  Wedge2D(Wedge2D&&) = default;
  Wedge2D& operator=(Wedge2D&&) = default;
  Wedge2D(const Wedge2D&) = default;
  Wedge2D& operator=(const Wedge2D&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 2> operator()(
      const std::array<T, 2>& source_coords) const noexcept;

  /// Returns invalid if \f$x<=0\f$ (for a \f$+x\f$-oriented `Wedge2D`).
  boost::optional<std::array<double, 2>> inverse(
      const std::array<double, 2>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> jacobian(
      const std::array<T, 2>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> inv_jacobian(
      const std::array<T, 2>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  friend bool operator==(const Wedge2D& lhs, const Wedge2D& rhs) noexcept;

  double radius_inner_{};
  double radius_outer_{};
  double circularity_inner_{};
  double circularity_outer_{};
  OrientationMap<2> orientation_of_wedge_{};
  bool with_equiangular_map_ = false;
  double scaled_trapezoid_zero_{std::numeric_limits<double>::signaling_NaN()};
  double annulus_zero_{std::numeric_limits<double>::signaling_NaN()};
  double scaled_trapezoid_rate_{std::numeric_limits<double>::signaling_NaN()};
  double annulus_rate_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Wedge2D& lhs, const Wedge2D& rhs) noexcept;
}  // namespace CoordinateMaps
}  // namespace domain
