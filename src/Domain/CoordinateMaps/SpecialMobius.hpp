// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Redistributes gridpoints within the unit sphere.
 * \image html SpecialMobius.png "A sphere with a `mu` of 0.25."
 *
 * \details A special case of the conformal Mobius transformation that
 * maps the unit ball to itself. This map depends on a single
 * parameter, `mu` \f$ = \mu\f$, which is the x-coordinate of the preimage
 * of the origin under this map. This map has the fixed points \f$x=1\f$ and
 * \f$x=-1\f$. The map is singular for \f$\mu=1\f$ but we have found that this
 * map is accurate up to 12 decimal places for values of \f$\mu\f$ up to 0.96.
 *
 * We define the auxiliary variables
 * \f[ r := \sqrt{x^2 + y^2 +z^2}\f]
 * and
 * \f[ \lambda := \frac{1}{1 - 2 x \mu + \mu^2 r^2}\f]
 *
 * The map corresponding to this transformation in cartesian coordinates
 * is then given by:
 *
 * \f[\vec{x}'(x,y,z) =
 * \lambda\begin{bmatrix}
 * x(1+\mu^2) - \mu(1+r^2)\\
 * y(1-\mu^2)\\
 * z(1-\mu^2)\\
 * \end{bmatrix}\f]
 *
 * The inverse map is the same as the forward map with \f$\mu\f$
 * replaced by \f$-\mu\f$.
 *
 * This map is intended to be used only inside the unit sphere.  A
 * point inside the unit sphere maps to another point inside the unit
 * sphere. The map can have undesirable behavior at certain points
 * outside the unit sphere: The map is singular at
 * \f$(x,y,z) = (1/\mu, 0, 0)\f$ (which is outside the unit sphere
 * since \f$|\mu| < 1\f$). Moreover, a point on the \f$x\f$-axis
 * arbitrarily close to the singularity maps to an arbitrarily large
 * value on the \f$\pm x\f$-axis, where the sign depends on which side
 * of the singularity the point is on.
 *
 * A general Mobius transformation is a function on the complex plane, and
 * takes the form \f$ f(z) = \frac{az+b}{cz+d}\f$, where
 * \f$z, a, b, c, d \in \mathbb{C}\f$, and \f$ad-bc\neq 0\f$.
 *
 * The special case used in this map is the function
 * \f$ f(z) = \frac{z - \mu}{1 - z\mu}\f$. This has the desired properties:
 * - The unit disk in the complex plane is mapped to itself.
 *
 * - The x-axis is mapped to itself.
 *
 * - \f$f(\mu) = 0\f$.
 *
 * The three-dimensional version of this map is obtained by rotating the disk
 * in the plane about the x-axis.
 *
 * This map is useful for performing transformations along the x-axis
 * that preserve the unit disk. A concrete example of this is in the BBH
 * domain, where two BBHs with a center-of-mass at x=\f$\mu\f$ can be shifted
 * such that the new center of mass is now located at x=0. Additionally,
 * the spherical shape of the outer wave-zone is preserved and, as a mobius
 * map, the spherical coordinate shapes of the black holes is also preserved.
 */
class SpecialMobius {
 public:
  static constexpr size_t dim = 3;
  explicit SpecialMobius(double mu) noexcept;
  SpecialMobius() = default;
  ~SpecialMobius() = default;
  SpecialMobius(SpecialMobius&&) = default;
  SpecialMobius(const SpecialMobius&) = default;
  SpecialMobius& operator=(const SpecialMobius&) = default;
  SpecialMobius& operator=(SpecialMobius&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  /// Returns boost::none for target_coords outside the unit sphere.
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

  bool is_identity() const noexcept { return is_identity_; }

 private:
  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> mobius_distortion(
      const std::array<T, 3>& coords, double mu) const noexcept;
  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
  mobius_distortion_jacobian(const std::array<T, 3>& coords, double mu) const
      noexcept;
  friend bool operator==(const SpecialMobius& lhs,
                         const SpecialMobius& rhs) noexcept;

  double mu_{std::numeric_limits<double>::signaling_NaN()};
  bool is_identity_{false};
};
bool operator!=(const SpecialMobius& lhs, const SpecialMobius& rhs) noexcept;
}  // namespace CoordinateMaps
}  // namespace domain
