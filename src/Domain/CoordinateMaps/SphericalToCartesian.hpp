// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Map from spherical to Cartesian coordinates
 *
 * The expected order of the input spherical coordinates is the same as in
 * `domain::CoordinateMaps::Wedge`: [theta, phi, r] in 3D and [r, phi] in 2D.
 *
 * To map a logical cube to a spherical shell consistent with `ylm::Spherepack`,
 * prepend the following maps:
 * - `r`: Any 1D radial map that maps [-1, 1] to [inner_radius, outer_radius],
 *   such as `domain::CoordinateMaps::Interval`.
 * - `theta`: `domain::CoordinateMaps::PolarAngle` to map [-1, 1] to [0, pi]
 *   with the relation $\xi = cos(\theta)$.
 * - `phi`: Affine map [-1, 1] to [0, 2pi), e.g.
 *   `domain::CoordinateMaps::Affine` or `domain::CoordinateMaps::Interval`
 */
template <size_t Dim>
class SphericalToCartesian {
 public:
  static constexpr size_t dim = Dim;
  static_assert(Dim == 2 or Dim == 3,
                "The spherical shell map is implemented in 2D and 3D");

  SphericalToCartesian() = default;
  ~SphericalToCartesian() = default;
  SphericalToCartesian(SphericalToCartesian&&) = default;
  SphericalToCartesian(const SphericalToCartesian&) = default;
  SphericalToCartesian& operator=(const SphericalToCartesian&) = default;
  SphericalToCartesian& operator=(SphericalToCartesian&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords) const;

  std::optional<std::array<double, Dim>> inverse(
      const std::array<double, Dim>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static constexpr bool is_identity() { return false; }

 private:
  // maps between 2D and 3D choices for coordinate axis orientations
  static constexpr size_t radial_coord =
      detail::WedgeCoordOrientation<Dim>::radial_coord;
  static constexpr size_t polar_coord =
      detail::WedgeCoordOrientation<Dim>::polar_coord;
  static constexpr size_t azimuth_coord =
      detail::WedgeCoordOrientation<Dim>::azimuth_coord;
};

template <size_t Dim>
bool operator==(const SphericalToCartesian<Dim>& /*lhs*/,
                const SphericalToCartesian<Dim>& /*rhs*/) {
  return true;
}
template <size_t Dim>
bool operator!=(const SphericalToCartesian<Dim>& lhs,
                const SphericalToCartesian<Dim>& rhs);
}  // namespace domain::CoordinateMaps
