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
 * \brief Map from the logical interval $\xi \in [-1, 1]$ to the polar angle
 * $\theta \in [0, \pi]$, where $\xi = -\cos(\theta)$ as used by
 * `ylm::Spherepack`.
 *
 * \warning The Jacobian of this map is ill-defined at the endpoints
 * $\xi = -1, 1$, which correspond to the poles $\theta = 0, \pi$.
 * Therefore it is typically used with Gauss-Legendre collocation points.
 */
class PolarAngle {
 public:
  static constexpr size_t dim = 1;

  PolarAngle() = default;
  ~PolarAngle() = default;
  PolarAngle(PolarAngle&&) = default;
  PolarAngle(const PolarAngle&) = default;
  PolarAngle& operator=(const PolarAngle&) = default;
  PolarAngle& operator=(PolarAngle&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> operator()(
      const std::array<T, 1>& source_coords) const;

  std::optional<std::array<double, 1>> inverse(
      const std::array<double, 1>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jacobian(
      const std::array<T, 1>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jacobian(
      const std::array<T, 1>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static constexpr bool is_identity() { return false; }
};

inline bool operator==(const PolarAngle& /*lhs*/, const PolarAngle& /*rhs*/) {
  return true;
}
bool operator!=(const PolarAngle& lhs, const PolarAngle& rhs);
}  // namespace domain::CoordinateMaps
