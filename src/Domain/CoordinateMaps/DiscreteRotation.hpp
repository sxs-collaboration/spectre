// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 * \brief A CoordinateMap that swaps/negates the coordinate
 * axes.
 *
 * Providing an OrientationMap to the constructor allows for
 * the resulting map to have different orientations.
 */
template <size_t VolumeDim>
class DiscreteRotation {
 public:
  static constexpr size_t dim = VolumeDim;

  explicit DiscreteRotation(
      OrientationMap<VolumeDim> orientation = OrientationMap<VolumeDim>{});
  ~DiscreteRotation() = default;
  DiscreteRotation(const DiscreteRotation&) = default;
  DiscreteRotation(DiscreteRotation&&) = default;  // NOLINT
  DiscreteRotation& operator=(const DiscreteRotation&) = default;
  DiscreteRotation& operator=(DiscreteRotation&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, VolumeDim> operator()(
      const std::array<T, VolumeDim>& source_coords) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, VolumeDim>> inverse(
      const std::array<double, VolumeDim>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame> jacobian(
      const std::array<T, VolumeDim>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame> inv_jacobian(
      const std::array<T, VolumeDim>& source_coords) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p);  // NOLINT

  bool is_identity() const { return is_identity_; }

 private:
  friend bool operator==(const DiscreteRotation& lhs,
                         const DiscreteRotation& rhs) {
    return lhs.orientation_ == rhs.orientation_ and
           lhs.is_identity_ == rhs.is_identity_;
  }

  OrientationMap<VolumeDim> orientation_{};
  bool is_identity_ = false;
};

template <size_t VolumeDim>
inline bool operator!=(const CoordinateMaps::DiscreteRotation<VolumeDim>& lhs,
                       const CoordinateMaps::DiscreteRotation<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace CoordinateMaps
}  // namespace domain
