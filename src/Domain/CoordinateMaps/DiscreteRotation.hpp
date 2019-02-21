// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/TypeTraits.hpp"

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

  explicit DiscreteRotation(OrientationMap<VolumeDim> orientation =
                                OrientationMap<VolumeDim>{}) noexcept;
  ~DiscreteRotation() = default;
  DiscreteRotation(const DiscreteRotation&) = default;
  DiscreteRotation(DiscreteRotation&&) noexcept = default;  // NOLINT
  DiscreteRotation& operator=(const DiscreteRotation&) = default;
  DiscreteRotation& operator=(DiscreteRotation&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, VolumeDim> operator()(
      const std::array<T, VolumeDim>& source_coords) const noexcept;

  boost::optional<std::array<double, VolumeDim>> inverse(
      const std::array<double, VolumeDim>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame> jacobian(
      const std::array<T, VolumeDim>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame> inv_jacobian(
      const std::array<T, VolumeDim>& source_coords) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return is_identity_; }

 private:
  friend bool operator==(const DiscreteRotation& lhs,
                         const DiscreteRotation& rhs) noexcept {
    return lhs.orientation_ == rhs.orientation_ and
           lhs.is_identity_ == rhs.is_identity_;
  }

  OrientationMap<VolumeDim> orientation_{};
  bool is_identity_ = false;
};

template <size_t VolumeDim>
inline bool operator!=(
    const CoordinateMaps::DiscreteRotation<VolumeDim>& lhs,
    const CoordinateMaps::DiscreteRotation<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace CoordinateMaps
}  // namespace domain
