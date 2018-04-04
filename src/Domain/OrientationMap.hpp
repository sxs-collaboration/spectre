// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>


#include "Domain/Direction.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits.hpp"

class SegmentId;
namespace PUP {
class er;
}  // namespace PUP

/*!
 * \ingroup DomainCreatorsGroup
 * \brief A mapping of the logical coordinate axes of a host to the logical
 * coordinate axes of a neighbor of the host.
 * \usage Given a `size_t dimension`, a `Direction`, or a `SegmentId` of the
 * host, an `OrientationMap` will give the corresponding value in the neighbor.
 * \tparam VolumeDim the dimension of the blocks.
 */
template <size_t VolumeDim>
class OrientationMap {
 public:
  /// The default orientation is the identity map on directions.
  /// The bool `is_aligned_` is correspondingly set to `true`.
  OrientationMap();
  explicit OrientationMap(
      std::array<Direction<VolumeDim>, VolumeDim> mapped_directions);
  OrientationMap(
      const std::array<Direction<VolumeDim>, VolumeDim>& directions_in_host,
      const std::array<Direction<VolumeDim>, VolumeDim>&
          directions_in_neighbor);
  ~OrientationMap() = default;
  OrientationMap(const OrientationMap&) = default;
  OrientationMap& operator=(const OrientationMap&) = default;
  OrientationMap(OrientationMap&& /*rhs*/) noexcept = default;
  OrientationMap& operator=(OrientationMap&& /*rhs*/) noexcept = default;

  /// True when mapped(Direction) == Direction
  bool is_aligned() const noexcept { return is_aligned_; }

  /// The corresponding dimension in the neighbor.
  size_t operator()(const size_t dim) const noexcept {
    return gsl::at(mapped_directions_, dim).dimension();
  }

  /// The corresponding direction in the neighbor.
  Direction<VolumeDim> operator()(const Direction<VolumeDim>& direction) const {
    return direction.side() == Side::Upper
               ? gsl::at(mapped_directions_, direction.dimension())
               : gsl::at(mapped_directions_, direction.dimension()).opposite();
  }

  /// The corresponding SegmentIds in the neighbor.
  std::array<SegmentId, VolumeDim> operator()(
      const std::array<SegmentId, VolumeDim>& segmentIds) const;

  /// The corresponding Orientation of the host in the frame of the neighbor.
  OrientationMap<VolumeDim> inverse_map() const noexcept;

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const OrientationMap& lhs,
                         const OrientationMap& rhs) noexcept {
    return (lhs.mapped_directions_ == rhs.mapped_directions_ and
            lhs.is_aligned_ == rhs.is_aligned_);
  }

  std::array<Direction<VolumeDim>, VolumeDim> mapped_directions_;
  bool is_aligned_ = true;
};

/// Output operator for OrientationMap.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const OrientationMap<VolumeDim>& orientation);

template <size_t VolumeDim>
bool operator!=(const OrientationMap<VolumeDim>& lhs,
                const OrientationMap<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \ingroup DomainCreatorsGroup
/// `OrientationMap`s define an active rotation of the logical axes that bring
/// the axes of a host block into alignment with the logical axes of the
/// neighbor block. `discrete_rotation` applies this active rotation on the
/// coordinates as opposed to the axes.
/// For a two-dimensional example, consider a host block and a neighbor block,
/// where the OrientationMap between them is \f$\{-\eta,+\xi\}\f$. A quarter-
/// turn counterclockwise of the host block's logical axes would bring them into
/// alignment with those of the neighbor. That is, after this active rotation,
/// the blocks would be Aligned. Now consider a point A with coordinates
/// (+1.0,-0.5). An active quarter-turn rotation counter-clockwise about the
/// origin, keeping the axes fixed, brings point A into the coordinates
/// (+0.5,+1.0). This is how `discrete_rotation` interprets the
/// `OrientationMap` passed to it.
template <size_t VolumeDim, typename T>
std::array<tt::remove_cvref_wrap_t<T>, VolumeDim> discrete_rotation(
    const OrientationMap<VolumeDim>& rotation,
    std::array<T, VolumeDim> source_coords) noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, VolumeDim> new_coords{};
  for (size_t i = 0; i < VolumeDim; i++) {
    const auto new_direction = rotation(Direction<VolumeDim>(i, Side::Upper));
    gsl::at(new_coords, i) =
        std::move(gsl::at(source_coords, new_direction.dimension()));
    if (new_direction.side() != Side::Upper) {
      gsl::at(new_coords, i) *= -1.0;
    }
  }
  return new_coords;
}
