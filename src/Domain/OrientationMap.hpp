// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template OrientationMap.

#pragma once

#include <array>
#include <iosfwd>
#include <memory>

#include "Domain/Direction.hpp"
#include "Domain/SegmentId.hpp"

/*!
 * \ingroup DomainCreators
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
  void pup(PUP::er& p);  // NOLINT

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
