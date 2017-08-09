// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Orientation.

#pragma once

#include <array>
#include <iosfwd>
#include <memory>

#include "Domain/Direction.hpp"
#include "Domain/SegmentId.hpp"

/*!
 * \ingroup DomainCreators
 * \brief Allows one to construct a custom Orientation between two Blocks.
 *
 * \usage The user provides `2*VolumeDim` Directions, encoding
 * the correspondence between the directions in each block.
 * \tparam VolumeDim the dimension of the blocks.
 */
template <size_t VolumeDim>
class Orientation {
 public:
  Orientation() = default;
  Orientation<VolumeDim>(
      std::array<Direction<VolumeDim>, VolumeDim> mapped_directions);
  Orientation<VolumeDim>(
      const std::array<Direction<VolumeDim>, VolumeDim>& directions_in_host,
      const std::array<Direction<VolumeDim>, VolumeDim>&
          directions_in_neighbor);
  ~Orientation() = default;
  Orientation(const Orientation&) = default;
  Orientation& operator=(const Orientation&) = default;
  Orientation(Orientation&& /*rhs*/) noexcept = default;
  Orientation& operator=(Orientation&& /*rhs*/) noexcept = default;

  /// True when mapped(Direction) == Direction
  bool is_aligned() const noexcept { return is_aligned_; }

  /// The corresponding dimension in the neighbor.
  size_t mapped(const size_t dim) const noexcept {
    return mapped_directions_[dim].dimension();
  }

  /// The corresponding direction in the neighbor.
  Direction<VolumeDim> mapped(const Direction<VolumeDim>& direction) const {
    return direction.side() == Side::Upper
               ? mapped_directions_[direction.dimension()]
               : mapped_directions_[direction.dimension()].opposite();
  }

  /// The corresponding SegmentIds in the neighbor.
  std::array<SegmentId, VolumeDim> mapped(
      const std::array<SegmentId, VolumeDim>& segmentIds) const;

  template <size_t VolumeDimLocal>
  friend bool operator==(const Orientation<VolumeDimLocal>& lhs,
                         const Orientation<VolumeDimLocal>& rhs) noexcept;

  /// Serialization for Charm++
  void pup(PUP::er& p);  // NOLINT

 private:
  std::array<Direction<VolumeDim>, VolumeDim> mapped_directions_;
  bool is_aligned_ = true;
};

/// Output operator for Orientation.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Orientation<VolumeDim>& orientation);

template <size_t VolumeDim>
bool operator==(const Orientation<VolumeDim>& lhs,
                const Orientation<VolumeDim>& rhs) noexcept {
  return (lhs.mapped_directions_ == rhs.mapped_directions_ and
          lhs.is_aligned_ == rhs.is_aligned_);
}

template <size_t VolumeDim>
bool operator!=(const Orientation<VolumeDim>& lhs,
                const Orientation<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}
