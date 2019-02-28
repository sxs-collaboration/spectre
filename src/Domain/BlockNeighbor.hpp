// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template BlockNeighbor.

#pragma once

#include <cstddef>
#include <iosfwd>

#include "Domain/OrientationMap.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// Information about the neighbor of a host Block in a particular direction.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class BlockNeighbor {
 public:
  BlockNeighbor() = default;

  /// Construct with the Id and orientation of the neighbor relative to the
  /// host.
  ///
  /// \param id the Id of the neighbor.
  /// \param orientation This OrientationMap takes objects in the logical
  /// coordinate frame of the host Block and maps them to the logical
  /// coordinate frame of the neighbor Block.
  BlockNeighbor(size_t id, OrientationMap<VolumeDim> orientation) noexcept;
  ~BlockNeighbor() = default;
  BlockNeighbor(const BlockNeighbor<VolumeDim>& neighbor) = default;
  BlockNeighbor(BlockNeighbor<VolumeDim>&&) noexcept = default;
  BlockNeighbor<VolumeDim>& operator=(const BlockNeighbor<VolumeDim>& rhs) =
      default;
  BlockNeighbor<VolumeDim>& operator=(BlockNeighbor<VolumeDim>&&) noexcept =
      default;

  size_t id() const noexcept { return id_; }

  const OrientationMap<VolumeDim>& orientation() const noexcept {
    return orientation_;
  }

  // Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  size_t id_{0};
  OrientationMap<VolumeDim> orientation_;
};

/// Output operator for BlockNeighbor.
template <size_t VolumeDim>
std::ostream& operator<<(
    std::ostream& os, const BlockNeighbor<VolumeDim>& block_neighbor) noexcept;

template <size_t VolumeDim>
bool operator==(const BlockNeighbor<VolumeDim>& lhs,
                const BlockNeighbor<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
bool operator!=(const BlockNeighbor<VolumeDim>& lhs,
                const BlockNeighbor<VolumeDim>& rhs) noexcept;
