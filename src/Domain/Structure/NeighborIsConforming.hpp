// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class OrientationMap;
namespace domain {
enum class Topology : uint8_t;
}  // namespace domain
/// \endcond

namespace domain {
/// \ingroup ComputationalDomainGroup
/// \brief Returns whether or not neighboring Blocks (or Elements) have
/// conforming block logical coordinates on their interface
///
/// \details Block logical coordinates are considered to be conforming if they
/// are identical or related by a discrete rotation (i.e. a valid non-aligned
/// OrientationMap).  It is a requirement that neighboring Blocks be conforming
/// if their oriented topologies are the same in the interface dimensions.
///
/// \note If neighboring Elements are conforming, they can exchange boundary
/// data via either copy (if they have the same h- and p-refinement) or
/// projection (if they don't), taking into account the discrete rotation if
/// necessary. If the neighbors are not conforming, boundary data will need to
/// be interpolated.
template <size_t VolumeDim>
bool neighbor_is_conforming(
    const std::array<Topology, VolumeDim>& self_topologies,
    const std::array<Topology, VolumeDim>& neighbor_topologies,
    const Direction<VolumeDim>& direction_to_neighbor,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);
}  // namespace domain
