// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Structure/Neighbors.hpp"

/// \ingroup ComputationalDomainGroup
/// Information about the neighbor of a host Block in a particular direction.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class BlockNeighbor : public Neighbors<VolumeDim, size_t> {
 public:
  using Neighbors<VolumeDim, size_t>::Neighbors;
};
