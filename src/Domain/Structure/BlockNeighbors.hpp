// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Structure/Neighbors.hpp"

/// \ingroup ComputationalDomainGroup
/// Information about the neighbors of a host Block in a particular direction.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class BlockNeighbors : public Neighbors<VolumeDim, size_t> {
 public:
  using Neighbors<VolumeDim, size_t>::Neighbors;
};
