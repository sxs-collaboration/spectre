// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/HasBoundary.hpp"

#include "Domain/Structure/Side.hpp"
#include "Domain/Structure/Topology.hpp"

namespace domain {
bool has_boundary(const domain::Topology topology, const Side side) {
  return topology == Topology::I1 or
         (topology == Topology::B2Radial and side == Side::Upper);
}
}  // namespace domain
