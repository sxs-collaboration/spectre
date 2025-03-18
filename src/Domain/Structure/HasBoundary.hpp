// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>

/// \cond
namespace domain {
enum class Topology : uint8_t;
}  // namespace domain
enum class Side : uint8_t;
/// \endcond

namespace domain {
/// \brief Whether or not a Topology has a boundary on a given Side
///
/// \note the boundary can either be an internal (i.e. an interface between
/// neighboring Elements or Blocks) or external boundary
bool has_boundary(domain::Topology topology, Side side);
}  // namespace domain
