// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/NeighborIsConforming.hpp"

#include <array>
#include <cstddef>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/HasBoundary.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
template <size_t VolumeDim>
std::array<domain::Topology, VolumeDim - 1> boundary_topologies(
    const std::array<domain::Topology, VolumeDim>& topologies,
    const Direction<VolumeDim>& direction) {
  const size_t dim = direction.dimension();
  ASSERT(domain::has_boundary(gsl::at(topologies, dim), direction.side()),
         "There is no boundary for topologies "
             << topologies << " in the direction " << direction);
  auto result = all_but_specified_element_of(topologies, dim);
  if (gsl::at(topologies, dim) == domain::Topology::B2Radial) {
    for (size_t d = 0; d < VolumeDim - 1; ++d) {
      if (gsl::at(result, d) == domain::Topology::B2Angular) {
        gsl::at(result, d) = domain::Topology::S1;
      }
    }
  }
  return result;
}
}  // namespace

namespace domain {
template <size_t VolumeDim>
bool neighbor_is_conforming(
    const std::array<Topology, VolumeDim>& self_topologies,
    const std::array<Topology, VolumeDim>& neighbor_topologies,
    const Direction<VolumeDim>& direction_to_neighbor,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  if constexpr (VolumeDim > 1) {
    const auto self_boundary_topologies =
        boundary_topologies(self_topologies, direction_to_neighbor);
    if (orientation_of_neighbor.is_aligned()) {
      const auto neighbor_boundary_topologies = boundary_topologies(
          neighbor_topologies, direction_to_neighbor.opposite());
      return self_boundary_topologies == neighbor_boundary_topologies;
    } else {
      for (size_t d = 0; d < VolumeDim; ++d) {
        if (d == direction_to_neighbor.dimension()) {
          continue;
        }
        if (orientation_of_neighbor(Direction<VolumeDim>{d, Side::Upper}) ==
            Direction<VolumeDim>::self()) {
          return false;
        }
      }
      const auto neighbor_boundary_topologies = boundary_topologies(
          orientation_of_neighbor.permute_from_neighbor(neighbor_topologies),
          direction_to_neighbor);
      return self_boundary_topologies == neighbor_boundary_topologies;
    }
  }
  return true;
}

template bool neighbor_is_conforming(
    const std::array<Topology, 1>& self_topologies,
    const std::array<Topology, 1>& neighbor_topologies,
    const Direction<1>& direction_to_neighbor,
    const OrientationMap<1>& orientation_of_neighbor);
template bool neighbor_is_conforming(
    const std::array<Topology, 2>& self_topologies,
    const std::array<Topology, 2>& neighbor_topologies,
    const Direction<2>& direction_to_neighbor,
    const OrientationMap<2>& orientation_of_neighbor);
template bool neighbor_is_conforming(
    const std::array<Topology, 3>& self_topologies,
    const std::array<Topology, 3>& neighbor_topologies,
    const Direction<3>& direction_to_neighbor,
    const OrientationMap<3>& orientation_of_neighbor);
}  // namespace domain
