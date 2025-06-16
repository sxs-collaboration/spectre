// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Block.hpp"

#include <cstddef>
#include <ios>
#include <memory>
#include <ostream>
#include <pup.h>
#include <typeinfo>
#include <utility>

#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/HasBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
Block<VolumeDim>::Block(
    std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>&& stationary_map,
    const size_t id,
    DirectionMap<VolumeDim, BlockNeighbors<VolumeDim>> neighbors,
    std::string name, std::array<domain::Topology, VolumeDim> topologies)
    : stationary_map_(std::move(stationary_map)),
      id_(id),
      neighbors_(std::move(neighbors)),
      name_(std::move(name)),
      topologies_(std::move(topologies)) {
  // Loop over Directions to search which Directions were not set to neighbors_,
  // set these Directions to external_boundaries_.
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    const bool has_boundary_in_this_direction = has_boundary(
        gsl::at(topologies_, direction.dimension()), direction.side());
    if (neighbors_.find(direction) == neighbors_.end()) {
      if (has_boundary_in_this_direction) {
        external_boundaries_.emplace(direction);
      }
    } else {
      ASSERT(has_boundary_in_this_direction,
             "Cannot specify a neighbor in a direction with no boundary.  "
             "Topologies: "
                 << topologies_ << "; Direction: " << direction);
    }
  }
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial,
                                VolumeDim>&
Block<VolumeDim>::stationary_map() const {
  ASSERT(stationary_map_ != nullptr,
         "The stationary map is set to nullptr and so cannot be retrieved. "
         "This is because the domain is time-dependent and so there are two "
         "maps: the Logical to Grid map and the Grid to Inertial map.");
  return *stationary_map_;
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::BlockLogical, Frame::Grid, VolumeDim>&
Block<VolumeDim>::moving_mesh_logical_to_grid_map() const {
  ASSERT(moving_mesh_logical_to_grid_map_ != nullptr,
         "The moving mesh Logical to Grid map is set to nullptr and so cannot "
         "be retrieved. This is because the domain is time-independent and so "
         "only the stationary map exists.");
  return *moving_mesh_logical_to_grid_map_;
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
Block<VolumeDim>::moving_mesh_grid_to_inertial_map() const {
  ASSERT(moving_mesh_grid_to_inertial_map_ != nullptr,
         "The moving mesh Grid to Inertial map is set to nullptr and so cannot "
         "be retrieved. This is because the domain is time-independent and so "
         "only the stationary map exists.");
  return *moving_mesh_grid_to_inertial_map_;
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>&
Block<VolumeDim>::moving_mesh_grid_to_distorted_map() const {
  ASSERT(
      moving_mesh_grid_to_distorted_map_ != nullptr,
      "The moving mesh Grid to Distorted map is set to nullptr and so cannot "
      "be retrieved. This is because there is no map from the Grid to the "
      "Distorted Frame.");
  return *moving_mesh_grid_to_distorted_map_;
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, VolumeDim>&
Block<VolumeDim>::moving_mesh_distorted_to_inertial_map() const {
  ASSERT(
      moving_mesh_distorted_to_inertial_map_ != nullptr,
      "The moving mesh Distorted to Inertial map is set to nullptr and so "
      "cannot "
      "be retrieved. This is because there is no map from the Distorted to the "
      "Inertial Frame.");
  return *moving_mesh_distorted_to_inertial_map_;
}

template <size_t VolumeDim>
void Block<VolumeDim>::inject_time_dependent_map(
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
        moving_mesh_grid_to_inertial_map,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>>
        moving_mesh_grid_to_distorted_map,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, VolumeDim>>
        moving_mesh_distorted_to_inertial_map) {
  ASSERT(stationary_map_ != nullptr,
         "Cannot inject time-dependent map into a block that already has a "
         "time-dependent map.");
  moving_mesh_grid_to_inertial_map_ =
      std::move(moving_mesh_grid_to_inertial_map);
  moving_mesh_logical_to_grid_map_ = stationary_map_->get_to_grid_frame();
  moving_mesh_grid_to_distorted_map_ =
      std::move(moving_mesh_grid_to_distorted_map);
  moving_mesh_distorted_to_inertial_map_ =
      std::move(moving_mesh_distorted_to_inertial_map);
  stationary_map_ = nullptr;
}

template <size_t VolumeDim>
void Block<VolumeDim>::pup(PUP::er& p) {
  size_t version = 3;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | stationary_map_;
    p | moving_mesh_logical_to_grid_map_;
    p | moving_mesh_grid_to_inertial_map_;
    p | moving_mesh_grid_to_distorted_map_;
    p | moving_mesh_distorted_to_inertial_map_;
    p | id_;
    p | neighbors_;
    p | external_boundaries_;
  }
  if (version >= 1) {
    p | name_;
  }
  if (version == 2) {
    auto basis = make_array<VolumeDim>(Spectral::Basis::Uninitialized);
    p | basis;
  }
  if (version < 3) {
    topologies_ = make_array<VolumeDim>(domain::Topology::I1);
  } else {
    p | topologies_;
  }
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Block<VolumeDim>& block) {
  os << "Block " << block.id() << " (" << block.name() << "):\n";
  os << "Topology: " << block.topologies() << '\n';
  os << "Neighbors: " << block.neighbors() << '\n';
  os << "External boundaries: " << block.external_boundaries() << '\n';
  os << "Is time dependent: " << std::boolalpha << block.is_time_dependent();
  return os;
}

template <size_t VolumeDim>
bool operator==(const Block<VolumeDim>& lhs, const Block<VolumeDim>& rhs) {
  bool blocks_are_equal =
      (lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
       lhs.external_boundaries() == rhs.external_boundaries() and
       lhs.name() == rhs.name() and lhs.topologies() == rhs.topologies() and
       lhs.is_time_dependent() == rhs.is_time_dependent() and
       lhs.has_distorted_frame() == rhs.has_distorted_frame());

  if (lhs.is_time_dependent() and not lhs.has_distorted_frame()) {
    blocks_are_equal =
        blocks_are_equal and (lhs.moving_mesh_logical_to_grid_map() ==
                                  rhs.moving_mesh_logical_to_grid_map() and
                              lhs.moving_mesh_grid_to_inertial_map() ==
                                  rhs.moving_mesh_grid_to_inertial_map());
  } else if (lhs.is_time_dependent() and lhs.has_distorted_frame()) {
    blocks_are_equal = blocks_are_equal and
                       (lhs.moving_mesh_logical_to_grid_map() ==
                        rhs.moving_mesh_logical_to_grid_map()) and
                       (lhs.moving_mesh_grid_to_distorted_map() ==
                            rhs.moving_mesh_grid_to_distorted_map() and
                        lhs.moving_mesh_distorted_to_inertial_map() ==
                            rhs.moving_mesh_distorted_to_inertial_map());
  } else {
    blocks_are_equal =
        blocks_are_equal and (lhs.stationary_map() == rhs.stationary_map());
  }
  return blocks_are_equal;
}

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs, const Block<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                          \
  template class Block<GET_DIM(data)>;                                  \
  template std::ostream& operator<<(std::ostream& os,                   \
                                    const Block<GET_DIM(data)>& block); \
  template bool operator==(const Block<GET_DIM(data)>& lhs,             \
                           const Block<GET_DIM(data)>& rhs);            \
  template bool operator!=(const Block<GET_DIM(data)>& lhs,             \
                           const Block<GET_DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
