// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Block.hpp"

#include <cstddef>
#include <ios>
#include <memory>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <typeinfo>
#include <utility>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

template <size_t VolumeDim>
Block<VolumeDim>::Block(
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, Frame::Inertial,
                                              VolumeDim>>&& stationary_map,
    const size_t id,
    DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors,
    DirectionMap<VolumeDim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
        external_boundary_conditions) noexcept
    : stationary_map_(std::move(stationary_map)),
      id_(id),
      neighbors_(std::move(neighbors)) {
  ASSERT(external_boundary_conditions.empty() or
             external_boundary_conditions.size() ==
                 2 * VolumeDim - neighbors_.size(),
         "Block " << id
                  << " must have either no boundary conditions or one boundary "
                     "condition "
                     "for each external boundary. Here "
                     "external_boundary_conditions.size()="
                  << external_boundary_conditions.size()
                  << " but VolumeDim= " << VolumeDim
                  << " and neighbors_.size()=" << neighbors_.size());
  // Loop over Directions to search which Directions were not set to neighbors_,
  // set these Directions to external_boundaries_.
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    if (neighbors_.find(direction) == neighbors_.end()) {
      ASSERT(external_boundary_conditions.empty() or
                 external_boundary_conditions.contains(direction),
             "Specifying boundary conditions but the external direction "
                 << direction << " in block " << id_
                 << " does not have a specified boundary condition.");
      if (external_boundary_conditions.contains(direction)) {
        external_boundary_conditions_[direction] =
            std::move(external_boundary_conditions.at(direction));
      } else {
        // Fill with a nullptr for now so we don't break existing code.
        external_boundary_conditions_[direction] = nullptr;
        // We cannot enable this error message until boundary conditions are
        // enabled everywhere in the code, unfortunately.
        // ERROR("Couldn't find boundary condition for external boundary "
        //       << direction);
      }
      external_boundaries_.emplace(direction);
    }
  }
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>&
Block<VolumeDim>::stationary_map() const noexcept {
  ASSERT(stationary_map_ != nullptr,
         "The stationary map is set to nullptr and so cannot be retrieved. "
         "This is because the domain is time-dependent and so there are two "
         "maps: the Logical to Grid map and the Grid to Inertial map.");
  return *stationary_map_;
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Logical, Frame::Grid, VolumeDim>&
Block<VolumeDim>::moving_mesh_logical_to_grid_map() const noexcept {
  ASSERT(moving_mesh_grid_map_ != nullptr,
         "The moving mesh Logical to Grid map is set to nullptr and so cannot "
         "be retrieved. This is because the domain is time-independent and so "
         "only the stationary map exists.");
  return *moving_mesh_grid_map_;
}

template <size_t VolumeDim>
const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
Block<VolumeDim>::moving_mesh_grid_to_inertial_map() const noexcept {
  ASSERT(moving_mesh_inertial_map_ != nullptr,
         "The moving mesh Grid to Inertial map is set to nullptr and so cannot "
         "be retrieved. This is because the domain is time-independent and so "
         "only the stationary map exists.");
  return *moving_mesh_inertial_map_;
}

template <size_t VolumeDim>
void Block<VolumeDim>::inject_time_dependent_map(
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
        moving_mesh_inertial_map) noexcept {
  ASSERT(stationary_map_ != nullptr,
         "Cannot inject time-dependent map into a block that already has a "
         "time-dependent map.");
  moving_mesh_inertial_map_ = std::move(moving_mesh_inertial_map);
  moving_mesh_grid_map_ = stationary_map_->get_to_grid_frame();
  stationary_map_ = nullptr;
}

template <size_t VolumeDim>
void Block<VolumeDim>::pup(PUP::er& p) noexcept {
  p | stationary_map_;
  p | moving_mesh_grid_map_;
  p | moving_mesh_inertial_map_;
  p | id_;
  p | neighbors_;
  p | external_boundaries_;
  p | external_boundary_conditions_;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Block<VolumeDim>& block) noexcept {
  os << "Block " << block.id() << ":\n";
  os << "Neighbors: " << block.neighbors() << '\n';
  os << "External boundaries: " << block.external_boundaries() << '\n';
  os << "Is time dependent: " << std::boolalpha << block.is_time_dependent();
  return os;
}

template <size_t VolumeDim>
bool operator==(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept {
  // Since the external boundary conditions are all abstract base classes we
  // can't compare them, but we check that the typeid matches between LHS and
  // RHS.
  return lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
         lhs.external_boundaries() == rhs.external_boundaries() and
         alg::all_of(
             rhs.external_boundaries(),
             [&lhs, &rhs](const Direction<VolumeDim>& dir) noexcept {
               return lhs.external_boundary_conditions().contains(dir) and
                      rhs.external_boundary_conditions().contains(dir) and
                      typeid(lhs.external_boundary_conditions().at(dir)) ==
                          typeid(rhs.external_boundary_conditions().at(dir));
             }) and
         lhs.is_time_dependent() == rhs.is_time_dependent() and
         (lhs.is_time_dependent()
              ? (lhs.moving_mesh_logical_to_grid_map() ==
                     rhs.moving_mesh_logical_to_grid_map() and
                 lhs.moving_mesh_grid_to_inertial_map() ==
                     rhs.moving_mesh_grid_to_inertial_map())
              : lhs.stationary_map() == rhs.stationary_map());
}

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                          \
  template class Block<GET_DIM(data)>;                                  \
  template std::ostream& operator<<(std::ostream& os,                   \
                                    const Block<GET_DIM(data)>& block); \
  template bool operator==(const Block<GET_DIM(data)>& lhs,             \
                           const Block<GET_DIM(data)>& rhs) noexcept;   \
  template bool operator!=(const Block<GET_DIM(data)>& lhs,             \
                           const Block<GET_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
