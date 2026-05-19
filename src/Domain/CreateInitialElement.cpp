// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CreateInitialElement.hpp"

#include <unordered_set>
#include <utility>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Block.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/NeighborIsConforming.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
SegmentId boundary_segment_id(const size_t refinement_level, const Side side) {
  if (side == Side::Lower) {
    return {refinement_level, 0_st};
  }
  ASSERT(side == Side::Upper, "Invalid side: " << side);
  return {refinement_level, two_to_the(refinement_level) - 1};
}

std::vector<SegmentId> valid_transverse_ids_conforming(
    const SegmentId& oriented_self_id, const size_t neighbor_refinement_level,
    const size_t self_block_id, const size_t neighbor_block_id) {
  const size_t self_refinement_level = oriented_self_id.refinement_level();
  if (self_refinement_level == neighbor_refinement_level) {
    return std::vector{oriented_self_id};
  }
  if (self_refinement_level == neighbor_refinement_level + 1) {
    return std::vector{oriented_self_id.id_of_parent()};
  }
  if (self_refinement_level + 1 != neighbor_refinement_level) {
    ERROR("Block " << self_block_id << " with refinement level "
                   << self_refinement_level << " and neighbor block "
                   << neighbor_block_id << " with refinement level "
                   << neighbor_refinement_level << " differ by more than one.");
  }
  return std::vector{oriented_self_id.id_of_child(Side::Lower),
                     oriented_self_id.id_of_child(Side::Upper)};
}

std::vector<SegmentId> valid_transverse_ids_nonconforming(
    const size_t neighbor_refinement_level) {
  std::vector<SegmentId> result(two_to_the(neighbor_refinement_level));
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = SegmentId(neighbor_refinement_level, i);
  }
  return result;
}

std::unordered_set<ElementId<1>> neighbor_ids(
    const std::array<std::vector<SegmentId>, 1>& valid_segment_ids,
    const size_t neighbor_block_id, const size_t grid_index) {
  ASSERT(valid_segment_ids[0].size() == 1,
         "Cannot have more than one neighbor in one dimension.");
  return std::unordered_set{ElementId<1>{
      neighbor_block_id, std::array{valid_segment_ids[0][0]}, grid_index}};
}

std::unordered_set<ElementId<2>> neighbor_ids(
    const std::array<std::vector<SegmentId>, 2>& valid_segment_ids,
    const size_t neighbor_block_id, const size_t grid_index) {
  std::unordered_set<ElementId<2>> result;
  for (const auto& xi_segment : valid_segment_ids[0]) {
    for (const auto& eta_segment : valid_segment_ids[1]) {
      result.emplace(neighbor_block_id, std::array{xi_segment, eta_segment},
                     grid_index);
    }
  }
  return result;
}

std::unordered_set<ElementId<3>> neighbor_ids(
    const std::array<std::vector<SegmentId>, 3>& valid_segment_ids,
    const size_t neighbor_block_id, const size_t grid_index) {
  std::unordered_set<ElementId<3>> result;
  for (const auto& xi_segment : valid_segment_ids[0]) {
    for (const auto& eta_segment : valid_segment_ids[1]) {
      for (const auto& zeta_segment : valid_segment_ids[2]) {
        result.emplace(neighbor_block_id,
                       std::array{xi_segment, eta_segment, zeta_segment},
                       grid_index);
      }
    }
  }
  return result;
}
}  // namespace

namespace domain {
template <size_t VolumeDim>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id,
    const std::vector<Block<VolumeDim>>& blocks,
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) {
  const auto& block = blocks[element_id.block_id()];
  const auto& neighbors_of_block = block.neighbors();
  const auto segment_ids = element_id.segment_ids();

  const auto compute_element_neighbors_in_other_block =
      [&block, &blocks, &initial_refinement_levels, &neighbors_of_block,
       &segment_ids, grid_index = element_id.grid_index()](
          const Direction<VolumeDim>& direction) {
        const auto& block_neighbors = neighbors_of_block.at(direction);
        Neighbors<VolumeDim> element_neighbors{
            std::unordered_set<ElementId<VolumeDim>>{},
            block_neighbors.orientations(), block_neighbors.are_conforming()};

        if (block_neighbors.size() == 1) {
          const size_t neighbor_block_id = *(block_neighbors.begin());
          const auto& orientation =
              block_neighbors.orientation(neighbor_block_id);
          const auto direction_from_neighbor =
              orientation(direction).opposite();
          std::array<std::vector<SegmentId>, VolumeDim> valid_segment_ids;
          if (neighbor_is_conforming(block.topologies(),
                                     blocks[neighbor_block_id].topologies(),
                                     direction, orientation)) {
            const auto oriented_segment_ids = orientation(segment_ids);
            for (size_t d = 0; d < VolumeDim; ++d) {
              const size_t level =
                  gsl::at(initial_refinement_levels[neighbor_block_id], d);
              if (d == direction_from_neighbor.dimension()) {
                gsl::at(valid_segment_ids, d) = std::vector{
                    boundary_segment_id(level, direction_from_neighbor.side())};
              } else {
                gsl::at(valid_segment_ids, d) = valid_transverse_ids_conforming(
                    gsl::at(oriented_segment_ids, d), level, block.id(),
                    neighbor_block_id);
              }
            }
          } else {
            for (size_t d = 0; d < VolumeDim; ++d) {
              const size_t level =
                  gsl::at(initial_refinement_levels[neighbor_block_id], d);
              if (d == direction_from_neighbor.dimension()) {
                gsl::at(valid_segment_ids, d) = std::vector{
                    boundary_segment_id(level, direction_from_neighbor.side())};
              } else {
                gsl::at(valid_segment_ids, d) =
                    valid_transverse_ids_nonconforming(level);
              }
            }
          }
          element_neighbors.add_ids(
              neighbor_ids(valid_segment_ids, neighbor_block_id, grid_index));
        } else {
          for (const auto& neighbor_block_id : block_neighbors.ids()) {
            const auto& orientation =
                block_neighbors.orientation(neighbor_block_id);
            const auto direction_from_neighbor =
                orientation(direction).opposite();
            std::array<std::vector<SegmentId>, VolumeDim> valid_segment_ids;
            for (size_t d = 0; d < VolumeDim; ++d) {
              const size_t level =
                  gsl::at(initial_refinement_levels[neighbor_block_id], d);
              if (d == direction_from_neighbor.dimension()) {
                gsl::at(valid_segment_ids, d) = std::vector{
                    boundary_segment_id(level, direction_from_neighbor.side())};
              } else {
                gsl::at(valid_segment_ids, d) =
                    valid_transverse_ids_nonconforming(level);
              }
            }
            element_neighbors.add_ids(
                neighbor_ids(valid_segment_ids, neighbor_block_id, grid_index));
          }
        }

        return std::make_pair(direction, std::move(element_neighbors));
      };

  const auto compute_element_neighbor_in_same_block = [&element_id,
                                                       &segment_ids](
                                                          const Direction<
                                                              VolumeDim>&
                                                              direction) {
    auto segment_ids_of_neighbor = segment_ids;
    auto& perpendicular_segment_id =
        gsl::at(segment_ids_of_neighbor, direction.dimension());
    const auto index = perpendicular_segment_id.index();
    perpendicular_segment_id =
        SegmentId(perpendicular_segment_id.refinement_level(),
                  direction.side() == Side::Upper ? index + 1 : index - 1);
    return std::make_pair(
        direction, Neighbors<VolumeDim>(
                       {ElementId<VolumeDim>{element_id.block_id(),
                                             std::move(segment_ids_of_neighbor),
                                             element_id.grid_index()}},
                       OrientationMap<VolumeDim>::create_aligned()));
  };

  typename Element<VolumeDim>::Neighbors_t neighbors_of_element;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto index = gsl::at(segment_ids, d).index();
    // lower neighbor
    const auto lower_direction = Direction<VolumeDim>{d, Side::Lower};
    if (0 == index and 1 == neighbors_of_block.count(lower_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbors_in_other_block(lower_direction));
    } else if (0 != index) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(lower_direction));
    }
    // upper neighbor
    const auto upper_direction = Direction<VolumeDim>{d, Side::Upper};
    if (index == two_to_the(gsl::at(segment_ids, d).refinement_level()) - 1 and
        1 == neighbors_of_block.count(upper_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbors_in_other_block(upper_direction));
    } else if (index !=
               two_to_the(gsl::at(segment_ids, d).refinement_level()) - 1) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(upper_direction));
    }
  }
  const auto topologies = refine_Bn_topology(block.topologies(), element_id);
  return Element<VolumeDim>(ElementId<VolumeDim>(element_id),
                            std::move(neighbors_of_element), topologies);
}
}  // namespace domain

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template Element<DIM(data)> domain::create_initial_element<DIM(data)>( \
      const ElementId<DIM(data)>&, const std::vector<Block<DIM(data)>>&, \
      const std::vector<std::array<size_t, DIM(data)>>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
