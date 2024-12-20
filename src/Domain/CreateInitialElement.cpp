// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CreateInitialElement.hpp"

#include <unordered_set>
#include <utility>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Block.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::Initialization {
template <size_t VolumeDim>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id,
    const std::vector<Block<VolumeDim>>& blocks,
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) {
  const auto& block = blocks[element_id.block_id()];
  const auto& neighbors_of_block = block.neighbors();
  const auto segment_ids = element_id.segment_ids();

  // Declare two helper lambdas for setting the neighbors of an element
  const auto compute_element_neighbor_in_other_block =
      [&block, &blocks, &initial_refinement_levels, &neighbors_of_block,
       &segment_ids, grid_index = element_id.grid_index()](
          const Direction<VolumeDim>& direction) {
        const auto& block_neighbors = neighbors_of_block.at(direction);

        // Handle spherical shells. We assume that spherical shells have no
        // angular h-refinement. That makes the logic here quite simple:
        // The single-element shell is the neighbor of all radial neighbors.
        if (block.geometry() == domain::BlockGeometry::SphericalShell or
            (block_neighbors.size() == 1 and
             blocks[block_neighbors.begin()->id()].geometry() ==
                 domain::BlockGeometry::SphericalShell)) {
#ifdef SPECTRE_DEBUG
          const auto check_angular_refinement =
              [&initial_refinement_levels,
               &direction](const Block<VolumeDim>& test_block) {
                if (test_block.geometry() !=
                    domain::BlockGeometry::SphericalShell) {
                  return;
                }
                const auto& refinement_levels =
                    initial_refinement_levels[test_block.id()];
                for (size_t d = 0; d < VolumeDim; ++d) {
                  if (d == direction.dimension()) {
                    continue;
                  }
                  ASSERT(refinement_levels[d] == 0,
                         "Spherical shells are assumed here to have no angular "
                         "refinement in angular directions.");
                }
              };
          check_angular_refinement(block);
#endif  // SPECTRE_DEBUG
          std::unordered_set<ElementId<VolumeDim>> neighbor_ids;
          for (const auto& block_neighbor : block_neighbors) {
#ifdef SPECTRE_DEBUG
            check_angular_refinement(blocks[block_neighbor.id()]);
#endif  // SPECTRE_DEBUG
            const auto& orientation = block_neighbor.orientation();
            const auto direction_from_neighbor =
                orientation(direction.opposite());
            const auto& refinement_of_neighbor =
                initial_refinement_levels[block_neighbor.id()];
            const size_t index_of_neighbor =
                direction_from_neighbor.side() == Side::Lower
                    ? 0
                    : (two_to_the(refinement_of_neighbor[direction_from_neighbor
                                                             .dimension()]) -
                       1);
            for (const auto& neighbor_id : initial_element_ids(
                     block_neighbor.id(), refinement_of_neighbor, grid_index)) {
              if (neighbor_id.segment_id(direction_from_neighbor.dimension())
                      .index() == index_of_neighbor) {
                neighbor_ids.insert(neighbor_id);
              }
            }
          }
          return std::make_pair(
              direction, Neighbors<VolumeDim>(
                             std::move(neighbor_ids),
                             // TODO: may have to set this orientation
                             OrientationMap<VolumeDim>::create_aligned(),
                             blocks[block_neighbors.begin()->id()].geometry()));
        }

        const auto& block_neighbor = *block_neighbors.begin();
        const auto& orientation = block_neighbor.orientation();
        const auto direction_in_neighbor = orientation(direction);

        // SegmentIds of the current element using the neighbor's axes.
        auto segment_ids_of_unrefined_neighbor = orientation(segment_ids);
        const auto& refinement_of_neighbor =
            initial_refinement_levels[block_neighbor.id()];

        // Check whether each dimension will have multiple neighbors
        // because the neighboring block is more refined.  For dimensions
        // that have only one neighbor, adjust the SegmentId to that of
        // the neighbor if necessary.
        auto neighbor_is_refined = make_array<VolumeDim>(false);
        for (size_t d = 0; d < VolumeDim; ++d) {
          // The refinement difference perpendicular to the interface is
          // not restricted.
          if (d == direction_in_neighbor.dimension()) {
            continue;
          }
          switch (static_cast<int>(gsl::at(refinement_of_neighbor, d)) -
                  static_cast<int>(gsl::at(segment_ids_of_unrefined_neighbor, d)
                                       .refinement_level())) {
            case 0:
              break;
            case 1:
              gsl::at(neighbor_is_refined, d) = true;
              break;
            case -1:
              gsl::at(segment_ids_of_unrefined_neighbor, d) =
                  gsl::at(segment_ids_of_unrefined_neighbor, d).id_of_parent();
              break;
            default:
              ERROR("Refinement levels "
                    << gsl::at(refinement_of_neighbor, d) << " and "
                    << gsl::at(segment_ids_of_unrefined_neighbor, d)
                           .refinement_level()
                    << " in blocks " << block_neighbor.id() << " and "
                    << block.id() << " differ by more than one.");
          }
        }

        // Set the segment in the perpendicular dimension to the segment
        // of the neighboring elements, which must be at one end of the
        // interval.  This sort of refinement never produces multiple
        // neighbors, so we do not set neighbor_is_refined[...].
        {
          auto& perpendicular_segment =
              gsl::at(segment_ids_of_unrefined_neighbor,
                      direction_in_neighbor.dimension());
          const size_t neighbor_refinement = gsl::at(
              refinement_of_neighbor, direction_in_neighbor.dimension());
          // direction_in_neighbor points into the element, so we want the
          // segment on the side it is pointing away from.
          if (direction_in_neighbor.side() == Side::Upper) {
            perpendicular_segment = SegmentId(neighbor_refinement, 0);
          } else {
            perpendicular_segment = SegmentId(
                neighbor_refinement, two_to_the(neighbor_refinement) - 1);
          }
        }

        // Consider all possible neighbors by looping over all elements of
        // the product set {Lower, Upper}^VolumeDim, checking whether each
        // actually describes a neighbor.  For dimensions where the
        // interface is not split, we arbitrarily call all neighbors Lower
        // and consider any Upper neighbors invalid.
        std::unordered_set<ElementId<VolumeDim>> neighbor_ids;
        for (IndexIterator<VolumeDim> index(Index<VolumeDim>(2)); index;
             ++index) {
          std::array<SegmentId, VolumeDim> segment_ids_of_neighbor{};
          for (size_t d = 0; d < VolumeDim; ++d) {
            const auto& unrefined_segment =
                gsl::at(segment_ids_of_unrefined_neighbor, d);
            auto& segment = gsl::at(segment_ids_of_neighbor, d);
            if (not gsl::at(neighbor_is_refined, d)) {
              if ((*index)[d] == 0) {
                segment = unrefined_segment;
              } else {
                goto next_index;
              }
            } else {
              if ((*index)[d] == 0) {
                segment = unrefined_segment.id_of_child(Side::Lower);
              } else {
                segment = unrefined_segment.id_of_child(Side::Upper);
              }
            }
          }
          neighbor_ids.insert(
              {block_neighbor.id(), segment_ids_of_neighbor, grid_index});
        next_index:;
        }
        return std::make_pair(
            direction,
            Neighbors<VolumeDim>(std::move(neighbor_ids), orientation,
                                 blocks[block_neighbor.id()].geometry()));
      };

  const auto compute_element_neighbor_in_same_block =
      [&element_id, &segment_ids,
       geometry = block.geometry()](const Direction<VolumeDim>& direction) {
        auto segment_ids_of_neighbor = segment_ids;
        auto& perpendicular_segment_id =
            gsl::at(segment_ids_of_neighbor, direction.dimension());
        const auto index = perpendicular_segment_id.index();
        perpendicular_segment_id =
            SegmentId(perpendicular_segment_id.refinement_level(),
                      direction.side() == Side::Upper ? index + 1 : index - 1);
        return std::make_pair(
            direction,
            Neighbors<VolumeDim>(
                {{ElementId<VolumeDim>{element_id.block_id(),
                                       std::move(segment_ids_of_neighbor),
                                       element_id.grid_index()}}},
                OrientationMap<VolumeDim>::create_aligned(), geometry));
      };

  typename Element<VolumeDim>::Neighbors_t neighbors_of_element;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto index = gsl::at(segment_ids, d).index();
    // lower neighbor
    const auto lower_direction = Direction<VolumeDim>{d, Side::Lower};
    if (0 == index and 1 == neighbors_of_block.count(lower_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_other_block(lower_direction));
    } else if (0 != index) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(lower_direction));
    }
    // upper neighbor
    const auto upper_direction = Direction<VolumeDim>{d, Side::Upper};
    if (index == two_to_the(gsl::at(segment_ids, d).refinement_level()) - 1 and
        1 == neighbors_of_block.count(upper_direction)) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_other_block(upper_direction));
    } else if (index !=
               two_to_the(gsl::at(segment_ids, d).refinement_level()) - 1) {
      neighbors_of_element.emplace(
          compute_element_neighbor_in_same_block(upper_direction));
    }
  }
  return Element<VolumeDim>(ElementId<VolumeDim>(element_id),
                            std::move(neighbors_of_element));
}
}  // namespace domain::Initialization

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template Element<DIM(data)>                                            \
  domain::Initialization::create_initial_element<DIM(data)>(             \
      const ElementId<DIM(data)>&, const std::vector<Block<DIM(data)>>&, \
      const std::vector<std::array<size_t, DIM(data)>>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
