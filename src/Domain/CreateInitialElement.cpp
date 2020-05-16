// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CreateInitialElement.hpp"

#include <unordered_set>
#include <utility>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::Initialization {
template <size_t VolumeDim>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id, const Block<VolumeDim>& block,
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) noexcept {
  const auto& neighbors_of_block = block.neighbors();
  const auto& segment_ids = element_id.segment_ids();

  // Declare two helper lambdas for setting the neighbors of an element
  const auto compute_element_neighbor_in_other_block =
      [&block, &initial_refinement_levels, &neighbors_of_block, &segment_ids](
          const Direction<VolumeDim>& direction) noexcept {
    const auto& block_neighbor = neighbors_of_block.at(direction);
    const auto& orientation = block_neighbor.orientation();
    // TODO(): An illustration of what is happening here would be useful
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
                << " in blocks " << block_neighbor.id() << " and " << block.id()
                << " differ by more than one.");
      }
    }

    // Set the segment in the perpendicular dimension to the segment
    // of the neighboring elements, which must be at one end of the
    // interval.  This sort of refinement never produces multiple
    // neighbors, so we do not set neighbor_is_refined[...].
    {
      auto& perpendicular_segment = gsl::at(segment_ids_of_unrefined_neighbor,
                                            direction_in_neighbor.dimension());
      const size_t neighbor_refinement =
          gsl::at(refinement_of_neighbor, direction_in_neighbor.dimension());
      // direction_in_neighbor points into the element, so we want the
      // segment on the side it is pointing away from.
      if (direction_in_neighbor.side() == Side::Upper) {
        perpendicular_segment = SegmentId(neighbor_refinement, 0);
      } else {
        perpendicular_segment =
            SegmentId(neighbor_refinement, two_to_the(neighbor_refinement) - 1);
      }
    }

    // Consider all possible neighbors by looping over all elements of
    // the product set {Lower, Upper}^VolumeDim, checking whether each
    // actually describes a neighbor.  For dimensions where the
    // interface is not split, we arbitrarily call all neighbors Lower
    // and consider any Upper neighbors invalid.
    std::unordered_set<ElementId<VolumeDim>> neighbor_ids;
    for (IndexIterator<VolumeDim> index(Index<VolumeDim>(2)); index; ++index) {
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
      neighbor_ids.insert({block_neighbor.id(), segment_ids_of_neighbor});
    next_index:;
    }
    return std::make_pair(
        direction, Neighbors<VolumeDim>(std::move(neighbor_ids), orientation));
  };

  const auto compute_element_neighbor_in_same_block = [&element_id,
                                                       &segment_ids](
      const Direction<VolumeDim>& direction) noexcept {
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
                                   std::move(segment_ids_of_neighbor)}}},
            OrientationMap<VolumeDim>{}));
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

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                 \
  template Element<DIM(data)>                                \
  domain::Initialization::create_initial_element<DIM(data)>( \
      const ElementId<DIM(data)>&, const Block<DIM(data)>&,  \
      const std::vector<std::array<size_t, DIM(data)>>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
