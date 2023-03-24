// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/NewNeighborIds.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
bool overlapping_within_one_level(const SegmentId& segment1,
                                  const SegmentId& segment2) {
  if (0 == segment1.refinement_level()) {
    return (2 > segment2.refinement_level());
  }
  if (0 == segment2.refinement_level()) {
    return (2 > segment1.refinement_level());
  }
  return (segment1 == segment2 or segment1.id_of_parent() == segment2 or
          segment1 == segment2.id_of_parent());
}

}  // namespace

namespace amr {
template <size_t VolumeDim>
std::unordered_set<ElementId<VolumeDim>> new_neighbor_ids(
    const ElementId<VolumeDim>& my_id, const Direction<VolumeDim>& direction,
    const Neighbors<VolumeDim>& previous_neighbors_in_direction,
    const std::unordered_map<ElementId<VolumeDim>,
                             std::array<amr::Flag, VolumeDim>>&
        previous_neighbors_amr_flags) {
  std::unordered_set<ElementId<VolumeDim>> new_neighbors_in_direction;

  const OrientationMap<VolumeDim>& orientation_map_from_me_to_neighbors =
      previous_neighbors_in_direction.orientation();
  const Direction<VolumeDim> direction_to_me_in_neighbor_frame =
      orientation_map_from_me_to_neighbors(direction.opposite());

  // Only one neighbor in 1D in a given direction
  if constexpr (VolumeDim == 1) {
    const ElementId<1>& previous_neighbor_id =
        *(previous_neighbors_in_direction.ids().begin());
    const amr::Flag neighbor_flag =
        previous_neighbors_amr_flags.at(previous_neighbor_id)[0];
    SegmentId previous_segment_id = previous_neighbor_id.segment_ids()[0];
    SegmentId new_segment_id =
        (amr::Flag::Join == neighbor_flag
             ? previous_segment_id.id_of_parent()
             : (amr::Flag::Split == neighbor_flag
                    ? previous_segment_id.id_of_child(
                          direction_to_me_in_neighbor_frame.side())
                    : previous_segment_id));
    new_neighbors_in_direction.emplace(
        previous_neighbor_id.block_id(),
        std::array<SegmentId, 1>({{new_segment_id}}));
    return new_neighbors_in_direction;
  }

  const size_t dim_of_direction_to_me_in_neighbor_frame =
      direction_to_me_in_neighbor_frame.dimension();
  const std::array<SegmentId, VolumeDim> my_segment_ids_in_neighbor_frame =
      orientation_map_from_me_to_neighbors(my_id.segment_ids());

  for (const auto& previous_neighbor_id :
       previous_neighbors_in_direction.ids()) {
    // a neighbor_segment_id is valid if it touches me in the normal direction
    // (which is in dim_of_direction_to_me_in_neighbor_frame) or overlaps with
    // me in the transverse directions
    std::array<std::vector<SegmentId>, VolumeDim> valid_neighbor_segment_ids;
    // it is possible that a previous neighbor (or its children) will not be
    // a neighbor of me (Note: the previous_neighbors may be those of my parent
    // or children if I am the result of h-refinement)
    bool there_is_no_neighbor = false;
    const auto neighbor_segment_ids = previous_neighbor_id.segment_ids();
    for (size_t d = 0; d < VolumeDim; ++d) {
      const amr::Flag neighbor_flag =
          previous_neighbors_amr_flags.at(previous_neighbor_id)[d];
      const SegmentId neighbor_segment_id = gsl::at(neighbor_segment_ids, d);
      if (dim_of_direction_to_me_in_neighbor_frame == d) {
        // This is the normal direction.  I know my previous neighbor touched
        // me so there is exactly one valid segment.
        gsl::at(valid_neighbor_segment_ids, d)
            .push_back(
                amr::Flag::Join == neighbor_flag
                    ? neighbor_segment_id.id_of_parent()
                    : (amr::Flag::Split == neighbor_flag
                           ? neighbor_segment_id.id_of_child(
                                 direction_to_me_in_neighbor_frame.side())
                           : neighbor_segment_id));
      } else {
        // This is a transverse direction.  Since we keep refinement levels
        // within one, there can be 0 to 2 valid segments.
        if (amr::Flag::Join == neighbor_flag) {
          if (overlapping_within_one_level(
                  neighbor_segment_id.id_of_parent(),
                  gsl::at(my_segment_ids_in_neighbor_frame, d))) {
            gsl::at(valid_neighbor_segment_ids, d)
                .push_back(neighbor_segment_id.id_of_parent());
          }
        } else if (amr::Flag::Split == neighbor_flag) {
          if (overlapping_within_one_level(
                  neighbor_segment_id.id_of_child(Side::Lower),
                  gsl::at(my_segment_ids_in_neighbor_frame, d))) {
            gsl::at(valid_neighbor_segment_ids, d)
                .push_back(neighbor_segment_id.id_of_child(Side::Lower));
          }
          if (overlapping_within_one_level(
                  neighbor_segment_id.id_of_child(Side::Upper),
                  gsl::at(my_segment_ids_in_neighbor_frame, d))) {
            gsl::at(valid_neighbor_segment_ids, d)
                .push_back(neighbor_segment_id.id_of_child(Side::Upper));
          }
        } else {
          if (overlapping_within_one_level(
                  neighbor_segment_id,
                  gsl::at(my_segment_ids_in_neighbor_frame, d))) {
            gsl::at(valid_neighbor_segment_ids, d)
                .push_back(neighbor_segment_id);
          }
        }  // if-elseif-else on neighbor_flag

        if (gsl::at(valid_neighbor_segment_ids, d).empty()) {
          there_is_no_neighbor = true;
        }
      }  // if-else on normal vs transverse dimension

      if (there_is_no_neighbor) {
        // No need to loop over the remaining dimensions
        break;
      }
    }  // loop over dimensions

    if (there_is_no_neighbor) {
      // This previous neighbor produced no new neighbors, continue with the
      // next neighbor
      continue;
    }

    for (const auto segment_id_xi : valid_neighbor_segment_ids[0]) {
      if constexpr (VolumeDim > 1) {
        for (const auto segment_id_eta : valid_neighbor_segment_ids[1]) {
          if constexpr (VolumeDim == 2) {
            new_neighbors_in_direction.emplace(
                previous_neighbor_id.block_id(),
                std::array{segment_id_xi, segment_id_eta},
                previous_neighbor_id.grid_index());
          } else if constexpr (VolumeDim == 3) {
            for (const auto segment_id_zeta : valid_neighbor_segment_ids[2]) {
              new_neighbors_in_direction.emplace(
                  previous_neighbor_id.block_id(),
                  std::array{segment_id_xi, segment_id_eta, segment_id_zeta},
                  previous_neighbor_id.grid_index());
            }
          }
        }
      }
    }
  }  // loop over previous neighbors

  return new_neighbors_in_direction;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                          \
  template std::unordered_set<ElementId<DIM(data)>> new_neighbor_ids( \
      const ElementId<DIM(data)>& my_id,                              \
      const Direction<DIM(data)>& direction,                          \
      const Neighbors<DIM(data)>& previous_neighbors_in_direction,    \
      const std::unordered_map<ElementId<DIM(data)>,                  \
                               std::array<amr::Flag, DIM(data)>>&     \
          previous_neighbors_amr_flags);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace amr
