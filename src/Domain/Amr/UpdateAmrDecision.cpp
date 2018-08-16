// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/UpdateAmrDecision.hpp"

#include "Domain/Amr/Helpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"    // IWYU pragma: keep
#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/Neighbors.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim>
class OrientationMap;

namespace amr {

template <size_t VolumeDim>
bool update_amr_decision(
    gsl::not_null<std::array<amr::Flag, VolumeDim>*> my_current_amr_flags,
    const Element<VolumeDim>& element, const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::Flag, VolumeDim>& neighbor_amr_flags) noexcept {
  const auto& element_id = element.id();
  bool my_amr_decision_changed = false;
  bool neighbor_found = false;
  std::array<size_t, VolumeDim> my_desired_levels =
      amr::desired_refinement_levels(element_id, *my_current_amr_flags);

  for (const auto& direction_neighbors : element.neighbors()) {
    const Neighbors<VolumeDim>& neighbors_in_dir = direction_neighbors.second;

    if (1 == neighbors_in_dir.ids().count(neighbor_id)) {
      // finding the same neighbor twice (which can happen with periodic
      // domains) is okay, and may be needed when examining a Join
      neighbor_found = true;
      const Direction<VolumeDim> direction_to_neighbor =
          direction_neighbors.first;
      const OrientationMap<VolumeDim>& orientation_of_neighbor =
          neighbors_in_dir.orientation();
      const std::array<size_t, VolumeDim> neighbor_desired_levels =
          amr::desired_refinement_levels_of_neighbor(
              neighbor_id, neighbor_amr_flags, orientation_of_neighbor);

      // update flags if my element wants to be two or more levels
      // coarser than the neighbor in any dimension
      for (size_t d = 0; d < VolumeDim; ++d) {
        if (amr::Flag::Split == gsl::at(*my_current_amr_flags, d) or
            gsl::at(my_desired_levels, d) >=
                gsl::at(neighbor_desired_levels, d)) {
          continue;
        }
        const size_t difference =
            gsl::at(neighbor_desired_levels, d) - gsl::at(my_desired_levels, d);
        ASSERT(difference < 4 and difference > 0,
               "neighbor level = " << gsl::at(neighbor_desired_levels, d)
                                   << ", my level = "
                                   << gsl::at(my_desired_levels, d));
        if (amr::Flag::Join == gsl::at(*my_current_amr_flags, d)) {
          if (3 == difference) {
            // My split neighbor wants to split, so I need to split to keep
            // refinement levels within one.
            gsl::at(*my_current_amr_flags, d) = amr::Flag::Split;
            gsl::at(my_desired_levels, d) += 2;
            my_amr_decision_changed = true;
          } else if (2 == difference) {
            // My split neighbor wants to stay the same, or my neighbor
            // split, so I need to stay the same to keep refinement levels
            // within one.
            gsl::at(*my_current_amr_flags, d) = amr::Flag::DoNothing;
            ++gsl::at(my_desired_levels, d);
            my_amr_decision_changed = true;
          }
        } else {
          if (2 == difference) {
            // my split neighbor wants to split, so I need to split to
            // keep refinement levels within one
            gsl::at(*my_current_amr_flags, d) = amr::Flag::Split;
            ++gsl::at(my_desired_levels, d);
            my_amr_decision_changed = true;
          }
        }
      }

      // update flags if the neighbor is a potential sibling that my element
      // cannot join
      const size_t dimension_of_direction_to_neighbor =
          direction_to_neighbor.dimension();
      if (amr::has_potential_sibling(element_id, direction_to_neighbor) and
          amr::Flag::Join == gsl::at(*my_current_amr_flags,
                                     dimension_of_direction_to_neighbor) and
          (my_desired_levels != neighbor_desired_levels)) {
        gsl::at(*my_current_amr_flags, dimension_of_direction_to_neighbor) =
            amr::Flag::DoNothing;
        ++gsl::at(my_desired_levels, dimension_of_direction_to_neighbor);
        my_amr_decision_changed = true;
      }
    }
  }
  ASSERT(neighbor_found, "Could not find neighbor " << neighbor_id);
  return my_amr_decision_changed;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template bool update_amr_decision(                                         \
      gsl::not_null<std::array<amr::Flag, DIM(data)>*> my_current_amr_flags, \
      const Element<DIM(data)>& element,                                     \
      const ElementId<DIM(data)>& neighbor_id,                               \
      const std::array<amr::Flag, DIM(data)>& neighbor_amr_flags) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
}  // namespace amr
