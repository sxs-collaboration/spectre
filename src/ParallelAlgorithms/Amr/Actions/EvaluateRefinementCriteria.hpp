// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Amr/Actions/UpdateAmrDecision.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace detail {
template <typename Criterion>
struct get_tags {
  using type = typename Criterion::compute_tags_for_observation_box;
};

}  // namespace detail

namespace amr::Actions {
/// \brief Evaluates the refinement criteria in order to set the
/// amr::domain::Flag%s of an Element and sends this information to the
/// neighbors of the Element.
///
/// DataBox:
/// - Uses:
///   * domain::Tags::Element<volume_dim>
///   * amr::domain::Tags::NeighborFlags<volume_dim>
///   * amr::Criteria::Tags::Criteria (from GlobalCache)
///   * any tags requested by the refinement criteria
/// - Modifies:
///   * amr::domain::Tags::Flags<volume_dim>
///
/// Invokes:
/// - UpdateAmrDecision on all neighboring Element%s
///
/// \details
/// - Evaluates each refinement criteria held by amr::Criteria::Tags::Criteria,
///   and in each dimension selects the amr::domain::Flag with the highest
///   priority (i.e the highest integral value).
/// - Checks if any neighbors have sent their AMR decision, and if so, calls
///   amr:::domain::update_amr_decision with the decision of each neighbor in
///   order to see if the current decision needs to be updated
/// - Sends the (possibly updated) decision to all of the neighboring Elements
struct EvaluateRefinementCriteria {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    ElementId<volume_dim> element_id{array_index};
    auto overall_decision =
        make_array<volume_dim>(amr::domain::Flag::Undefined);

    using compute_tags = tmpl::remove_duplicates<tmpl::filter<
        tmpl::flatten<tmpl::transform<
            tmpl::at<typename Metavariables::factory_creation::factory_classes,
                     Criterion>,
            detail::get_tags<tmpl::_1>>>,
        db::is_compute_tag<tmpl::_1>>>;
    auto observation_box = make_observation_box<compute_tags>(box);

    const auto& refinement_criteria =
        db::get<amr::Criteria::Tags::Criteria>(box);
    for (const auto& criterion : refinement_criteria) {
      auto decision = criterion->evaluate(observation_box, cache, array_index);
      for (size_t d = 0; d < volume_dim; ++d) {
        overall_decision[d] = std::max(overall_decision[d], decision[d]);
      }
    }

    // Check if we received any neighbor flags prior to determining our own
    // flags.  If yes, then possible update our flags (e.g. sibling doesn't want
    // to join, maintain 2:1 balance, etc.)
    const auto& my_element = get<::domain::Tags::Element<volume_dim>>(box);
    const auto& my_neighbors_amr_flags =
        get<amr::domain::Tags::NeighborFlags<volume_dim>>(box);
    if (not my_neighbors_amr_flags.empty()) {
      for (const auto& [neighbor_id, neighbor_amr_flags] :
           my_neighbors_amr_flags) {
        amr::domain::update_amr_decision(make_not_null(&overall_decision),
                                         my_element, neighbor_id,
                                         neighbor_amr_flags);
      }
    }

    db::mutate<amr::domain::Tags::Flags<Metavariables::volume_dim>>(
        make_not_null(&box),
        [&overall_decision](
            const gsl::not_null<std::array<amr::domain::Flag, volume_dim>*>
                amr_flags) { *amr_flags = overall_decision; });

    auto& amr_element_array =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const std::array<amr::domain::Flag, Metavariables::volume_dim>& my_flags =
        get<amr::domain::Tags::Flags<volume_dim>>(box);
    for (const auto& [direction, neighbors] : my_element.neighbors()) {
      (void)direction;
      for (const auto& neighbor_id : neighbors.ids()) {
        Parallel::simple_action<UpdateAmrDecision>(
            amr_element_array[neighbor_id], element_id, my_flags);
      }
    }
  }
};
}  // namespace amr::Actions
