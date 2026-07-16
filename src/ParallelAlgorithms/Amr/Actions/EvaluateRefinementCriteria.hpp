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
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/ArrayCollection/SimpleActionOnElement.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "ParallelAlgorithms/Amr/Actions/UpdateAmrDecision.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Policies/EnforcePolicies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
/// \brief Evaluates the refinement criteria in order to set the amr::Info of an
/// Element and sends this information to the neighbors of the Element.
///
/// DataBox:
/// - Uses:
///   * domain::Tags::Element<volume_dim>
///   * amr::Tags::NeighborInfo<volume_dim>
///   * amr::Criteria::Tags::Criteria (from GlobalCache)
///   * amr::Tags::Policies (from GlobalCache)
///   * any tags requested by the refinement criteria
/// - Modifies:
///   * amr::Tags::Info<volume_dim>
///
/// Invokes:
/// - UpdateAmrDecision on all neighboring Element%s
///
/// \details
/// - Evaluates each refinement criteria held by amr::Criteria::Tags::Criteria,
///   and in each dimension selects the amr::Flag with the highest
///   priority (i.e the highest integral value).  If
///   Metavariables::amr::p_refine_only_in_event is true, only h-refinement
///   criteria will be evaluated
/// - If necessary, changes the refinement decision in order to satisfy the
///   amr::Policies
/// - An Element that is splitting in one dimension is not allowed to join
///   in another dimension.  If this is requested by the refinement critiera,
///   the decision to join is changed to do nothing
/// - Checks if any neighbors have sent their AMR decision, and if so, calls
///   amr:::update_amr_decision with the decision of each neighbor in
///   order to see if the current decision needs to be updated
/// - Sends the (possibly updated) decision to all of the neighboring Elements
struct EvaluateRefinementCriteria {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, size_t Dim>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& element_id) {
    if constexpr (Metavariables::amr::keep_coarse_grids) {
      // Only evaluate the criteria on the finest grid. The other elements keep
      // their flags as Undefined and will therefore be skipped by AdjustDomain.
      const bool is_finest_grid =
          db::get<amr::Tags::ChildIds<Dim>>(box).empty();
      if (not is_finest_grid) {
        return;
      }
    }

    constexpr size_t volume_dim = Metavariables::volume_dim;
    auto overall_decision = make_array<volume_dim>(amr::Flag::Undefined);

    // Evaluate criteria on blocks that have AMR enabled, and "do nothing" on
    // the other blocks.
    // Note that blocks that do not have AMR enabled can still be refined to
    // satisfy 2:1 balance. When this happens, they won't coarsen back since "do
    // nothing" has higher priority than coarsening. This behavior can be
    // changed if needed.
    const auto& amr_blocks = db::get<amr::Tags::AmrBlocks<volume_dim>>(box);
    if (not amr_blocks.has_value() or
        alg::found(amr_blocks.value(), element_id.block_id())) {
      using compute_tags =
          tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
              tmpl::at<
                  typename Metavariables::factory_creation::factory_classes,
                  Criterion>,
              detail::get_tags<tmpl::_1>>>>;
      auto observation_box =
          make_observation_box<compute_tags>(make_not_null(&box));

      const auto& refinement_criteria =
          db::get<amr::Criteria::Tags::Criteria>(box);
      for (const auto& criterion : refinement_criteria) {
        const auto refinement_type = criterion->type();
        if (Metavariables::amr::p_refine_only_in_event and
            refinement_type == amr::Criteria::Type::p) {
          continue;
        }
        auto decision = criterion->evaluate(observation_box, cache, element_id);
        if constexpr (Metavariables::amr::p_refine_only_in_event) {
          ASSERT(alg::none_of(decision,
                              [](amr::Flag flag) {
                                return flag == amr::Flag::IncreaseResolution or
                                       flag == amr::Flag::DecreaseResolution;
                              }),
                 "The criterion '"
                     << typeid(*criterion).name()
                     << "' requested p-refinement, but claims to be "
                        "for h-refinement.");
        }

        // Enforce restrictions from the topologies of the Element
        const auto& topologies =
            db::get<domain::Tags::Element<Dim>>(box).topologies();
        if (refinement_type == amr::Criteria::Type::p) {
          enforce_p_refinement_topology_restrictions(make_not_null(&decision),
                                                     topologies);
        } else if (refinement_type == amr::Criteria::Type::h) {
          enforce_h_refinement_topology_restrictions(make_not_null(&decision),
                                                     topologies);
        } else {
          ERROR("Bad refinement type " << refinement_type);
        }
        for (size_t d = 0; d < volume_dim; ++d) {
          overall_decision[d] = std::max(overall_decision[d], decision[d]);
        }
      }
    }  // amr_blocks.contains(element_id.block_id())

    // If no refinement criteria were called, then set flag to do nothing
    for (size_t d = 0; d < volume_dim; ++d) {
      if (overall_decision[d] == amr::Flag::Undefined) {
        overall_decision[d] = amr::Flag::DoNothing;
      }
    }

    const auto& policies = db::get<amr::Tags::Policies>(box);
    amr::enforce_policies(make_not_null(&overall_decision), policies,
                          element_id,
                          get<::domain::Tags::Mesh<volume_dim>>(box));

    // An element cannot join if it is splitting in another dimension.
    // Update the flags now before sending to neighbors as each time
    // a flag is changed by UpdateAmrDecision, it sends the new flags
    // to its neighbors.  So updating now will save some commmunication.
    amr::prevent_element_from_joining_while_splitting(
        make_not_null(&overall_decision));

    // Check if we received any neighbor flags prior to determining our own
    // flags.  If yes, then possible update our flags (e.g. sibling doesn't want
    // to join, maintain 2:1 balance, etc.)
    const auto& my_element = get<::domain::Tags::Element<volume_dim>>(box);
    const auto& my_neighbors_amr_info =
        get<amr::Tags::NeighborInfo<volume_dim>>(box);
    if (not my_neighbors_amr_info.empty()) {
      for (const auto& [neighbor_id, neighbor_amr_info] :
           my_neighbors_amr_info) {
        amr::update_amr_decision(
            make_not_null(&overall_decision), my_element, neighbor_id,
            neighbor_amr_info.flags,
            policies.enforce_two_to_one_balance_in_normal_direction());
      }
    }

    const auto new_mesh = amr::projectors::new_mesh(
        get<::domain::Tags::Mesh<volume_dim>>(box), overall_decision,
        my_element, my_neighbors_amr_info);

    db::mutate<amr::Tags::Info<Metavariables::volume_dim>>(
        [&overall_decision,
         &new_mesh](const gsl::not_null<amr::Info<volume_dim>*> amr_info) {
          amr_info->flags = overall_decision;
          amr_info->new_mesh = new_mesh;
        },
        make_not_null(&box));

    auto& amr_element_array =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const amr::Info<Metavariables::volume_dim>& my_info =
        get<amr::Tags::Info<volume_dim>>(box);
    for (const auto& [direction, neighbors] : my_element.neighbors()) {
      (void)direction;
      for (const auto& neighbor_id : neighbors.ids()) {
        if constexpr (Parallel::is_dg_element_collection_v<ParallelComponent>) {
          const auto neighbor_location = static_cast<int>(
              Parallel::local_synchronous_action<
                  Parallel::Actions::GetItemFromDistributedOject<
                      Parallel::Tags::ElementLocations<volume_dim>>>(
                  Parallel::get_parallel_component<ParallelComponent>(cache))
                  ->at(neighbor_id));
          Parallel::threaded_action<Parallel::Actions::SimpleActionOnElement<
              UpdateAmrDecision, true>>(amr_element_array[neighbor_location],
                                        neighbor_id, element_id, my_info);
        } else {
          Parallel::simple_action<UpdateAmrDecision>(
              amr_element_array[neighbor_id], element_id, my_info);
        }
      }
    }
  }
};
}  // namespace amr::Actions
