// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// Actions to control the elliptic AMR algorithm
namespace elliptic::amr::Actions {

/*!
 * \brief Initializes items for the elliptic AMR algorithm projector
 *
 * When projecting these items during h-AMR, they are copied from the parent or
 * children to retain the AMR state.
 */
struct Initialize : tt::ConformsTo<::amr::protocols::Projector> {
 private:
  using iteration_id_tag =
      Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>;
  using num_iterations_tag =
      Convergence::Tags::Iterations<::amr::OptionTags::AmrGroup>;
  using has_converged_tag =
      Convergence::Tags::HasConverged<::amr::OptionTags::AmrGroup>;

 public:  // Initialization mutator
  using simple_tags = tmpl::list<iteration_id_tag, has_converged_tag>;
  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<num_iterations_tag>;
  using mutable_global_cache_tags = tmpl::list<>;
  using return_tags = simple_tags;
  using argument_tags = tmpl::list<num_iterations_tag>;

  static void apply(
      const gsl::not_null<size_t*> amr_iteration_id,
      const gsl::not_null<Convergence::HasConverged*> amr_has_converged,
      const size_t num_iterations) {
    *amr_iteration_id = 0;
    *amr_has_converged = Convergence::HasConverged{num_iterations, 0};
  }

 public:  // amr::protocols::Projector
  // p-refinement
  template <size_t Dim>
  static void apply(
      const gsl::not_null<size_t*> /*amr_iteration_id*/,
      const gsl::not_null<Convergence::HasConverged*> /*amr_has_converged*/,
      const size_t /*num_iterations*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {}

  // h-refinement
  template <typename... ParentTags>
  static void apply(
      const gsl::not_null<size_t*> amr_iteration_id,
      const gsl::not_null<Convergence::HasConverged*> amr_has_converged,
      const size_t /*num_iterations*/,
      const tuples::TaggedTuple<ParentTags...>& parent_items) {
    *amr_iteration_id = get<iteration_id_tag>(parent_items);
    *amr_has_converged = get<has_converged_tag>(parent_items);
  }

  // h-coarsening
  template <size_t Dim, typename... ChildTags>
  static void apply(
      const gsl::not_null<size_t*> amr_iteration_id,
      const gsl::not_null<Convergence::HasConverged*> amr_has_converged,
      const size_t /*num_iterations*/,
      const std::unordered_map<
          ElementId<Dim>, tuples::TaggedTuple<ChildTags...>>& children_items) {
    *amr_iteration_id = get<iteration_id_tag>(children_items.begin()->second);
    *amr_has_converged = get<has_converged_tag>(children_items.begin()->second);
  }
};

/// Increment the AMR iteration ID and determine convergence
struct IncrementIterationId {
 private:
  using iteration_id_tag =
      Convergence::Tags::IterationId<::amr::OptionTags::AmrGroup>;
  using num_iterations_tag =
      Convergence::Tags::Iterations<::amr::OptionTags::AmrGroup>;
  using has_converged_tag =
      Convergence::Tags::HasConverged<::amr::OptionTags::AmrGroup>;

 public:
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t num_iterations = get<num_iterations_tag>(box);
    // Increment AMR iteration id and determine convergence
    db::mutate<iteration_id_tag, has_converged_tag>(
        [&num_iterations](
            const gsl::not_null<size_t*> iteration_id,
            const gsl::not_null<Convergence::HasConverged*> has_converged) {
          ++(*iteration_id);
          *has_converged =
              Convergence::HasConverged{num_iterations, *iteration_id};
        },
        make_not_null(&box));
    // Do some logging
    if (db::get<logging::Tags::Verbosity<::amr::OptionTags::AmrGroup>>(box) >=
        ::Verbosity::Debug) {
      Parallel::printf("%s AMR iteration %zu / %zu\n", element_id,
                       db::get<iteration_id_tag>(box), num_iterations);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// Stop the algorithm if it has converged
struct StopAmr {
 private:
  using has_converged_tag =
      Convergence::Tags::HasConverged<::amr::OptionTags::AmrGroup>;

 public:
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {db::get<has_converged_tag>(box)
                ? Parallel::AlgorithmExecution::Pause
                : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};

}  // namespace elliptic::amr::Actions
