// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <utility>

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::gmres::detail {
template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor;
}  // namespace LinearSolver::gmres::detail
/// \endcond

namespace LinearSolver::gmres::detail {

template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Verbosity<OptionsGroup>,
                 LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>>;
  using metavariables = Metavariables;
  // The actions in `ResidualMonitorActions.hpp` are invoked as simple actions
  // on this component as the result of reductions from the actions in
  // `ElementActions.hpp`. See `LinearSolver::gmres::Gmres` for details.
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<InitializeResidualMonitor<FieldsTag, OptionsGroup>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              LinearSolver::observe_detail::Registration<OptionsGroup>>>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ResidualMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor {
 private:
  using fields_tag = FieldsTag;
  using initial_residual_magnitude_tag =
      LinearSolver::Tags::Initial<LinearSolver::Tags::Magnitude<
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;
  using orthogonalization_iteration_id_tag =
      LinearSolver::Tags::Orthogonalization<
          LinearSolver::Tags::IterationId<OptionsGroup>>;
  using orthogonalization_history_tag =
      LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeResidualMonitor,
            db::AddSimpleTags<initial_residual_magnitude_tag,
                              orthogonalization_history_tag>>(
            std::move(box),
            // The `InitializeResidualMagnitude` action populates these tags
            // with initial values
            std::numeric_limits<double>::signaling_NaN(),
            DenseMatrix<double>{}),
        true);
  }
};

}  // namespace LinearSolver::gmres::detail
