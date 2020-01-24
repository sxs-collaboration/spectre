// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "IO/Observer/Actions.hpp"
#include "Informer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
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
namespace LinearSolver {
namespace gmres_detail {
template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor;
}  // namespace gmres_detail
}  // namespace LinearSolver
namespace Convergence {
struct Criteria;
}  // namespace Convergence
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Verbosity<OptionsGroup>,
                 LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<InitializeResidualMonitor<FieldsTag, OptionsGroup>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              LinearSolver::observe_detail::Registration>>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ResidualMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor {
 private:
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId<OptionsGroup>>;
  using orthogonalization_history_tag =
      db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory,
                         fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        LinearSolver::Tags::HasConvergedCompute<fields_tag, OptionsGroup>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeResidualMonitor,
            db::AddSimpleTags<residual_magnitude_tag,
                              initial_residual_magnitude_tag,
                              LinearSolver::Tags::IterationId<OptionsGroup>,
                              orthogonalization_iteration_id_tag,
                              orthogonalization_history_tag>,
            compute_tags>(std::move(box),
                          // The `InitializeResidualMagnitude` action populates
                          // these tags with initial values
                          std::numeric_limits<double>::signaling_NaN(),
                          std::numeric_limits<double>::signaling_NaN(),
                          std::numeric_limits<size_t>::max(),
                          std::numeric_limits<size_t>::max(),
                          DenseMatrix<double>{}),
        true);
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
