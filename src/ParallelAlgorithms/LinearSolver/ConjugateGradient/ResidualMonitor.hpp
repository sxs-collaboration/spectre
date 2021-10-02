// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::cg::detail {
template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor;
}  // namespace LinearSolver::cg::detail
/// \endcond

namespace LinearSolver::cg::detail {

template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>,
                 Convergence::Tags::Criteria<OptionsGroup>>;
  using metavariables = Metavariables;
  // The actions in `ResidualMonitorActions.hpp` are invoked as simple actions
  // on this component as the result of reductions from the actions in
  // `ElementActions.hpp`. See `LinearSolver::cg::ConjugateGradient` for
  // details.
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<::Actions::SetupDataBox,
                     InitializeResidualMonitor<FieldsTag, OptionsGroup>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              LinearSolver::observe_detail::Registration<OptionsGroup>>>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ResidualMonitor>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename FieldsTag, typename OptionsGroup>
struct InitializeResidualMonitor {
 private:
  using fields_tag = FieldsTag;
  using residual_square_tag = LinearSolver::Tags::MagnitudeSquare<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;

 public:
  using simple_tags =
      tmpl::list<residual_square_tag, initial_residual_magnitude_tag>;
  using compute_tags = tmpl::list<>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    // The `InitializeResidual` action populates these tags with initial values
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN());
    return std::make_tuple(std::move(box), true);
  }
};

}  // namespace LinearSolver::cg::detail
