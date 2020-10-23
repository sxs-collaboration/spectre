// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Observe.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace NonlinearSolver::newton_raphson::detail {
template <typename Metavariables, typename FieldsTag>
struct InitializeResidualMonitor;
}  // namespace NonlinearSolver::newton_raphson::detail
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>,
                 Convergence::Tags::Criteria<OptionsGroup>,
                 NonlinearSolver::Tags::SufficientDecrease<OptionsGroup>,
                 NonlinearSolver::Tags::MaxGlobalizationSteps<OptionsGroup>>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<::Actions::SetupDataBox,
                     InitializeResidualMonitor<FieldsTag, OptionsGroup>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
              NonlinearSolver::observe_detail::Registration<OptionsGroup>>>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void initialize(
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) noexcept {}

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
  using residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<residual_tag>>;
  using prev_residual_magnitude_square_tag =
      NonlinearSolver::Tags::Globalization<residual_magnitude_square_tag>;

 public:
  using simple_tags =
      db::AddSimpleTags<residual_magnitude_square_tag,
                        initial_residual_magnitude_tag,
                        NonlinearSolver::Tags::StepLength<OptionsGroup>,
                        prev_residual_magnitude_square_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN());
    return std::make_tuple(std::move(box), true);
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
