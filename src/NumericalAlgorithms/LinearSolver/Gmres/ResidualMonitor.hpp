// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "IO/Observer/Actions.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/Observe.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace gmres_detail {
template <typename Metavariables>
struct InitializeResidualMonitor;
}  // namespace gmres_detail
}  // namespace LinearSolver
namespace Convergence {
struct Criteria;
}  // namespace Convergence
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename Metavariables>
struct ResidualMonitor {
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr OptionString help = {"Verbosity"};
    static type default_value() { return ::Verbosity::Quiet; }
  };

  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using options =
      tmpl::list<Verbosity, LinearSolver::Tags::ConvergenceCriteria>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<tmpl::append<
      typename InitializeResidualMonitor<Metavariables>::simple_tags,
      typename InitializeResidualMonitor<Metavariables>::compute_tags>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const ::Verbosity& verbosity,
      const Convergence::Criteria& convergence_criteria) noexcept {
    Parallel::simple_action<InitializeResidualMonitor<Metavariables>>(
        Parallel::get_parallel_component<ResidualMonitor>(
            *(global_cache.ckLocalBranch())),
        verbosity, convergence_criteria);

    const auto initial_observation_id = observers::ObservationId(
        db::item_type<LinearSolver::Tags::IterationId>{0},
        typename LinearSolver::observe_detail::ObservationType{});
    Parallel::simple_action<
        observers::Actions::RegisterSingletonWithObserverWriter>(
        Parallel::get_parallel_component<ResidualMonitor>(
            *(global_cache.ckLocalBranch())),
        initial_observation_id);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};

template <typename Metavariables>
struct InitializeResidualMonitor {
 private:
  using fields_tag = typename Metavariables::system::fields_tag;
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId>;
  using orthogonalization_history_tag =
      db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory,
                         fields_tag>;

 public:
  using simple_tags = db::AddSimpleTags<
      ::Tags::Verbosity, LinearSolver::Tags::ConvergenceCriteria,
      residual_magnitude_tag, initial_residual_magnitude_tag,
      LinearSolver::Tags::IterationId, orthogonalization_iteration_id_tag,
      orthogonalization_history_tag>;
  using compute_tags =
      db::AddComputeTags<LinearSolver::Tags::HasConvergedCompute<fields_tag>>;

  template <typename... InboxTags, typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(
      const db::DataBox<tmpl::list<>>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const ::Verbosity& verbosity,
      const Convergence::Criteria& convergence_criteria) noexcept {
    auto box = db::create<simple_tags, compute_tags>(
        verbosity, convergence_criteria,
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        db::item_type<LinearSolver::Tags::IterationId>{0},
        db::item_type<orthogonalization_iteration_id_tag>{0},
        DenseMatrix<double>{2, 1, 0.});
    return std::make_tuple(std::move(box));
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
