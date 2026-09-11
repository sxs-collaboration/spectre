// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Cce::Actions {
template <typename BoundaryComponent, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution;
}  // namespace Cce::Actions
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace Cce {
namespace Actions {
template <typename Metavariables>
struct MockBoundaryComputeAndSendToEvolution {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex>
    requires(... or
             std::is_same_v<
                 ::Tags::Variables<
                     typename Metavariables::cce_boundary_communication_tags>,
                 DbTags>)
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const TimeStepId& time) {
    times_requested.push_back(time.substep_time());
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::vector<double> times_requested;
};
template <typename Metavariables>
std::vector<double>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    MockBoundaryComputeAndSendToEvolution<Metavariables>::times_requested;
}  // namespace Actions

template <typename Metavariables, typename ComponentBeingMocked,
          typename MockCharacteristicComponent = void>
struct mock_worldtube_boundary {
  using component_being_mocked = ComponentBeingMocked;
  using replace_these_simple_actions = tmpl::conditional_t<
      std::is_void_v<MockCharacteristicComponent>, tmpl::list<>,
      tmpl::list<Actions::BoundaryComputeAndSendToEvolution<
          ComponentBeingMocked, MockCharacteristicComponent>>>;
  using with_these_simple_actions = tmpl::conditional_t<
      std::is_void_v<MockCharacteristicComponent>, tmpl::list<>,
      tmpl::list<
          Actions::MockBoundaryComputeAndSendToEvolution<Metavariables>>>;

  using initialize_action_list =
      tmpl::list<Actions::InitializeWorldtubeBoundary<ComponentBeingMocked>,
                 Parallel::Actions::TerminatePhase>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<>>>;
};
}  // namespace Cce
