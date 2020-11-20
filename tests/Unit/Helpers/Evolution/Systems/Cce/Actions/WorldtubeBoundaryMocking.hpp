// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
/// \cond
namespace {  // NOLINT
struct test_metavariables;
template <typename Metavariables>
struct mock_characteristic_evolution;
}  // namespace
namespace Actions {
namespace {  // NOLINT
template <typename BoundaryComponent, typename EvolutionComponent>
struct MockBoundaryComputeAndSendToEvolution;
}  // namespace
}  // namespace Actions
/// \endcond

template <typename Metavariables>
struct mock_h5_worldtube_boundary {
  using component_being_mocked = H5WorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<Actions::BoundaryComputeAndSendToEvolution<
          H5WorldtubeBoundary<Metavariables>,
          mock_characteristic_evolution<test_metavariables>>>;
  using with_these_simple_actions =
      tmpl::list<Actions::MockBoundaryComputeAndSendToEvolution<
          H5WorldtubeBoundary<Metavariables>,
          mock_characteristic_evolution<test_metavariables>>>;

  using initialize_action_list =
      tmpl::list<::Actions::SetupDataBox,
                 Actions::InitializeH5WorldtubeBoundary<
                     typename Metavariables::cce_boundary_communication_tags>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve, tmpl::list<>>>;
};

template <typename Metavariables>
struct mock_gh_worldtube_boundary {
  using component_being_mocked = GhWorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<Actions::BoundaryComputeAndSendToEvolution<
          GhWorldtubeBoundary<Metavariables>,
          mock_characteristic_evolution<test_metavariables>>>;
  using with_these_simple_actions =
      tmpl::list<Actions::MockBoundaryComputeAndSendToEvolution<
          GhWorldtubeBoundary<Metavariables>,
          mock_characteristic_evolution<test_metavariables>>>;

  using initialize_action_list =
      tmpl::list<::Actions::SetupDataBox,
                 Actions::InitializeGhWorldtubeBoundary<
                     typename Metavariables::cce_boundary_communication_tags>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve, tmpl::list<>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
    phase_dependent_action_list>;
};
}  // namespace Cce
