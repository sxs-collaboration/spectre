// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

#include "Framework/ActionTesting.hpp"
#include "IO/Observer/ArrayComponentId.hpp"  // IWYU pragma: keep
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"                   // IWYU pragma: keep
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <exception>

namespace {
template <typename Metavariables>
struct observer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags =
      typename observers::Actions::Initialize<Metavariables>::simple_tags;
  using compute_tags =
      typename observers::Actions::Initialize<Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<observers::Actions::Initialize<Metavariables>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<observer_component<Metavariables>>;
  using observed_reduction_data_tags = tmpl::list<>;

  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.IO.Observers.Initialize", "[Unit][Observers]") {
  using obs_component = observer_component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<obs_component>(&runner, 0);
  runner.next_action<obs_component>(0);

  CHECK(
      ActionTesting::get_databox_tag<
          obs_component, observers::Tags::ExpectedContributorsForObservations>(
          runner, 0)
          .empty());
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::ReductionsContributed>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            obs_component, observers::Tags::ContributorsOfTensorData>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<obs_component,
                                       observers::Tags::TensorData>(runner, 0)
            .empty());
}
}  // namespace
