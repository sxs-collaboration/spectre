// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/PausePhase.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Testing, tmpl::list<Parallel::Actions::PausePhase>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Actions.PausePhase",
                  "[Unit][Parallel][Actions]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_terminate<component>(runner, 0));
}
