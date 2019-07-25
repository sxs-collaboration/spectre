// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "ParallelBackend/Actions/Goto.hpp"  // IWYU pragma: keep
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct Label1;
struct Label2;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<Actions::Goto<Label1>, Actions::Label<Label2>,
                 Actions::Label<Label1>, Actions::Goto<Label2>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelBackend.GotoAction",
                  "[Unit][ParallelBackend][Actions]") {
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);

  runner.set_phase(Metavariables::Phase::Testing);
  runner.force_next_action_to_be<component, Actions::Label<Label1>>(0);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) == 3);

  runner.force_next_action_to_be<component, Actions::Goto<Label1>>(0);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) == 2);

  runner.force_next_action_to_be<component, Actions::Goto<Label2>>(0);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) == 1);
}
