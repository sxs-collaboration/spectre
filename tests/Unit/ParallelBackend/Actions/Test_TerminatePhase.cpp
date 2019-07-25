// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "ParallelBackend/Actions/TerminatePhase.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<Parallel::Actions::TerminatePhase>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelBackend.Actions.TerminatePhase",
                  "[Unit][ParallelBackend][Actions]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);

  runner.set_phase(Metavariables::Phase::Testing);

  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_terminate<component>(runner, 0));
}
