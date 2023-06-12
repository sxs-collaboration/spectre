// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Actions/Goto.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Label1;
struct Label2;

struct Counter : db::SimpleTag {
  using type = size_t;
};

struct HasConverged : db::SimpleTag {
  using type = bool;
};

struct HasConvergedCompute : HasConverged, db::ComputeTag {
  using argument_tags = tmpl::list<Counter>;
  static void function(const gsl::not_null<bool*> result,
                       const size_t counter) {
    *result = (counter >= 2);
  }
  using return_type = bool;
  using base = HasConverged;
};

struct Increment {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Counter>(
        [](const gsl::not_null<size_t*> counter) { (*counter)++; },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// [component]
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using repeat_until_phase_action_list = tmpl::flatten<
      tmpl::list<Actions::RepeatUntil<HasConverged, tmpl::list<Increment>>,
                 Parallel::Actions::TerminatePhase>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<Counter>,
                                 db::AddComputeTags<HasConvergedCompute>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::Goto<Label1>, Actions::Label<Label2>,
                     Actions::Label<Label1>, Actions::Goto<Label2>>>,
      Parallel::PhaseActions<Parallel::Phase::Execute,
                             repeat_until_phase_action_list>>;
};
// [component]

// [metavariables]
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

};
// [metavariables]
}  // namespace

// [test case]
SPECTRE_TEST_CASE("Unit.Parallel.GotoAction", "[Unit][Parallel][Actions]") {
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(&runner, 0,
                                                             {size_t{0}});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.force_next_action_to_be<component, Actions::Label<Label1>>(0);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) == 3);

  runner.force_next_action_to_be<component, Actions::Goto<Label1>>(0);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) == 2);

  runner.force_next_action_to_be<component, Actions::Goto<Label2>>(0);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) == 1);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Execute);
  while (not ActionTesting::get_terminate<component>(runner, 0)) {
    runner.next_action<component>(0);
  }
  CHECK(ActionTesting::get_databox_tag<component, HasConverged>(runner, 0));
  CHECK(ActionTesting::get_databox_tag<component, Counter>(runner, 0) == 2);

  // Test zero iterations of the `RepeatUntil` loop
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Execute);
  runner.next_action<component>(0);
  CHECK(runner.get_next_action_index<component>(0) ==
        tmpl::index_of<typename component::repeat_until_phase_action_list,
                       Parallel::Actions::TerminatePhase>::value);
  // Make sure `Increment` was not called for this situation where the
  // condition is already fulfilled at the start.
  CHECK(ActionTesting::get_databox_tag<component, Counter>(runner, 0) == 2);
}
// [test case]
