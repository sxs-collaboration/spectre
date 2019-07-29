// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct SomeNumber : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "SomeNumber"; }
};

struct TestValue : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "TestValue"; }
};

struct AddTheNumber {
  using argument_tags = tmpl::list<SomeNumber>;
  using return_tags = tmpl::list<TestValue>;
  static void apply(const gsl::not_null<int*> test_value,
                    const int& some_number) noexcept {
    *test_value += some_number;
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 tmpl::list<TestValue, SomeNumber>>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<::Actions::MutateApply<AddTheNumber>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Actions.MutateApply", "[Unit][Actions]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(&runner, 0,
                                                             {1, 3});

  runner.set_phase(Metavariables::Phase::Testing);

  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component, TestValue>(runner, 0) == 4);
}
