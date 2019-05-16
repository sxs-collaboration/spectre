// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct SomeNumber : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "SomeNumber"; }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<SomeNumber>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Actions.SetData", "[Unit][Actions]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(&runner, 0, {0});

  runner.set_phase(Metavariables::Phase::Testing);

  ActionTesting::simple_action<component,
                               Actions::SetData<tmpl::list<SomeNumber>>>(
      make_not_null(&runner), 0, tuples::TaggedTuple<SomeNumber>{3});
  CHECK(ActionTesting::get_databox_tag<component, SomeNumber>(runner, 0) == 3);
}
