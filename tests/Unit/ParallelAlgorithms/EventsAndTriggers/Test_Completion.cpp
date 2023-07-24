// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::Completion>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.EventsAndTriggers.Completion",
                  "[Unit][ParallelAlgorithms]") {
  const auto completion =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "Completion");
  CHECK(not completion->needs_evolved_variables());

  using my_component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<my_component>(&runner, 0);

  const ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>> box{};
  auto& cache = ActionTesting::cache<my_component>(runner, 0);
  my_component* const component_ptr = nullptr;
  completion->run(box, cache, 0, component_ptr, {"Unused", -1.0});

  CHECK(ActionTesting::get_terminate<my_component>(runner, 0));
}
