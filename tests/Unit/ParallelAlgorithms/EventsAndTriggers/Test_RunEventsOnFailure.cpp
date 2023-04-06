// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      typename Actions::RunEventsOnFailure::const_global_cache_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Testing, tmpl::list<Actions::RunEventsOnFailure>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::Completion>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers.RunEventsOnFailure",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();

  const std::vector<std::unique_ptr<Event>> events =
      make_vector<std::unique_ptr<Event>>(
          std::make_unique<Events::Completion>());

  using my_component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {serialize_and_deserialize(events)}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);

  CHECK(ActionTesting::get_terminate<my_component>(runner, 0) == true);
}
}  // namespace
