// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveAdaptiveSteppingDiagnostics.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace observers::Actions {
struct ContributeReductionData;
}  // namespace observers::Actions

namespace {
struct MockContributeReductionData {
  using ReductionData =
      tmpl::wrap<tmpl::front<Events::ObserveAdaptiveSteppingDiagnostics::
                                 observed_reduction_data_tags>,
                 Parallel::ReductionData>;
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    ReductionData reduction_data;
  };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::optional<Results> results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    observers::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    ReductionData&& reduction_data) {
    if (results) {
      CHECK(results->observation_id == observation_id);
      CHECK(results->subfile_name == subfile_name);
      CHECK(results->reduction_names == reduction_names);
      results->reduction_data.combine(std::move(reduction_data));
    } else {
      results.emplace();
      *results = {observation_id, subfile_name, reduction_names,
                  std::move(reduction_data)};
    }
  }
};

std::optional<MockContributeReductionData::Results>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Event, tmpl::list<Events::ObserveAdaptiveSteppingDiagnostics>>>;
  };
};

template <typename Observer>
void test_observe(const Observer& observer) {
  using element_component = ElementComponent<Metavariables>;
  using observer_component = MockObserverComponent<Metavariables>;

  auto& results = MockContributeReductionData::results;
  results.reset();

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<observer_component>(&runner);

  using tag_list = tmpl::list<Parallel::Tags::MetavariablesImpl<Metavariables>,
                              Tags::Time, Tags::AdaptiveSteppingDiagnostics>;
  std::vector<db::compute_databox_type<tag_list>> element_boxes;

  const double observation_time = 2.0;
  const uint64_t num_slabs = 100;
  const uint64_t num_slab_changes = 20;
  uint64_t total_num_steps = 0;
  uint64_t total_num_step_changes = 0;
  uint64_t total_num_step_rejections = 0;

  const auto create_element = [&](const uint64_t num_steps,
                                  const uint64_t num_step_changes,
                                  const uint64_t num_step_rejections) {
    auto box = db::create<tag_list>(
        Metavariables{}, observation_time,
        AdaptiveSteppingDiagnostics{num_slabs, num_slab_changes, num_steps,
                                    num_step_changes, num_step_rejections});
    total_num_steps += num_steps;
    total_num_step_changes += num_step_changes;
    total_num_step_rejections += num_step_rejections;

    const auto ids_to_register =
        observers::get_registration_observation_type_and_key(observer, box);
    CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
    CHECK(ids_to_register->second == observers::ObservationKey("/subfile.dat"));

    element_boxes.push_back(std::move(box));

    ActionTesting::emplace_component<element_component>(
        &runner, element_boxes.size() - 1);
  };

  create_element(100, 12, 5);
  create_element(130, 90, 54);
  create_element(18, 2, 2);

  for (size_t index = 0; index < element_boxes.size(); ++index) {
    CHECK(static_cast<const Event&>(observer).is_ready(
        element_boxes[index],
        ActionTesting::cache<element_component>(runner, index),
        static_cast<element_component::array_index>(index),
        std::add_pointer_t<element_component>{}));
    observer.run(
        make_observation_box<db::AddComputeTags<>>(element_boxes[index]),
        ActionTesting::cache<element_component>(runner, index),
        static_cast<element_component::array_index>(index),
        std::add_pointer_t<element_component>{});
  }

  // Process the data
  for (size_t i = 0; i < element_boxes.size(); ++i) {
    REQUIRE(
        not runner.template is_simple_action_queue_empty<observer_component>(
            0));
    runner.template invoke_queued_simple_action<observer_component>(0);
  }
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  REQUIRE(results);
  auto& reduction_data = results->reduction_data;
  reduction_data.finalize();

  CHECK(results->observation_id.value() == observation_time);
  CHECK(results->subfile_name == "/subfile");
  CHECK(results->reduction_names[0] == "Time");
  CHECK(std::get<0>(reduction_data.data()) == observation_time);
  CHECK(results->reduction_names[1] == "Number of slabs");
  CHECK(std::get<1>(reduction_data.data()) == num_slabs);
  CHECK(results->reduction_names[2] == "Number of slab size changes");
  CHECK(std::get<2>(reduction_data.data()) == num_slab_changes);
  CHECK(results->reduction_names[3] == "Total steps on all elements");
  CHECK(std::get<3>(reduction_data.data()) == total_num_steps);
  CHECK(results->reduction_names[4] == "Number of LTS step changes");
  CHECK(std::get<4>(reduction_data.data()) == total_num_step_changes);
  CHECK(results->reduction_names[5] == "Number of step rejections");
  CHECK(std::get<5>(reduction_data.data()) == total_num_step_rejections);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.ParallelAlgorithms.Events.ObserveAdaptiveSteppingDiagnostics",
    "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();

  {
    const Events::ObserveAdaptiveSteppingDiagnostics observer("subfile");
    CHECK(not observer.needs_evolved_variables());
    test_observe(observer);
    test_observe(serialize_and_deserialize(observer));
  }
  {
    const auto event =
        TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
            "ObserveAdaptiveSteppingDiagnostics:\n"
            "  SubfileName: subfile");
    test_observe(*event);
    test_observe(*serialize_and_deserialize(event));
  }
}
