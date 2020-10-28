// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"  // for Variables

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare dg::Events::ObserveTime
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
namespace observers::Actions {
struct ContributeReductionData;
}  // namespace observers::Actions

namespace {

struct ObservationTimeTag : db::SimpleTag {
  using type = double;
};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::vector<std::string> reduction_names;
    String info_to_print;
  };
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    observers::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) noexcept {
    results.observation_id = observation_id;
    results.reduction_names = reduction_names;
    results.info_to_print = std::get<0>(reduction_data.data());
  }
};

MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>; // unused
  enum class Phase { Initialization, Testing, Exit };
};

template <typename ObserveEvent>
void test_observe(const std::unique_ptr<ObserveEvent> observe) noexcept {
  using metavariables = Metavariables<System>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;

  const typename element_component::array_index array_index(0);
  const double observation_time = 2.0;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_component<observer_component>(&runner, 0);

  const auto box = db::create<db::AddSimpleTags<
      ObservationTimeTag, Tags::Variables<typename decltype(vars)::tags_list>>>(
      observation_time, vars);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const observers::ObservationKey expected_observation_key_for_reg(
      "/reduction0.dat");
  CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
  CHECK(ids_to_register->second == expected_observation_key_for_reg);

  observe->run(box, runner.cache(), array_index,
               std::add_pointer_t<element_component>{});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.reduction_names[0] == "StringToPrint");
  CHECK(results.info_to_print == "???");
}

template <typename System>
void test_system() noexcept {
  INFO(pretty_type::get_name<System>());
  test_observe<System>(
      std::make_unique<dg::Events::ObserveErrorNorms<
          ObservationTimeTag, typename System::vars_for_test>>("reduction0"));

  INFO("create/serialize");
  using EventType = Event<tmpl::list<dg::Events::Registrars::ObserveErrorNorms<
      ObservationTimeTag, typename System::vars_for_test>>>;
  Parallel::register_derived_classes_with_charm<EventType>();
  const auto factory_event = TestHelpers::test_factory_creation<EventType>(
      "ObserveErrorNorms:\n"
      "  SubfileName: reduction0");
  auto serialized_event = serialize_and_deserialize(factory_event);
  test_observe<System>(std::move(serialized_event));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveErrorNorms", "[Unit][Evolution]") {
  test_system<ScalarSystem>();
  test_system<ComplicatedSystem>();
}
