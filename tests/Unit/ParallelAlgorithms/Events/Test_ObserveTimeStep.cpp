// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace observers::Actions {
struct ContributeReductionData;
}  // namespace observers::Actions

namespace {
static_assert(
    tt::assert_conforms_to<Events::detail::FormatTimeOutput,
                           observers::protocols::ReductionDataFormatter>);

template <typename Metavariables>
struct MockContributeReductionData {
  using ReductionData =
      tmpl::wrap<tmpl::front<typename Events::ObserveTimeStep<
                     Metavariables>::observed_reduction_data_tags>,
                 Parallel::ReductionData>;
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    ReductionData reduction_data;
    bool formatter_is_set {};
  };

  static std::optional<Results> results;

  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Formatter>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    observers::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    ReductionData&& reduction_data,
                    std::optional<Formatter>&& formatter) noexcept {
    if (results) {
      CHECK(results->observation_id == observation_id);
      CHECK(results->subfile_name == subfile_name);
      CHECK(results->reduction_names == reduction_names);
      CHECK(results->formatter_is_set == formatter.has_value());
      results->reduction_data.combine(std::move(reduction_data));
    } else {
      results.emplace();
      *results = {observation_id, subfile_name, reduction_names,
                  std::move(reduction_data), formatter.has_value()};
    }
  }
};

template <typename Metavariables>
std::optional<typename MockContributeReductionData<Metavariables>::Results>
    MockContributeReductionData<Metavariables>::results{};

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
  using with_these_simple_actions =
      tmpl::list<MockContributeReductionData<Metavariables>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Observer>
void test_observe(const Observer& observer,
                  const bool backwards_in_time) noexcept {
  using element_component = ElementComponent<Metavariables>;
  using observer_component = MockObserverComponent<Metavariables>;

  auto& results = MockContributeReductionData<Metavariables>::results;
  results.reset();

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const double observation_time = 2.0;
  const Slab slab(1.23, 4.56);

  using tag_list =
      tmpl::list<Tags::Time, Tags::TimeStep, System::variables_tag>;
  std::vector<db::compute_databox_type<tag_list>> element_boxes;

  const auto create_element =
      [&backwards_in_time, &element_boxes, &observation_time, &observer,
       &runner, &slab](const size_t num_points,
                       TimeDelta::rational_t slab_fraction) noexcept {
        if (backwards_in_time) {
          slab_fraction *= -1;
        }
        auto box = db::create<tag_list>(
            observation_time, slab.duration() * slab_fraction,
            System::variables_tag::type(num_points));

        const auto ids_to_register =
            observers::get_registration_observation_type_and_key(observer, box);
        CHECK(ids_to_register->first ==
              observers::TypeOfObservation::Reduction);
        CHECK(ids_to_register->second ==
              observers::ObservationKey("/time_step_subfile.dat"));

        element_boxes.push_back(std::move(box));

        ActionTesting::emplace_component<element_component>(
            &runner, element_boxes.size() - 1);
      };

  create_element(5, {1, 20});
  create_element(30, {1, 2});
  create_element(10, {1, 2});
  const size_t expected_num_points = 45;
  const double expected_slab_size = slab.duration().value();
  const double expected_min_step = expected_slab_size / 20.0;
  const double expected_max_step = expected_slab_size / 2.0;
  const double expected_effective_step = expected_slab_size / 4.0;

  for (size_t index = 0; index < element_boxes.size(); ++index) {
    observer.run(element_boxes[index],
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
  CHECK(results->subfile_name == "/time_step_subfile");
  CHECK(results->reduction_names[0] == "Time");
  CHECK(std::get<0>(reduction_data.data()) == observation_time);
  CHECK(results->reduction_names[1] == "NumberOfPoints");
  CHECK(std::get<1>(reduction_data.data()) == expected_num_points);
  CHECK(results->reduction_names[2] == "Slab size");
  CHECK(std::get<2>(reduction_data.data()) == expected_slab_size);
  CHECK(results->reduction_names[3] == "Minimum time step");
  CHECK(std::get<3>(reduction_data.data()) == expected_min_step);
  CHECK(results->reduction_names[4] == "Maximum time step");
  CHECK(std::get<4>(reduction_data.data()) == expected_max_step);
  CHECK(results->reduction_names[5] == "Effective time step");
  CHECK(std::get<5>(reduction_data.data()) == approx(expected_effective_step));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ObserveTimeStep", "[Unit][Evolution]") {
  using EventType =
      Event<tmpl::list<Events::Registrars::ObserveTimeStep<Metavariables>>>;
  Parallel::register_derived_classes_with_charm<EventType>();

  for (const bool print_to_terminal : {true, false}) {
    const Events::ObserveTimeStep<Metavariables> observer("time_step_subfile",
                                                          print_to_terminal);
    CHECK(not observer.needs_evolved_variables());
    test_observe(observer, false);
    test_observe(observer, true);
    test_observe(serialize_and_deserialize(observer), false);
    test_observe(serialize_and_deserialize(observer), true);

    const auto event = TestHelpers::test_factory_creation<EventType>(
        "ObserveTimeStep:\n"
        "  SubfileName: time_step_subfile\n"
        "  PrintTimeToTerminal: " +
        std::string(print_to_terminal ? "true" : "false"));
    test_observe(*event, false);
    test_observe(*event, true);
    test_observe(*serialize_and_deserialize(event), false);
    test_observe(*serialize_and_deserialize(event), true);
  }
}
