// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var1 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

struct Var0TimesTwo : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var0TimesTwoCompute : db::ComputeTag, Var0TimesTwo {
  using base = Var0TimesTwo;
  using return_type = Scalar<DataVector>;
  using argument_tags = tmpl::list<Var0>;
  static void function(const gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& scalar_var) {
    get(*result) = 2.0 * get(scalar_var);
  }
};

struct Var0TimesThree : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var0TimesThreeCompute : db::ComputeTag,
                               ::Tags::Variables<tmpl::list<Var0TimesThree>> {
  using base = Var0TimesThree;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Var0>;
  static void function(
      const gsl::not_null<::Variables<tmpl::list<Var0TimesThree>>*> result,
      const Scalar<DataVector>& scalar_var) {
    result->initialize(get(scalar_var).size());
    get(get<Var0TimesThree>(*result)) = 3.0 * get(scalar_var);
  }
};

struct TestSectionIdTag {};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    double time;
    size_t number_of_grid_points;
    std::vector<double> max_values;
    std::vector<double> min_values;
    std::vector<double> l2_norm_values;
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
                    Parallel::ReductionData<Ts...>&& reduction_data) {
    reduction_data.finalize();
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.reduction_names = reduction_names;
    results.time = std::get<0>(reduction_data.data());
    results.number_of_grid_points = std::get<1>(reduction_data.data());
    results.max_values = std::get<2>(reduction_data.data());
    results.min_values = std::get<3>(reduction_data.data());
    results.l2_norm_values = std::get<4>(reduction_data.data());
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename ArraySectionIdTag>
using ObserveNormsEvent = Events::ObserveNorms<
    ::Tags::Time, tmpl::list<Var0, Var1, Var0TimesTwoCompute, Var0TimesThree>,
    tmpl::list<Var0TimesThreeCompute>, ArraySectionIdTag>;

template <typename ArraySectionIdTag>
struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<ObserveNormsEvent<ArraySectionIdTag>>>>;
  };

  enum class Phase { Initialization, Testing, Exit };
};

template <typename ArraySectionIdTag, typename ObserveEvent>
void test(const std::unique_ptr<ObserveEvent> observe,
          const std::optional<std::string>& section) {
  CAPTURE(pretty_type::short_name<ArraySectionIdTag>());
  CAPTURE(section);
  using metavariables = Metavariables<ArraySectionIdTag>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  const typename element_component::array_index array_index(0);
  const size_t num_points = 5;
  const double observation_time = 2.0;
  Variables<tmpl::list<Var0, Var1>> vars(num_points);
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavariables>, ::Tags::Time,
      Tags::Variables<typename decltype(vars)::tags_list>,
      observers::Tags::ObservationKey<ArraySectionIdTag>>>(
      metavariables{}, observation_time, vars, section);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const std::string expected_subfile_name{
      "/reduction0" +
      (std::is_same_v<ArraySectionIdTag, void> ? ""
                                               : section.value_or("Unused"))};
  const observers::ObservationKey expected_observation_key_for_reg(
      expected_subfile_name + ".dat");
  if (std::is_same_v<ArraySectionIdTag, void> or section.has_value()) {
    CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
    CHECK(ids_to_register->second == expected_observation_key_for_reg);
  } else {
    CHECK_FALSE(ids_to_register.has_value());
  }

  CHECK(static_cast<const Event&>(*observe).is_ready(
      box, ActionTesting::cache<element_component>(runner, array_index),
      array_index, std::add_pointer_t<element_component>{}));

  observe->run(
      make_observation_box<
          tmpl::filter<typename ObserveNormsEvent<
                           ArraySectionIdTag>::compute_tags_for_observation_box,
                       db::is_compute_tag<tmpl::_1>>>(box),
      ActionTesting::cache<element_component>(runner, array_index), array_index,
      std::add_pointer_t<element_component>{});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.subfile_name == expected_subfile_name);
  CHECK(results.reduction_names[0] == "Time");
  CHECK(results.time == observation_time);
  CHECK(results.reduction_names[1] == "NumberOfPoints");
  CHECK(results.number_of_grid_points == num_points);
  // Check max values
  CHECK(results.reduction_names[2] == "Max(Var0)");
  CHECK(results.reduction_names[3] == "Max(Var0)");
  CHECK(results.reduction_names[4] == "Max(Var0TimesTwo)");
  CHECK(results.reduction_names[5] == "Max(Var0TimesThree)");
  CHECK(results.max_values == std::vector<double>{5.0, 5.0, 10.0, 15.0});

  // Check min values
  CHECK(results.reduction_names[6] == "Min(Var1_x)");
  CHECK(results.reduction_names[7] == "Min(Var1_y)");
  CHECK(results.reduction_names[8] == "Min(Var1_z)");
  CHECK(results.reduction_names[9] == "Min(Var1)");
  CHECK(results.min_values == std::vector<double>{6.0, 11.0, 16.0, 6.0});

  // Check L2 norms
  CHECK(results.reduction_names[10] == "L2Norm(Var1)");
  CHECK(results.reduction_names[11] == "L2Norm(Var1_x)");
  CHECK(results.reduction_names[12] == "L2Norm(Var1_y)");
  CHECK(results.reduction_names[13] == "L2Norm(Var1_z)");
  CHECK(results.l2_norm_values[0] == approx(23.72762103540934575));
  CHECK(results.l2_norm_values[1] == approx(8.12403840463596083));
  CHECK(results.l2_norm_values[2] == approx(13.076696830622021));
  CHECK(results.l2_norm_values[3] == approx(18.055470085267789));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ObserveNorms", "[Unit][Evolution]") {
  test<TestSectionIdTag>(std::make_unique<ObserveNormsEvent<TestSectionIdTag>>(
                             ObserveNormsEvent<TestSectionIdTag>{
                                 "reduction0",
                                 {{"Var0", "Max", "Individual"},
                                  {"Var1", "Min", "Individual"},
                                  {"Var0", "Max", "Sum"},
                                  {"Var0TimesTwo", "Max", "Individual"},
                                  {"Var0TimesThree", "Max", "Individual"},
                                  {"Var1", "L2Norm", "Sum"},
                                  {"Var1", "L2Norm", "Individual"},
                                  {"Var1", "Min", "Sum"}}}),
                         "Section0");

  INFO("create/serialize");
  Parallel::register_factory_classes_with_charm<Metavariables<void>>();
  const auto factory_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables<void>>(
          // [input_file_examples]
          R"(
  ObserveNorms:
    SubfileName: reduction0
    TensorsToObserve:
    - Name: Var0
      NormType: Max
      Components: Individual
    - Name: Var1
      NormType: Min
      Components: Individual
    - Name: Var0
      NormType: Max
      Components: Sum
    - Name: Var0TimesTwo
      NormType: Max
      Components: Individual
    - Name: Var0TimesThree
      NormType: Max
      Components: Individual
    - Name: Var1
      NormType: L2Norm
      Components: Sum
    - Name: Var1
      NormType: L2Norm
      Components: Individual
    - Name: Var1
      NormType: Min
      Components: Sum
        )");
  // [input_file_examples]
  auto serialized_event = serialize_and_deserialize(factory_event);
  test<void>(std::move(serialized_event), std::nullopt);
}
