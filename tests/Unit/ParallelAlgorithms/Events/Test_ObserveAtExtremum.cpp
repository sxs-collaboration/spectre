// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
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
#include "IO/Observer/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveAtExtremum.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
  using type = std::optional<Scalar<DataVector>>;
};

struct Var0TimesTwoCompute : db::ComputeTag, Var0TimesTwo {
  using base = Var0TimesTwo;
  using return_type = std::optional<Scalar<DataVector>>;
  using argument_tags = tmpl::list<Var0>;
  static void function(
      const gsl::not_null<std::optional<Scalar<DataVector>>*> result,
      const Scalar<DataVector>& scalar_var) {
    *result = Scalar<DataVector>{DataVector{2.0 * get(scalar_var)}};
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
    std::vector<std::string> legend;
    double time;
    std::vector<double> reduced_vector;
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
                    const std::vector<std::string>& legend,
                    Parallel::ReductionData<Ts...>&& reduction_data) {
    reduction_data.finalize();
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.legend = legend;
    results.time = std::get<0>(reduction_data.data());
    results.reduced_vector = std::get<1>(reduction_data.data());
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
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

template <typename ArraySectionIdTag>
using ObserveAtExtremumEvent = Events::ObserveAtExtremum<
    ::Tags::Time, tmpl::list<Var0, Var1, Var0TimesTwoCompute, Var0TimesThree>,
    tmpl::list<Var0TimesThreeCompute>, ArraySectionIdTag>;

template <size_t Dim, typename ArraySectionIdTag>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Event, tmpl::list<ObserveAtExtremumEvent<ArraySectionIdTag>>>>;
  };
};

template <typename ArraySectionIdTag, typename ObserveEvent>
void test(const std::unique_ptr<ObserveEvent> observe,
          const std::string& extremum_type, const Spectral::Basis basis,
          const Spectral::Quadrature quadrature,
          const std::optional<std::string>& section) {
  CAPTURE(pretty_type::name<ArraySectionIdTag>());
  CAPTURE(section);
  using metavariables = Metavariables<3, ArraySectionIdTag>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  const typename element_component::array_index array_index(0);
  const Mesh<3> mesh{3, basis, quadrature};
  const size_t num_points = mesh.number_of_grid_points();
  const double observation_time = 2.0;

  Variables<tmpl::list<Var0, Var1>> vars(num_points);
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  // std::iota(vars.data(), vars.data() + vars.size(), 1.0);
  std::iota(vars.data(), vars.data() + num_points + 5, 1.0);
  std::iota(vars.data() + num_points + 5, vars.data() + vars.size(), -10.0);

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      array_index);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavariables>,
                        ::Events::Tags::ObserverMesh<3>, ::Tags::Time,
                        Tags::Variables<typename decltype(vars)::tags_list>,
                        observers::Tags::ObservationKey<ArraySectionIdTag>>>(
      metavariables{}, mesh, observation_time, vars, section);

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
          tmpl::filter<typename ObserveAtExtremumEvent<
                           ArraySectionIdTag>::compute_tags_for_observation_box,
                       db::is_compute_tag<tmpl::_1>>>(box),
      ActionTesting::cache<element_component>(runner, array_index), array_index,
      std::add_pointer_t<element_component>{});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.observation_id.observation_key() ==
        expected_observation_key_for_reg);
  CHECK(results.subfile_name == expected_subfile_name);
  CHECK(results.time == observation_time);
  CHECK(results.legend[0] == "Time");
  if (extremum_type == "Max") {
    CHECK(results.legend[1] == "Max(Var0)");
    CHECK(results.legend[2] == "AtVar0Max(Var0)");
    CHECK(results.legend[3] == "AtVar0Max(Var1_x)");
    CHECK(results.legend[4] == "AtVar0Max(Var1_y)");
    CHECK(results.legend[5] == "AtVar0Max(Var1_z)");
    CHECK(results.reduced_vector ==
          std::vector<double>{27.0, 27.0, 11.0, 38.0, 65.0});
  } else {
    CHECK(results.legend[1] == "Min(Var0)");
    CHECK(results.legend[2] == "AtVar0Min(Var0)");
    CHECK(results.legend[3] == "AtVar0Min(Var1_x)");
    CHECK(results.legend[4] == "AtVar0Min(Var1_y)");
    CHECK(results.legend[5] == "AtVar0Min(Var1_z)");
    CHECK(results.reduced_vector ==
          std::vector<double>{1.0, 1.0, 28.0, 12.0, 39.0});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ObserveAtExtremum", "[Unit][Evolution]") {
  test<TestSectionIdTag>(
      std::make_unique<ObserveAtExtremumEvent<TestSectionIdTag>>(
          ObserveAtExtremumEvent<TestSectionIdTag>{
              "reduction0", {"Var0", "Max", {"Var0", "Var1"}}}),
      "Max", Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto,
      "Section0");
  test<TestSectionIdTag>(
      std::make_unique<ObserveAtExtremumEvent<TestSectionIdTag>>(
          ObserveAtExtremumEvent<TestSectionIdTag>{
              "reduction0", {"Var0", "Min", {"Var0", "Var1"}}}),
      "Min", Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto,
      "Section0");

  INFO("create/serialize");
  register_factory_classes_with_charm<Metavariables<3, void>>();
  const auto factory_event = TestHelpers::test_creation<std::unique_ptr<Event>,
                                                        Metavariables<3, void>>(
      // [input_file_examples]
      R"(
  ObserveAtExtremum:
    SubfileName: reduction0
    TensorsToObserve:
      Name: Var0
      ExtremumType: Max
      AdditionalData:
      - Var0
      - Var1
        )");
  // [input_file_examples]
  auto serialized_event = serialize_and_deserialize(factory_event);
  test<void>(std::move(serialized_event), "Max", Spectral::Basis::Legendre,
             Spectral::Quadrature::GaussLobatto, std::nullopt);

  test<void>(std::make_unique<ObserveAtExtremumEvent<void>>(
                 ObserveAtExtremumEvent<void>{
                     "reduction0", {"Var0", "Max", {"Var0", "Var1"}}}),
             "Max", Spectral::Basis::FiniteDifference,
             Spectral::Quadrature::CellCentered, std::nullopt);

  // Test that Max reduction has the desired behavior on vectors
  {
    using ReductionData = Parallel::ReductionData<
        // Observation value
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
        // Maximum of first component of a vector
        Parallel::ReductionDatum<std::vector<double>, funcl::Max<>>>;
    ReductionData first_data_set{1.0, std::vector<double>{1.0, 2.0, -1.0}};
    ReductionData second_data_set{1.0, std::vector<double>{2.0, 1.0, -3.0}};
    first_data_set.combine(std::move(second_data_set));
    CHECK(std::get<0>(first_data_set.data()) == 1.0);
    CHECK(std::get<1>(first_data_set.data()) ==
          std::vector<double>{2.0, 1.0, -3.0});
  }
  // Test that Min reduction has the desired behavior on vectors
  {
    using ReductionData = Parallel::ReductionData<
        // Observation value
        Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
        // Maximum of first component of a vector
        Parallel::ReductionDatum<std::vector<double>, funcl::Min<>>>;
    ReductionData first_data_set{1.0, std::vector<double>{1.0, 2.0, -1.0}};
    ReductionData second_data_set{1.0, std::vector<double>{2.0, 1.0, -3.0}};
    first_data_set.combine(std::move(second_data_set));
    CHECK(std::get<0>(first_data_set.data()) == 1.0);
    CHECK(std::get<1>(first_data_set.data()) ==
          std::vector<double>{1.0, 2.0, -1.0});
  }
}
