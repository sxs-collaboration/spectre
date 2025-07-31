// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockFindApparentHorizon {
  struct Results {
    LinkedMessageId<double> time{};
    ElementId<3> element_id{};
    Mesh<3> mesh;
    Variables<ah::source_vars<3>> vars;
    std::optional<std::string> dependency;
  };
  static Results results;  // NOLINT

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<double>& incoming_time,
      const ElementId<3>& incoming_element_id, const ::Mesh<3>& incoming_mesh,
      Variables<ah::source_vars<3>>&& incoming_source_vars,
      const std::optional<std::string>& dependency,
      const bool /*source_vars_have_already_been_received*/ = false) {
    results.time = incoming_time;
    results.element_id = incoming_element_id;
    results.mesh = incoming_mesh;
    results.vars = incoming_source_vars;
    results.dependency = dependency;
  }
};

MockFindApparentHorizon::Results MockFindApparentHorizon::results{};  // NOLINT

struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;

  using frame = ::Frame::Grid;

  // Don't need callbacks
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "MockHorizonMetavars"; }
};

template <typename Metavariables>
struct MockComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      ah::Component<Metavariables, MockHorizonMetavars>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, ah::Tags::BlocksForHorizonFind>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<ah::FindApparentHorizon<MockHorizonMetavars>>;
  using with_these_simple_actions = tmpl::list<MockFindApparentHorizon>;
};

template <typename Metavariables>
struct MockElement {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
};

struct MockMetavariables {
  using component_list = tmpl::list<MockComponent<MockMetavariables>,
                                    MockElement<MockMetavariables>>;

  using event = ah::Events::FindApparentHorizon<MockHorizonMetavars>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FindApparentHorizonEvent",
                  "[ApparentHorizonFinder][Unit]") {
  (void)MockHorizonMetavars::destination;
  ::domain::FunctionsOfTime::register_derived_with_charm();
  ::domain::creators::register_derived_with_charm();
  using metavars = MockMetavariables;
  const ElementId<3> element_id(2);
  const ElementId<3> array_index(element_id);

  using component = MockComponent<metavars>;
  using elem_component = MockElement<metavars>;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  const double initial_expr_time = 0.1;
  const std::string name{"FunctionToCheck"};
  functions_of_time[name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array<DataVector, 1>{{DataVector{1, 0.0}}},
          initial_expr_time);
  const auto domain_creator = domain::creators::Sphere(
      1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);
  const auto block_names = domain_creator.block_names();
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain(),
       std::unordered_map<std::string, std::unordered_set<std::string>>{
           {"MockHorizonMetavars", {block_names.begin(), block_names.end()}}}},
      {std::move(functions_of_time)}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_array_component<elem_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      array_index);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const Mesh<3> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const LinkedMessageId<double> observation_time{2.0, {1.0}};
  Variables<ah::source_vars<3>> vars(mesh.number_of_grid_points(), 9.876);
  std::optional<std::string> dependency{"FakeDependency"};
  auto& cache = ActionTesting::cache<elem_component>(runner, array_index);

  // Test the event version
  auto box =
      db::create<db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavars>,
                                   typename MockHorizonMetavars::time_tag,
                                   ::Events::Tags::ObserverMesh<3>,
                                   ::Tags::Variables<ah::source_vars<3>>>>(
          metavars{}, observation_time, mesh, vars);

  const metavars::event event{dependency};
  const metavars::event serialized_event = serialize_and_deserialize(event);

  CHECK(serialized_event.needs_evolved_variables());

  auto obs_box = make_observation_box<
      typename metavars::event::compute_tags_for_observation_box>(
      make_not_null(&box));
  serialized_event.run(make_not_null(&obs_box), cache, array_index,
                       std::add_pointer_t<elem_component>{}, {});

  const auto check_results = [&]() {
    // Invoke all actions
    runner.invoke_queued_simple_action<component>(0);

    // No more queued simple actions.
    CHECK(runner.is_simple_action_queue_empty<component>(0));
    CHECK(runner.is_simple_action_queue_empty<elem_component>(array_index));

    const auto& results = MockFindApparentHorizon::results;
    CHECK(results.time == observation_time);
    CHECK(results.element_id == element_id);
    CHECK(results.mesh == mesh);
    CHECK(results.vars == vars);
    CHECK(results.dependency == dependency);
  };

  check_results();

  MockFindApparentHorizon::results = MockFindApparentHorizon::Results{};
  dependency.reset();

  const auto option_event =
      TestHelpers::test_creation<typename metavars::event>("");
  const metavars::event serialized_option_event =
      serialize_and_deserialize(option_event);

  serialized_option_event.run(make_not_null(&obs_box), cache, array_index,
                              std::add_pointer_t<elem_component>{}, {});

  check_results();
}
}  // namespace
