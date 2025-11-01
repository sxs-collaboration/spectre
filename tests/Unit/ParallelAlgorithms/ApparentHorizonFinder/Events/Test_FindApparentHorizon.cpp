// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockFindApparentHorizon {
  using frame = ::Frame::Grid;

  struct Results {
    LinkedMessageId<double> time{};
    ElementId<3> element_id{};
    Mesh<3> mesh;
    Variables<ah::vars_to_interpolate_to_target<3, frame>> vars;
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
      Variables<ah::vars_to_interpolate_to_target<3, frame>>&&
          incoming_vars_to_interpolate,
      const std::optional<std::string>& dependency,
      const bool /*source_vars_have_already_been_received*/ = false) {
    results.time = incoming_time;
    results.element_id = incoming_element_id;
    results.mesh = incoming_mesh;
    results.vars = std::move(incoming_vars_to_interpolate);
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
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 ah::Tags::PreviousSurface<MockHorizonMetavars>>;

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
  ::domain::creators::time_dependence::register_derived_with_charm();
  using metavars = MockMetavariables;
  const ElementId<3> element_id(2);

  using component = MockComponent<metavars>;
  using elem_component = MockElement<metavars>;
  const double initial_time = 0.0;
  const std::array<double, 3> translation_velocity{{0.01, 0.02, 0.03}};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence = std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          initial_time, translation_velocity);
  const auto domain_creator = domain::creators::Sphere(
      1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st, false,
      std::nullopt, std::vector<double>{},
      domain::CoordinateMaps::Distribution::Linear, ShellWedges::All,
      std::optional{domain::creators::Sphere::TimeDepOptionType{
          std::move(time_dependence)}});
  const auto block_names = domain_creator.block_names();
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain(),
       std::unordered_map<std::string, std::unordered_set<std::string>>{
           {"MockHorizonMetavars", {block_names.begin(), block_names.end()}}}},
      {domain_creator.functions_of_time(),
       ah::Storage::LockedPreviousSurface<Frame::Grid>{}}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_array_component<elem_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      element_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const Mesh<3> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const LinkedMessageId<double> observation_time{2.0, {1.0}};
  auto& cache = ActionTesting::cache<elem_component>(runner, element_id);

  // Fill source vars with an analytic solution so that
  // ah::vars_to_interpolate_to_target can be computed from
  // the ah::source_vars. Previously, the source vars were all just set
  // to a constant value in this test. But in that case, the call to
  // FindApparentHorizon::apply() doesn't receive well-defined values, now
  // that it receives vars_to_interpolate_to_target instead of source_vars.
  Variables<ah::source_vars<3>> vars{};
  {
    const double mass = 1.0;
    const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
    const gr::Solutions::KerrSchild solution(mass, spin, {0.0, 0.0, 0.0});

    const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
    const auto& block = domain.blocks()[element_id.block_id()];

    const auto logical_coords = logical_coordinates(mesh);
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        inv_jacobian_logical_to_inertial{mesh.number_of_grid_points(), 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};
    if (block.is_time_dependent()) {
      const ElementMap<3, Frame::Grid> map_logical_to_grid{
          element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto& functions_of_time = domain_creator.functions_of_time();
      inertial_coords = block.moving_mesh_grid_to_inertial_map()(
          map_logical_to_grid(logical_coords), observation_time.id,
          functions_of_time);

      const auto inv_jacobian_logical_to_grid =
          map_logical_to_grid.inv_jacobian(logical_coords);
      const auto inv_jacobian_grid_to_inertial =
          block.moving_mesh_grid_to_inertial_map().inv_jacobian(
              map_logical_to_grid(logical_coords), observation_time.id,
              functions_of_time);
      inv_jacobian_logical_to_inertial = tenex::evaluate<ti::I, ti::j>(
          inv_jacobian_logical_to_grid(ti::I, ti::k) *
          inv_jacobian_grid_to_inertial(ti::K, ti::j));
    } else {
      const ElementMap<3, Frame::Inertial> map_logical_to_inertial{
          element_id, block.stationary_map().get_clone()};
      inertial_coords = map_logical_to_inertial(logical_coords);
      inv_jacobian_logical_to_inertial =
          map_logical_to_inertial.inv_jacobian(logical_coords);
    }

    const auto solution_vars = solution.variables(
        inertial_coords, observation_time.id,
        typename gr::Solutions::KerrSchild::tags<DataVector,
                                                 Frame::Inertial>{});

    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
    const auto& dt_lapse =
        get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
    const auto& d_lapse = get<typename gr::Solutions::KerrSchild::DerivLapse<
        DataVector, Frame::Inertial>>(solution_vars);
    const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(solution_vars);
    const auto& dt_shift =
        get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(solution_vars);
    const auto& d_shift = get<typename gr::Solutions::KerrSchild::DerivShift<
        DataVector, Frame::Inertial>>(solution_vars);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3>>(solution_vars);
    const auto& dt_spatial_metric =
        get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(solution_vars);
    const auto& d_spatial_metric =
        get<typename gr::Solutions::KerrSchild::DerivSpatialMetric<
            DataVector, Frame::Inertial>>(solution_vars);

    vars.initialize(get(lapse).size());
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars) =
        gr::spacetime_metric(lapse, shift, spatial_metric);
    get<gh::Tags::Phi<DataVector, 3>>(vars) = gh::phi(
        lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
    get<gh::Tags::Pi<DataVector, 3>>(vars) =
        gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
               dt_spatial_metric, get<gh::Tags::Phi<DataVector, 3>>(vars));
    get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>>(vars) =
        partial_derivative(get<gh::Tags::Phi<DataVector, 3>>(vars), mesh,
                           inv_jacobian_logical_to_inertial);
  }
  std::optional<std::string> dependency{"FakeDependency"};

  // Test the event version
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavars>,
      typename MockHorizonMetavars::time_tag, ::Events::Tags::ObserverMesh<3>,
      domain::Tags::Element<3>, ::Tags::Variables<ah::source_vars<3>>>>(
      metavars{}, observation_time, mesh, Element<3>{element_id, {}}, vars);

  const metavars::event event{dependency};
  const metavars::event serialized_event = serialize_and_deserialize(event);

  CHECK(serialized_event.needs_evolved_variables());

  auto obs_box = make_observation_box<
      typename metavars::event::compute_tags_for_observation_box>(
      make_not_null(&box));
  serialized_event.run(make_not_null(&obs_box), cache, element_id,
                       std::add_pointer_t<elem_component>{}, {});

  const auto check_results = [&]() {
    // Invoke all actions
    runner.invoke_queued_simple_action<component>(0);

    // No more queued simple actions.
    CHECK(runner.is_simple_action_queue_empty<component>(0));
    CHECK(runner.is_simple_action_queue_empty<elem_component>(element_id));

    const auto& results = MockFindApparentHorizon::results;
    CHECK(results.time == observation_time);
    CHECK(results.element_id == element_id);
    CHECK(results.mesh == mesh);

    Variables<ah::vars_to_interpolate_to_target<3, ::Frame::Grid>> target_vars{
        vars.number_of_grid_points()};
    const auto functions_of_time = domain_creator.functions_of_time();
    ah::compute_vars_to_interpolate_to_target(
        make_not_null(&target_vars),
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars),
        get<gh::Tags::Pi<DataVector, 3>>(vars),
        get<gh::Tags::Phi<DataVector, 3>>(vars),
        get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars),
        observation_time, domain_creator.create_domain(), mesh, element_id,
        functions_of_time);
    CHECK(results.vars == target_vars);

    CHECK(results.dependency == dependency);
  };

  check_results();

  MockFindApparentHorizon::results = MockFindApparentHorizon::Results{};
  dependency.reset();

  const auto option_event =
      TestHelpers::test_creation<typename metavars::event>("");
  const metavars::event serialized_option_event =
      serialize_and_deserialize(option_event);

  serialized_option_event.run(make_not_null(&obs_box), cache, element_id,
                              std::add_pointer_t<elem_component>{}, {});

  check_results();
}
}  // namespace
