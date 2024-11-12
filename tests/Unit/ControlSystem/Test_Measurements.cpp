// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <typeinfo>

#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Measurements/CharSpeed.hpp"
#include "ControlSystem/Measurements/SingleHorizon.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

using example_list =
    tmpl::list<control_system::TestHelpers::ExampleControlSystem>;

// BothHorizons
static_assert(
    tt::assert_conforms_to_v<control_system::measurements::BothHorizons::
                                 FindHorizon<::domain::ObjectLabel::A>,
                             control_system::protocols::Submeasurement>);
static_assert(
    tt::assert_conforms_to_v<control_system::measurements::BothHorizons,
                             control_system::protocols::Measurement>);
static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::BothHorizons::FindHorizon<
            ::domain::ObjectLabel::A>::interpolation_target_tag<example_list>,
        intrp::protocols::InterpolationTargetTag>);
static_assert(
    not control_system::measurements::BothHorizons::FindHorizon<
        ::domain::ObjectLabel::A>::event<example_list>::factory_creatable);

// SingleHorizon
static_assert(
    tt::assert_conforms_to_v<control_system::measurements::SingleHorizon<
                                 ::domain::ObjectLabel::B>::Submeasurement,
                             control_system::protocols::Submeasurement>);
static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::SingleHorizon<::domain::ObjectLabel::B>,
        control_system::protocols::Measurement>);
static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::SingleHorizon<::domain::ObjectLabel::B>::
            Submeasurement::interpolation_target_tag<example_list>,
        intrp::protocols::InterpolationTargetTag>);
static_assert(
    not control_system::measurements::SingleHorizon<::domain::ObjectLabel::B>::
        Submeasurement::event<example_list>::factory_creatable);

// BothNSCenters
static_assert(tt::assert_conforms_to_v<
              control_system::measurements::BothNSCenters::FindTwoCenters,
              control_system::protocols::Submeasurement>);
static_assert(
    tt::assert_conforms_to_v<control_system::measurements::BothNSCenters,
                             control_system::protocols::Measurement>);
static_assert(
    not control_system::measurements::BothNSCenters::FindTwoCenters::event<
        example_list>::factory_creatable);

// CharSpeed
static_assert(tt::assert_conforms_to_v<
              control_system::measurements::CharSpeed<domain::ObjectLabel::A>,
              control_system::protocols::Measurement>);
static_assert(
    tt::assert_conforms_to_v<control_system::measurements::CharSpeed<
                                 domain::ObjectLabel::A>::Excision,
                             control_system::protocols::Submeasurement>);
static_assert(
    tt::assert_conforms_to_v<control_system::measurements::CharSpeed<
                                 domain::ObjectLabel::A>::Horizon,
                             control_system::protocols::Submeasurement>);
static_assert(tt::assert_conforms_to_v<
              control_system::measurements::CharSpeed<domain::ObjectLabel::A>::
                  Excision::interpolation_target_tag<example_list>,
              intrp::protocols::InterpolationTargetTag>);
static_assert(tt::assert_conforms_to_v<
              control_system::measurements::CharSpeed<domain::ObjectLabel::A>::
                  Horizon::interpolation_target_tag<example_list>,
              intrp::protocols::InterpolationTargetTag>);
static_assert(
    not control_system::measurements::CharSpeed<domain::ObjectLabel::A>::
        Excision::event<example_list>::factory_creatable);
static_assert(
    not control_system::measurements::CharSpeed<domain::ObjectLabel::A>::
        Horizon::event<example_list>::factory_creatable);

namespace {
void test_serialization() {
  using EventPtr = std::unique_ptr<::Event>;
  using non_factory_creatable_event =
      control_system::measurements::BothHorizons::FindHorizon<
          ::domain::ObjectLabel::A>::event<example_list>;
  using factory_creatable_event =
      typename non_factory_creatable_event::factory_creatable_class;

  register_classes_with_charm<non_factory_creatable_event,
                              factory_creatable_event>();

  const EventPtr non_factory_event =
      std::make_unique<non_factory_creatable_event>();
  const EventPtr factory_event = std::make_unique<factory_creatable_event>();

  const EventPtr serialized_non_factory_event =
      serialize_and_deserialize(non_factory_event);
  const EventPtr serialized_factory_event =
      serialize_and_deserialize(factory_event);

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
#endif  // __clang__
  CHECK(typeid(*serialized_non_factory_event.get()) ==
        typeid(*non_factory_event.get()));
  CHECK(typeid(*serialized_factory_event.get()) ==
        typeid(*factory_event.get()));
#ifdef __clang__
#pragma clang diagnostic pop
#endif  // __clang__
}

// Check that the center of mass is at the expected location for this test.
void check_centers(const std::array<double, 3>& center_a,
                   const std::array<double, 3>& center_b,
                   const bool grid_centers = true) {
  const std::array<double, 3> expected_grid_center_a{1.0, 4.5 / 8.0, 3.0 / 8.0};
  const std::array<double, 3> expected_grid_center_b{-1.0, 1.0, 0.0};

  if (grid_centers) {
    CHECK(center_a == expected_grid_center_a);
    CHECK(center_b == expected_grid_center_b);
  } else {
    const std::array<double, 3> expected_inertial_center_a{
        cos(0.1) * expected_grid_center_a[0] +
            -sin(0.1) * expected_grid_center_a[1],
        sin(0.1) * expected_grid_center_a[0] +
            cos(0.1) * expected_grid_center_a[1],
        expected_grid_center_a[2]};
    const std::array<double, 3> expected_inertial_center_b{
        cos(0.1) * expected_grid_center_b[0] +
            -sin(0.1) * expected_grid_center_b[1],
        sin(0.1) * expected_grid_center_b[0] +
            cos(0.1) * expected_grid_center_b[1],
        expected_grid_center_b[2]};

    CHECK_ITERABLE_APPROX(center_a, expected_inertial_center_a);
    CHECK_ITERABLE_APPROX(center_b, expected_inertial_center_b);
  }
}

struct MockControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "MockControlSystem"; }
  static std::optional<std::string> component_name(
      const size_t /*i*/, const size_t num_components) {
    ASSERT(num_components == 1,
           "This control system expected 1 component but there are "
               << num_components << " instead.");
    return "Phi";
  }
  using measurement = control_system::measurements::BothNSCenters;
  using simple_tags = tmpl::list<
      control_system::TestHelpers::ExampleControlSystem::MeasurementQueue>;
  static constexpr size_t deriv_order = 2;
  using control_error = control_system::TestHelpers::ExampleControlError;
  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags =
        tmpl::list<control_system::measurements::Tags::NeutronStarCenter<
                       ::domain::ObjectLabel::A>,
                   control_system::measurements::Tags::NeutronStarCenter<
                       ::domain::ObjectLabel::B>>;
    using submeasurement =
        control_system::measurements::BothNSCenters::FindTwoCenters;

    template <typename Metavariables>
    static void apply(submeasurement /*meta*/,
                      const std::array<double, 3> center_a,
                      const std::array<double, 3> center_b,
                      Parallel::GlobalCache<Metavariables>& /*cache*/,
                      const LinkedMessageId<double>& /*measurement_id*/) {
      check_centers(center_a, center_b);
      // Avoid unused variable warning for deriv_order, which is required
      // as part of the control_system protocol.
      CHECK(2 == deriv_order);
    }
  };
};

template <typename Metavariables>
struct MockControlSystemComponent {
  using component_being_mocked =
      ControlComponent<Metavariables, MockControlSystem>;
  using const_global_cache_tags =
      tmpl::list<control_system::Tags::Verbosity, domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using simple_tags_from_options = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct MockWriteReductionDataRow {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& file_legend,
                    const std::tuple<std::vector<double>>& data_row) {
    CHECK((subfile_name == "/ControlSystems/BnsGridCenters" or
           subfile_name == "/ControlSystems/BnsInertialCenters"));
    CHECK(file_legend == std::vector<std::string>{
                             "Time", "Center_A_x", "Center_A_y", "Center_A_z",
                             "Center_B_x", "Center_B_y", "Center_B_z"});

    const std::vector<double>& data = get<0>(data_row);

    CHECK(data[0] == 1.0);
    CHECK(data.size() == file_legend.size());
    check_centers({data[1], data[2], data[3]}, {data[4], data[5], data[6]},
                  subfile_name == "/ControlSystems/BnsGridCenters");
  }
};

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using const_global_cache_tags =
      tmpl::list<control_system::Tags::WriteDataToDisk>;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<>>>>>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionDataRow>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionDataRow>;
};

struct Metavariables {
  using observed_reduction_data_tags = tmpl::list<>;
  void pup(PUP::er& /*p*/) {}
  using component_list = tmpl::list<MockObserverWriter<Metavariables>,
                                    MockControlSystemComponent<Metavariables>>;
};

}  // namespace

// This test tests the calculation of the center of mass of a star
// in the FindTwoCenters submeasurement
SPECTRE_TEST_CASE("Unit.ControlSystem.FindTwoCenters",
                  "[ControlSystem][Unit]") {
  test_serialization();
  domain::FunctionsOfTime::register_derived_with_charm();
  domain::creators::register_derived_with_charm();

  // Part 1 of the test: calculation of the relevant integrals
  // within a (mock) element.
  const Mesh<3> mesh(2, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const Scalar<DataVector> tilde_d{
      DataVector{0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0}};
  const Scalar<DataVector> inv_det_jacobian{
      DataVector{0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 1.0, 1.0}};
  const DataVector x_coord{-1.0, 1.0, -1.0, 1.0, 1.0, 2.0, 0.0, 2.0};
  const DataVector y_coord{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
  const DataVector z_coord{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
  const tnsr::I<DataVector, 3, Frame::Grid> x_grid{{x_coord, y_coord, z_coord}};

  // mass_a and mass_b: Integral of tilde_d/inv_det_jacobian for x>=0 and x<0
  // first_moment_a, first_moment_b : Same as masses, but multiplying the
  // integrand by x_grid
  double mass_a = 0.;
  double mass_b = 0.;
  std::array<double, 3> first_moment_a = {0., 0., 0.};
  std::array<double, 3> first_moment_b = {0., 0., 0.};
  control_system::measurements::center_of_mass_integral_on_element(
      &mass_a, &mass_b, &first_moment_a, &first_moment_b, mesh,
      inv_det_jacobian, tilde_d, x_grid);
  // Comparison with expected answer
  CHECK(mass_a == 8.0);
  CHECK(mass_b == 2.5);
  CHECK(first_moment_a == std::array<double, 3>{8.0, 4.5, 3.0});
  CHECK(first_moment_b == std::array<double, 3>{-2.5, 2.5, 0.0});

  // Use BNS domain with rotation for realistic block structure, but simple time
  // dependent maps for testing
  const domain::creators::bco::TimeDependentMapOptions<false> time_dep_opts{
      0.0,
      std::nullopt,
      domain::creators::bco::TimeDependentMapOptions<false>::RotationMapOptions{
          std::array{0.0, 0.0, 0.1}},
      std::nullopt,
      std::nullopt,
      std::nullopt};
  const domain::creators::BinaryCompactObject<false> binary_compact_object{
      domain::creators::BinaryCompactObject<false>::CartesianCubeAtXCoord{20.0},
      domain::creators::BinaryCompactObject<false>::CartesianCubeAtXCoord{
          -20.0},
      {0.0, 0.0},
      100.0,
      500.0,
      1.0,
      {0_st},
      {4_st},
      true,
      domain::CoordinateMaps::Distribution::Projective,
      domain::CoordinateMaps::Distribution::Linear,
      90.0,
      time_dep_opts};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using control_system_component = MockControlSystemComponent<Metavariables>;
  using obs_writer = MockObserverWriter<Metavariables>;

  MockRuntimeSystem runner{
      {true, ::Verbosity::Silent, binary_compact_object.create_domain()},
      {binary_compact_object.functions_of_time()}};
  ActionTesting::emplace_singleton_component<control_system_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0});
  ActionTesting::emplace_nodegroup_component_and_initialize<obs_writer>(
      make_not_null(&runner), {});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& cache = ActionTesting::cache<control_system_component>(runner, 0);

  LinkedMessageId<double> measurement_id{1.0, 0.0};
  ElementId<3> unused_element_id;

  using ControlSystems = tmpl::list<MockControlSystem>;
  auto box = db::create<db::AddSimpleTags<Tags::Time>>(1.0);

  // Test post-reduction action
  control_system::measurements::PostReductionSendBNSStarCentersToControlSystem<
      ControlSystems>::
      template apply<control_system_component>(box, cache, unused_element_id,
                                               measurement_id, mass_a, mass_b,
                                               first_moment_a, first_moment_b);

  // One for grid centers, one for inertial centers
  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer>(runner,
                                                                     0) == 2);
  for (size_t i = 0; i < 2; i++) {
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
  }
}
