// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Measurements/CharSpeed.hpp"
#include "ControlSystem/Measurements/SingleHorizon.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ProtocolHelpers.hpp"

static_assert(
    tt::assert_conforms_to_v<control_system::measurements::BothHorizons::
                                 FindHorizon<::domain::ObjectLabel::A>,
                             control_system::protocols::Submeasurement>);

static_assert(
    tt::assert_conforms_to_v<control_system::measurements::BothHorizons,
                             control_system::protocols::Measurement>);

static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::BothHorizons::
            FindHorizon<::domain::ObjectLabel::A>::interpolation_target_tag<
                tmpl::list<control_system::TestHelpers::ExampleControlSystem>>,
        intrp::protocols::InterpolationTargetTag>);

static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::SingleHorizon<::domain::ObjectLabel::B>::
            Submeasurement::interpolation_target_tag<
                tmpl::list<control_system::TestHelpers::ExampleControlSystem>>,
        intrp::protocols::InterpolationTargetTag>);

static_assert(tt::assert_conforms_to_v<
              control_system::measurements::BothNSCenters::FindTwoCenters,
              control_system::protocols::Submeasurement>);

static_assert(
    tt::assert_conforms_to_v<control_system::measurements::BothNSCenters,
                             control_system::protocols::Measurement>);

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

static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::CharSpeed<domain::ObjectLabel::A>::
            Excision::interpolation_target_tag<
                tmpl::list<control_system::TestHelpers::ExampleControlSystem>>,
        intrp::protocols::InterpolationTargetTag>);

static_assert(
    tt::assert_conforms_to_v<
        control_system::measurements::CharSpeed<domain::ObjectLabel::A>::
            Horizon::interpolation_target_tag<
                tmpl::list<control_system::TestHelpers::ExampleControlSystem>>,
        intrp::protocols::InterpolationTargetTag>);

namespace {

// Check that the center of mass is at the expected location for this test.
void check_centers(const double center_a_x, const double center_a_y,
                   const double center_a_z, const double center_b_x,
                   const double center_b_y, const double center_b_z) {
  CHECK(center_a_x == 1.0);
  CHECK(center_a_y == 4.5 / 8.0);
  CHECK(center_a_z == 3.0 / 8.0);
  CHECK(center_b_x == -1.0);
  CHECK(center_b_y == 1.0);
  CHECK(center_b_z == 0.0);
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
      check_centers(center_a[0], center_a[1], center_a[2], center_b[0],
                    center_b[1], center_b[2]);
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
  using const_global_cache_tags = tmpl::list<control_system::Tags::Verbosity>;
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
    CHECK(subfile_name == "/ControlSystems/BnsCenters.dat");
    CHECK(file_legend == std::vector<std::string>{
                             "Time", "Center_A_x", "Center_A_y", "Center_A_z",
                             "Center_B_x", "Center_B_y", "Center_B_z"});

    const std::vector<double>& data = get<0>(data_row);

    CHECK(data[0] == 1.0);
    CHECK(data.size() == file_legend.size());
    check_centers(data[1], data[2], data[3], data[4], data[5], data[6]);
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

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using control_system_component = MockControlSystemComponent<Metavariables>;
  using obs_writer = MockObserverWriter<Metavariables>;

  MockRuntimeSystem runner{{true, ::Verbosity::Silent}};
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
}
