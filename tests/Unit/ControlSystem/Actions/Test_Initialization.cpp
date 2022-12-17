// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Actions/Initialization.hpp"
#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct LabelA {};

constexpr size_t order = 2;

template <typename Label, typename Measurement>
struct MockControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return pretty_type::short_name<Label>(); }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = Measurement;
  using control_error = control_system::TestHelpers::ControlError<0>;
  static constexpr size_t deriv_order = order;
  using simple_tags = tmpl::list<>;
};

using mock_control_sys =
    MockControlSystem<LabelA, control_system::TestHelpers::Measurement<LabelA>>;

template <typename Metavariables>
struct MockControlComponent {
  using array_index = int;
  using component_being_mocked =
      ControlComponent<Metavariables, mock_control_sys>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;

  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;

  using simple_tags =
      tmpl::list<control_system::Tags::Averager<mock_control_sys>,
                 control_system::Tags::TimescaleTuner<mock_control_sys>,
                 control_system::Tags::WriteDataToDisk,
                 control_system::Tags::ControlError<mock_control_sys>,
                 control_system::Tags::Controller<mock_control_sys>,
                 control_system::Tags::CurrentNumberOfMeasurements>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>,
                 control_system::Actions::Initialize<Metavariables,
                                                     mock_control_sys>>>>;
};

struct MockMetavars {
  using component_list = tmpl::list<MockControlComponent<MockMetavars>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Initialization",
                  "[Unit][ControlSystem]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  using component = MockControlComponent<MockMetavars>;
  using tags = typename component::simple_tags;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<MockMetavars>;

  Averager<order - 1> averager{0.5, true};
  const double damping_time = 1.0;
  TimescaleTuner tuner{
      std::vector<double>{damping_time}, 10.0, 0.1, 2.0, 0.1, 1.01, 0.99};
  Controller<order> controller{0.3};
  bool write_data = false;
  const control_system::TestHelpers::ControlError<0> control_error{};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      measurement_timescales{};

  const double initial_time = 0.5;
  const double expr_time = 1.0;
  const DataVector timescale =
      control_system::calculate_measurement_timescales(controller, tuner, 4);
  measurement_timescales[mock_control_sys::name()] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array<DataVector, 1>{{timescale}}, expr_time);

  controller.set_initial_update_time(initial_time);
  controller.assign_time_between_updates(damping_time);

  tuples::tagged_tuple_from_typelist<tags> init_tuple{
      averager, tuner, write_data, control_error, controller, 0};

  MockRuntimeSystem runner{{}, {std::move(measurement_timescales)}};
  ActionTesting::emplace_singleton_component_and_initialize<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, init_tuple);

  // This has to come after we create the component so that it isn't initialized
  // immediately
  averager.assign_time_between_measurements(min(timescale));

  const auto& box_averager = ActionTesting::get_databox_tag<
      component, control_system::Tags::Averager<mock_control_sys>>(runner, 0);
  const auto& box_current_measurement = ActionTesting::get_databox_tag<
      component, control_system::Tags::CurrentNumberOfMeasurements>(runner, 0);

  // Check that things haven't been initialized
  CHECK(box_averager != averager);
  // int's are initialized to 0 anyways
  CHECK(box_current_measurement == 0);

  // Now initialize everything
  runner.next_action<component>(0);

  // Check that box values are same as expected values
  CHECK(box_averager == averager);
  CHECK(box_current_measurement == 0);
}
