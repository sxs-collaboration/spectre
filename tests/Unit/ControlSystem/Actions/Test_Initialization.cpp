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
#include "ControlSystem/Tags/IsActiveMap.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct LabelA {};
struct LabelB {};
struct LabelC {};

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

using mock_control_sys_1 =
    MockControlSystem<LabelA, control_system::TestHelpers::Measurement<LabelA>>;
using mock_control_sys_2 =
    MockControlSystem<LabelB, control_system::TestHelpers::Measurement<LabelA>>;
using mock_control_sys_3 =
    MockControlSystem<LabelC, control_system::TestHelpers::Measurement<LabelC>>;

template <typename Metavariables, typename ControlSystem>
struct MockControlComponent {
  using array_index = int;
  using component_being_mocked = ControlComponent<Metavariables, ControlSystem>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;

  using system = ControlSystem;

  using simple_tags = tmpl::list<>;

  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct MockMetavars {
  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;
  using const_global_cache_tags =
      tmpl::list<control_system::Tags::SystemToCombinedNames,
                 control_system::Tags::IsActiveMap>;
  using component_list = tmpl::transform<
      tmpl::list<mock_control_sys_1, mock_control_sys_2, mock_control_sys_3>,
      tmpl::bind<MockControlComponent, tmpl::pin<MockMetavars>, tmpl::_1>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Initialization",
                  "[Unit][ControlSystem]") {
  domain::FunctionsOfTime::register_derived_with_charm();

  Averager<order - 1> averager{0.5, true};
  Averager<order - 1> expected_averager = averager;
  int current_measurement{};

  const double damping_time = 1.0;
  TimescaleTuner tuner{
      std::vector<double>{damping_time}, 10.0, 0.1, 2.0, 0.1, 1.01, 0.99};
  Controller<order> controller{0.3};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      measurement_timescales{};

  const double initial_time = 0.5;
  const double expr_time = 1.0;

  const DataVector timescale =
      control_system::calculate_measurement_timescales(controller, tuner, 4);

  expected_averager.assign_time_between_measurements(min(timescale));

  measurement_timescales["LabelALabelB"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array<DataVector, 1>{{timescale}}, expr_time);
  measurement_timescales["LabelC"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array<DataVector, 1>{{timescale}}, expr_time);

  std::unordered_map<std::string, std::string> system_to_combined_names{};
  system_to_combined_names["LabelA"] = "LabelALabelB";
  system_to_combined_names["LabelB"] = "LabelALabelB";
  system_to_combined_names["LabelC"] = "LabelC";
  std::unordered_map<std::string, bool> is_active_map{};
  is_active_map["LabelA"] = true;
  is_active_map["LabelB"] = false;
  is_active_map["LabelC"] = true;

  std::unordered_map<std::string, control_system::UpdateAggregator>
      aggregators{};

  Parallel::GlobalCache<MockMetavars> cache{
      {std::move(system_to_combined_names), std::move(is_active_map)},
      {std::move(measurement_timescales)}};

  const Parallel::GlobalCache<MockMetavars>& cache_reference = cache;

  control_system::Actions::Initialize<MockMetavars, mock_control_sys_1>::apply(
      make_not_null(&averager), make_not_null(&current_measurement),
      make_not_null(&aggregators), &cache_reference);

  CHECK(expected_averager == averager);
  CHECK(current_measurement == 0);

  CHECK(aggregators.count("LabelALabelB") == 1);
  CHECK(aggregators.count("LabelC") == 1);

  CHECK(aggregators.at("LabelALabelB").combined_name() == "LabelALabelB");
  CHECK(aggregators.at("LabelC").combined_name() == "LabelC");

  CHECK_FALSE(aggregators.at("LabelALabelB").is_ready());
  CHECK_FALSE(aggregators.at("LabelC").is_ready());
}
