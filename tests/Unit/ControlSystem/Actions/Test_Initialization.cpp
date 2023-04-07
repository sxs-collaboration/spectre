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
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

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

struct MockMetavars {
  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;
  using component_list = tmpl::list<>;
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

  measurement_timescales[mock_control_sys::name()] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array<DataVector, 1>{{timescale}}, expr_time);

  Parallel::MutableGlobalCache<MockMetavars> mutable_cache{
      {std::move(measurement_timescales)}};
  Parallel::GlobalCache<MockMetavars> cache{{}, &mutable_cache};

  const Parallel::GlobalCache<MockMetavars>& cache_reference = cache;

  control_system::Actions::Initialize<MockMetavars, mock_control_sys>::apply(
      make_not_null(&averager), make_not_null(&current_measurement),
      &cache_reference);

  CHECK(expected_averager == averager);
  CHECK(current_measurement == 0);
}
