// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
const double initial_time = 2.0;

template <size_t Index>
struct FakeControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static constexpr size_t deriv_order = 2;
  static std::string name() { return "Controlled"s + get_output(Index); }
  using measurement = control_system::TestHelpers::Measurement<
      control_system::TestHelpers::TestStructs_detail::LabelA>;
  using simple_tags = tmpl::list<>;
  struct process_measurement {
    using argument_tags = tmpl::list<>;
  };
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;

  using control_systems = tmpl::list<FakeControlSystem<1>, FakeControlSystem<2>,
                                     FakeControlSystem<3>>;
  using component_list = control_components<Metavariables, control_systems>;
};

class TestCreator : public DomainCreator<1> {
 public:
  explicit TestCreator(const bool add_controlled)
      : add_controlled_(add_controlled) {}
  Domain<1> create_domain() const override { ERROR(""); }
  std::vector<std::array<size_t, 1>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_refinement_levels()
      const override {
    ERROR("");
  }
  auto functions_of_time() const -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    const std::array<DataVector, 3> initial_values{{{-1.0}, {-2.0}, {-3.0}}};

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        result{};
    if (add_controlled_) {
      result.insert(
          {"Controlled1",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, initial_time + 7.0)});
      result.insert(
          {"Controlled2",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, initial_time + 10.0)});
      result.insert(
          {"Controlled3",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values, initial_time + 0.5)});
    }
    result.insert(
        {"Uncontrolled",
         std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
             initial_time, initial_values,
             std::numeric_limits<double>::infinity())});
    return result;
  }

 private:
  bool add_controlled_{};
};

void test_all_tags() {
  INFO("Test all tags");
  using name_tag = control_system::Tags::ControlSystemName;
  TestHelpers::db::test_simple_tag<name_tag>("ControlSystemName");
  using averager_tag = control_system::Tags::Averager<2>;
  TestHelpers::db::test_simple_tag<averager_tag>("Averager");
  using timescaletuner_tag = control_system::Tags::TimescaleTuner;
  TestHelpers::db::test_simple_tag<timescaletuner_tag>("TimescaleTuner");
  using controller_tag = control_system::Tags::Controller<2>;
  TestHelpers::db::test_simple_tag<controller_tag>("Controller");
  using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
  TestHelpers::db::test_simple_tag<fot_tag>("FunctionsOfTime");

  using system = control_system::TestHelpers::System<
      2, control_system::TestHelpers::TestStructs_detail::LabelA,
      control_system::TestHelpers::Measurement<
          control_system::TestHelpers::TestStructs_detail::LabelA>>;
  using control_system_inputs_tag =
      control_system::Tags::ControlSystemInputs<system>;
  TestHelpers::db::test_simple_tag<control_system_inputs_tag>(
      "ControlSystemInputs");

  using measurement_tag = control_system::Tags::MeasurementTimescales;
  TestHelpers::db::test_simple_tag<measurement_tag>("MeasurementTimescales");
}

void test_control_sys_inputs() {
  INFO("Test control system inputs");
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const TimescaleTuner expected_tuner(
      {1.}, max_timescale, min_timescale, decrease_timescale_threshold,
      increase_timescale_threshold, increase_factor, decrease_factor);
  const Averager<2> expected_averager(0.25, true);
  const Controller<2> expected_controller(0.3);
  const std::string expected_name{"LabelA"};

  using system = control_system::TestHelpers::System<
      2, control_system::TestHelpers::TestStructs_detail::LabelA,
      control_system::TestHelpers::Measurement<
          control_system::TestHelpers::TestStructs_detail::LabelA>>;
  const auto input_holder = TestHelpers::test_option_tag<
      control_system::OptionTags::ControlSystemInputs<system>>(
      "Averager:\n"
      "  AverageTimescaleFraction: 0.25\n"
      "  Average0thDeriv: true\n"
      "Controller:\n"
      "  UpdateFraction: 0.3\n"
      "TimescaleTuner:\n"
      "  InitialTimescales: [1.]\n"
      "  MinTimescale: 1e-3\n"
      "  MaxTimescale: 10.\n"
      "  DecreaseThreshold: 1e-2\n"
      "  IncreaseThreshold: 1e-4\n"
      "  IncreaseFactor: 1.01\n"
      "  DecreaseFactor: 0.99\n");
  CHECK(expected_averager == input_holder.averager);
  CHECK(expected_controller == input_holder.controller);
  CHECK(expected_tuner == input_holder.tuner);
  CHECK(expected_name ==
        std::decay_t<decltype(input_holder)>::control_system::name());
}

template <size_t Index>
using OptionHolder = control_system::OptionHolder<FakeControlSystem<Index>>;

template <typename ControlSys>
using ControlSysInputs =
    control_system::OptionTags::ControlSystemInputs<ControlSys>;

void test_functions_of_time_tag() {
  INFO("Test FunctionsOfTimeInitialize tag");
  using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
  using Creator = tmpl::front<fot_tag::option_tags<Metavariables>>::type;

  const Creator creator = std::make_unique<TestCreator>(true);

  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  // Initial expiration times are set to be update_fraction *
  // min(current_timescale) where update_fraction is the argument to the
  // Controller. This value for the timescale was chosen to give an expiration
  // time between the two expiration times used above in the TestCreator
  const double timescale = 27.0;
  const TimescaleTuner tuner1(
      {timescale}, max_timescale, min_timescale, decrease_timescale_threshold,
      increase_timescale_threshold, increase_factor, decrease_factor);
  const TimescaleTuner tuner2(
      {0.1}, max_timescale, min_timescale, decrease_timescale_threshold,
      increase_timescale_threshold, increase_factor, decrease_factor);
  const Averager<2> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);

  OptionHolder<1> option_holder1(averager, controller, tuner1);
  OptionHolder<2> option_holder2(averager, controller, tuner1);
  OptionHolder<3> option_holder3(averager, controller, tuner2);

  const double initial_time_step = 1.0;
  fot_tag::type functions_of_time = fot_tag::create_from_options<Metavariables>(
      creator, initial_time, initial_time_step, option_holder1, option_holder2,
      option_holder3);

  CHECK(functions_of_time.at("Controlled1")->time_bounds()[1] ==
        initial_time + update_fraction * timescale);
  CHECK(functions_of_time.at("Controlled2")->time_bounds()[1] ==
        initial_time + 10.0);
  CHECK(functions_of_time.at("Controlled3")->time_bounds()[1] ==
        initial_time + initial_time_step);
  CHECK(functions_of_time.at("Uncontrolled")->time_bounds()[1] ==
        std::numeric_limits<double>::infinity());

  static_assert(
      std::is_same_v<
          fot_tag::option_tags<Metavariables>,
          tmpl::list<
              domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
              ::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
              ControlSysInputs<FakeControlSystem<1>>,
              ControlSysInputs<FakeControlSystem<2>>,
              ControlSysInputs<FakeControlSystem<3>>>>);
}

void test_measurement_tag() {
  INFO("Test measurement tag");
  using measurement_tag = control_system::Tags::MeasurementTimescales;
  static_assert(
      tmpl::size<measurement_tag::option_tags<Metavariables>>::value == 2);
  using Creator =
      tmpl::front<measurement_tag::option_tags<Metavariables>>::type;
  const double time_step = 0.2;
  {
    const Creator creator = std::make_unique<TestCreator>(true);

    const measurement_tag::type timescales =
        measurement_tag::create_from_options<Metavariables>(creator, time_step);
    CHECK(timescales.size() == 3);
    // The lack of expiration is a placeholder until the control systems
    // have been implemented sufficiently to manage their timescales.
    CHECK(timescales.at("Controlled1")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled1")->func(2.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled1")->func(3.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled2")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled2")->func(2.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled2")->func(3.0)[0] == DataVector{time_step});
    CHECK(timescales.at("Controlled3")->time_bounds() ==
          std::array{initial_time, std::numeric_limits<double>::infinity()});
    CHECK(timescales.at("Controlled3")->func(2.5)[0] == DataVector{time_step});
  }
  {
    const Creator creator = std::make_unique<TestCreator>(false);

    // Verify that negative time steps are accepted with no control
    // systems.
    const measurement_tag::type timescales =
        measurement_tag::create_from_options<Metavariables>(creator,
                                                            -time_step);
    CHECK(timescales.empty());
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags", "[ControlSystem][Unit]") {
  test_all_tags();
  test_control_sys_inputs();
  test_measurement_tag();
  test_functions_of_time_tag();
}

// [[OutputRegex, Control systems can only be used in forward-in-time
// evolutions.]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.Backwards",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  using measurement_tag = control_system::Tags::MeasurementTimescales;
  const std::unique_ptr<DomainCreator<1>> creator =
      std::make_unique<TestCreator>(true);
  measurement_tag::create_from_options<Metavariables>(creator, -1.0);
}
