// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace OptionTags {
struct InitialTime;
}  // namespace OptionTags

namespace {
const double initial_time = 2.0;

template <size_t Index>
struct FakeControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static constexpr size_t deriv_order = 2;
  static std::string name() { return "Controlled"s + get_output(Index); }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = control_system::TestHelpers::Measurement<
      control_system::TestHelpers::TestStructs_detail::LabelA>;
  using simple_tags = tmpl::list<>;
  using control_error = control_system::TestHelpers::ControlError<1>;
  struct process_measurement {
    using argument_tags = tmpl::list<>;
  };
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;

  using control_systems = tmpl::list<FakeControlSystem<1>, FakeControlSystem<2>,
                                     FakeControlSystem<3>>;
  using component_list =
      control_system::control_components<Metavariables, control_systems>;
};

struct MetavariablesNoControlSystems {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<>;
};

class TestCreator : public DomainCreator<1> {
 public:
  explicit TestCreator(const bool add_controlled)
      : add_controlled_(add_controlled) {}
  Domain<1> create_domain() const override { ERROR(""); }
  std::vector<DirectionMap<
      1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_refinement_levels()
      const override {
    ERROR("");
  }
  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    const std::array<DataVector, 3> initial_values{{{-1.0}, {-2.0}, {-3.0}}};
    const std::array<DataVector, 3> initial_values3{
        {{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}}};

    std::unordered_map<std::string, double> expiration_times{
        {"Uncontrolled", std::numeric_limits<double>::infinity()}};
    if (add_controlled_) {
      expiration_times["Controlled1"] = std::numeric_limits<double>::infinity();
      expiration_times["Controlled2"] = std::numeric_limits<double>::infinity();
      expiration_times["Controlled3"] = std::numeric_limits<double>::infinity();
      for (auto& [name, expr_time] : initial_expiration_times) {
        if (expiration_times.count(name) == 1) {
          expiration_times[name] = expr_time;
        }
      }
    }

    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        result{};

    if (add_controlled_) {
      result.insert(
          {"Controlled1",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values,
               expiration_times.at("Controlled1"))});
      result.insert(
          {"Controlled2",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values,
               expiration_times.at("Controlled2"))});
      result.insert(
          {"Controlled3",
           std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
               initial_time, initial_values3,
               expiration_times.at("Controlled3"))});
    }
    result.insert(
        {"Uncontrolled",
         std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
             initial_time, initial_values,
             expiration_times.at("Uncontrolled"))});
    return result;
  }

 private:
  bool add_controlled_{};
};

class BadCreator : public DomainCreator<1> {
 public:
  Domain<1> create_domain() const override { ERROR(""); }
  std::vector<DirectionMap<
      1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 1>> initial_refinement_levels()
      const override {
    ERROR("");
  }
  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    TestCreator good_creator{true};
    auto functions_of_time =
        good_creator.functions_of_time(initial_expiration_times);

    // Mimick a domain creator that has improperly set the expiration time for
    // one of the functions of time
    const auto& function_to_replace = functions_of_time.begin()->second;
    if (not initial_expiration_times.empty()) {
      functions_of_time[initial_expiration_times.begin()->first] =
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
              initial_time,
              function_to_replace->func_and_2_derivs(initial_time),
              std::numeric_limits<double>::infinity());
    }

    return functions_of_time;
  }
};

template <size_t Index>
using OptionHolder = control_system::OptionHolder<FakeControlSystem<Index>>;

template <typename ControlSys>
using ControlSysInputs =
    control_system::OptionTags::ControlSystemInputs<ControlSys>;

constexpr int measurements_per_update = 4;

void test_functions_of_time_tag() {
  INFO("Test FunctionsOfTimeInitialize tag");
  using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
  using Creator = std::unique_ptr<::DomainCreator<1>>;

  const Creator creator = std::make_unique<TestCreator>(true);

  // Initial expiration times are set to be update_fraction *
  // min(current_timescale) where update_fraction is the argument to the
  // Controller. This value for the timescale was chosen to give an expiration
  // time between the two expiration times used above in the TestCreator
  const double timescale = 27.0;
  const double timescale2 = 0.1;
  const TimescaleTuner tuner1(std::vector<double>{timescale}, 10.0, 1.0e-3,
                              1.0e-2, 1.0e-4, 1.01, 0.99);
  const TimescaleTuner tuner2(timescale2, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01,
                              0.99);
  const Averager<1> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);
  const control_system::TestHelpers::ControlError<1> control_error{};

  OptionHolder<1> option_holder1(false, averager, controller, tuner1,
                                 control_error);
  OptionHolder<2> option_holder2(true, averager, controller, tuner1,
                                 control_error);
  OptionHolder<3> option_holder3(true, averager, controller, tuner2,
                                 control_error);

  // First test construction with only control systems
  fot_tag::type functions_of_time = fot_tag::create_from_options<Metavariables>(
      creator, measurements_per_update, initial_time, option_holder1,
      option_holder2, option_holder3);

  const double expiration_controlled_2 =
      initial_time + update_fraction * timescale;
  const double expiration_controlled_3 =
      initial_time + update_fraction * timescale2;
  const double min_expiration_time =
      std::min(expiration_controlled_2, expiration_controlled_3);

  // 1 isn't active
  CHECK(functions_of_time.at("Controlled1")->time_bounds()[1] ==
        std::numeric_limits<double>::infinity());
  CHECK(functions_of_time.at("Controlled2")->time_bounds()[1] ==
        min_expiration_time);
  CHECK(functions_of_time.at("Controlled3")->time_bounds()[1] ==
        min_expiration_time);
  CHECK(functions_of_time.at("Uncontrolled")->time_bounds()[1] ==
        std::numeric_limits<double>::infinity());

  static_assert(
      std::is_same_v<
          fot_tag::option_tags<Metavariables>,
          tmpl::list<
              domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
              control_system::OptionTags::MeasurementsPerUpdate,
              ::OptionTags::InitialTime, ControlSysInputs<FakeControlSystem<1>>,
              ControlSysInputs<FakeControlSystem<2>>,
              ControlSysInputs<FakeControlSystem<3>>>>);

  {
    // Next test construction without control systems
    static_assert(
        std::is_same_v<fot_tag::option_tags<MetavariablesNoControlSystems>,
                       tmpl::list<domain::OptionTags::DomainCreator<
                           MetavariablesNoControlSystems::volume_dim>>>);
    const Creator not_controlled_creator = std::make_unique<TestCreator>(false);
    auto no_control_sys_fot =
        fot_tag::create_from_options<MetavariablesNoControlSystems>(
            not_controlled_creator);
    CHECK(no_control_sys_fot.size() == 1);
    CHECK(no_control_sys_fot.count("Uncontrolled") == 1);
    // -3.0 comes from the TestCreator above
    CHECK(no_control_sys_fot.at("Uncontrolled")
              ->func_and_2_derivs(initial_time)[2][0] == -3.0);
    CHECK(no_control_sys_fot.count("Controlled2") == 0);
  }
}

using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
using Creator = std::unique_ptr<::DomainCreator<1>>;

void not_controlling(const bool is_active) {
  const Creator creator = std::make_unique<TestCreator>(true);

  const TimescaleTuner tuner(std::vector<double>{1.0}, 10.0, 1.0e-3, 1.0e-2,
                             1.0e-4, 1.01, 0.99);
  const Averager<1> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);
  const control_system::TestHelpers::ControlError<1> control_error{};

  OptionHolder<1> option_holder1(is_active, averager, controller, tuner,
                                 control_error);
  OptionHolder<2> option_holder2(is_active, averager, controller, tuner,
                                 control_error);
  OptionHolder<3> option_holder3(is_active, averager, controller, tuner,
                                 control_error);
  OptionHolder<4> option_holder4(is_active, averager, controller, tuner,
                                 control_error);

  [[maybe_unused]] fot_tag::type functions_of_time =
      fot_tag::create_from_options<Metavariables>(
          creator, measurements_per_update, initial_time, option_holder1,
          option_holder2, option_holder3, option_holder4);
}

void incompatible(const bool is_active) {
  const Creator creator = std::make_unique<BadCreator>();

  const TimescaleTuner tuner(std::vector<double>{1.0}, 10.0, 1.0e-3, 1.0e-2,
                             1.0e-4, 1.01, 0.99);
  const Averager<1> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);
  const control_system::TestHelpers::ControlError<1> control_error{};

  OptionHolder<1> option_holder1(is_active, averager, controller, tuner,
                                 control_error);
  OptionHolder<2> option_holder2(is_active, averager, controller, tuner,
                                 control_error);
  OptionHolder<3> option_holder3(is_active, averager, controller, tuner,
                                 control_error);

  [[maybe_unused]] fot_tag::type functions_of_time =
      fot_tag::create_from_options<Metavariables>(
          creator, measurements_per_update, initial_time, option_holder1,
          option_holder2, option_holder3);
}

void test_errors(const bool is_active) {
  if (is_active) {
    CHECK_THROWS_WITH(
        not_controlling(is_active),
        Catch::Matchers::ContainsSubstring(
            "is not controlling a function of time. Check that the "
            "DomainCreator you have chosen uses all of the control systems in "
            "the executable. The existing functions of time are"));
    CHECK_THROWS_WITH(incompatible(is_active),
                      Catch::Matchers::ContainsSubstring(
                          "It is possible that the DomainCreator you are using "
                          "isn't compatible with the control systems"));
  } else {
    CHECK_NOTHROW(not_controlling(is_active));
    CHECK_NOTHROW(incompatible(is_active));
  }
}

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.FunctionsOfTimeInitialize",
                  "[ControlSystem][Unit]") {
  test_functions_of_time_tag();
  test_errors(true);
  test_errors(false);
}
}  // namespace
