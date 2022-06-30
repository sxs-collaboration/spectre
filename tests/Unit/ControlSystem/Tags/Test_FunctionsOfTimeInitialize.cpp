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
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/FunctionsOfTimeInitialize.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

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
  using control_error = control_system::TestHelpers::ControlError;
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

struct MetavariablesReplace : Metavariables {
  static constexpr size_t volume_dim = 1;
  static constexpr bool override_functions_of_time = true;
  using control_systems = tmpl::list<FakeControlSystem<1>, FakeControlSystem<2>,
                                     FakeControlSystem<3>>;
  using component_list =
      control_system::control_components<Metavariables, control_systems>;
};

struct MetavariablesNoControlSystems {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<>;
};

struct MetavariablesNoControlSystemsReplace {
  static constexpr size_t volume_dim = 1;
  static constexpr bool override_functions_of_time = true;
  using component_list = tmpl::list<>;
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
  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    const std::array<DataVector, 3> initial_values{{{-1.0}, {-2.0}, {-3.0}}};

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
               initial_time, initial_values,
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
    functions_of_time[initial_expiration_times.begin()->first] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time, function_to_replace->func_and_2_derivs(initial_time),
            std::numeric_limits<double>::infinity());

    return functions_of_time;
  }
};

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

  // Initial expiration times are set to be update_fraction *
  // min(current_timescale) where update_fraction is the argument to the
  // Controller. This value for the timescale was chosen to give an expiration
  // time between the two expiration times used above in the TestCreator
  const double timescale = 27.0;
  const TimescaleTuner tuner1({timescale}, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01,
                              0.99);
  const TimescaleTuner tuner2({0.1}, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01, 0.99);
  const Averager<1> averager(0.25, true);
  const double update_fraction = 0.3;
  const Controller<2> controller(update_fraction);
  const control_system::TestHelpers::ControlError control_error{};

  OptionHolder<1> option_holder1(averager, controller, tuner1, control_error);
  OptionHolder<2> option_holder2(averager, controller, tuner1, control_error);
  OptionHolder<3> option_holder3(averager, controller, tuner2, control_error);

  // First test construction with only control systems and no
  // override_functions_of_time
  const double initial_time_step = 1.0;
  fot_tag::type functions_of_time = fot_tag::create_from_options<Metavariables>(
      creator, initial_time, initial_time_step, option_holder1, option_holder2,
      option_holder3);

  CHECK(functions_of_time.at("Controlled1")->time_bounds()[1] ==
        initial_time + update_fraction * timescale);
  CHECK(functions_of_time.at("Controlled2")->time_bounds()[1] ==
        initial_time + update_fraction * timescale);
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

  {
    // Create a temporary file with test data to read in
    // First, check if the file exists, and delete it if so
    const std::string test_filename{"TestReplaceFoTInTag.h5"};
    std::map<std::string, std::string> test_name_map{
        {{"FakeSpecName", "Controlled2"}}};

    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }

    h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

    constexpr size_t number_of_times = 3;
    constexpr uint32_t version_number = 0;
    const std::string expected_name{"FakeSpecName"};
    const std::array<double, number_of_times> expected_times{{0.0, 20.0, 40.0}};

    const std::array<DataVector, 3> initial_func{
        {{{2.0}}, {{-0.1}}, {{-0.02}}}};
    const std::array<DataVector, number_of_times - 1>
        next_replaced_second_deriv{{{{-0.5}}, {{-0.75}}}};
    domain::FunctionsOfTime::PiecewisePolynomial<2> replaced(
        expected_times[0], initial_func, expected_times[1]);
    replaced.update(expected_times[1], next_replaced_second_deriv[0],
                    expected_times[2]);
    replaced.update(expected_times[2], next_replaced_second_deriv[1],
                    expected_times[2] + 10.0);
    const std::array<std::array<DataVector, 3>, number_of_times - 1>&
        replaced_func_and_2_derivs_next{
            {replaced.func_and_2_derivs(expected_times[1]),
             replaced.func_and_2_derivs(expected_times[2])}};

    const std::vector<std::vector<double>> test_replaced{
        {expected_times[0], expected_times[0], 1.0, 2.0, 1.0,
         initial_func[0][0], initial_func[1][0], initial_func[2][0]},
        {expected_times[1], expected_times[1], 1.0, 2.0, 1.0,
         replaced_func_and_2_derivs_next[0][0][0],
         replaced_func_and_2_derivs_next[0][1][0],
         next_replaced_second_deriv[0][0]},
        {expected_times[2], expected_times[2], 1.0, 2.0, 1.0,
         replaced_func_and_2_derivs_next[1][0][0],
         replaced_func_and_2_derivs_next[1][1][0],
         next_replaced_second_deriv[1][0]}};
    const std::vector<std::string> replaced_legend{
        "Time",    "TLastUpdate", "Nc",   "DerivOrder",
        "Version", "Phi",         "dPhi", "d2Phi"};
    auto& replaced_file = test_file.insert<h5::Dat>(
        "/" + expected_name, replaced_legend, version_number);
    replaced_file.append(test_replaced);

    // Next test construction with control systems and
    // override_functions_of_time
    static_assert(
        std::is_same_v<
            fot_tag::option_tags<MetavariablesReplace>,
            tmpl::list<
                domain::OptionTags::DomainCreator<
                    MetavariablesReplace::volume_dim>,
                domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
                domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap,
                ::OptionTags::InitialTime, ::OptionTags::InitialTimeStep,
                ControlSysInputs<FakeControlSystem<1>>,
                ControlSysInputs<FakeControlSystem<2>>,
                ControlSysInputs<FakeControlSystem<3>>>>);
    auto replace_functions_of_time =
        fot_tag::create_from_options<MetavariablesReplace>(
            creator, {test_filename}, test_name_map, initial_time,
            initial_time_step, option_holder1, option_holder2, option_holder3);

    const double final_time = expected_times[number_of_times - 1];
    CHECK(replace_functions_of_time.at("Controlled2")->time_bounds()[1] ==
          final_time);
    CHECK(replace_functions_of_time.at("Controlled2")
              ->func_and_2_derivs(final_time)[2][0] ==
          next_replaced_second_deriv[0][0]);

    // Next test construction with control systems and
    // override_functions_of_time, but file is nullopt. Do same checks as if we
    // had control systems but not override_functions_of_time
    auto no_replace_functions_of_time =
        fot_tag::create_from_options<MetavariablesReplace>(
            creator, std::nullopt, test_name_map, initial_time,
            initial_time_step, option_holder1, option_holder2, option_holder3);
    CHECK(no_replace_functions_of_time.at("Controlled1")->time_bounds()[1] ==
          initial_time + update_fraction * timescale);
    CHECK(no_replace_functions_of_time.at("Controlled2")->time_bounds()[1] ==
          initial_time + update_fraction * timescale);
    CHECK(no_replace_functions_of_time.at("Controlled3")->time_bounds()[1] ==
          initial_time + initial_time_step);
    CHECK(no_replace_functions_of_time.at("Uncontrolled")->time_bounds()[1] ==
          std::numeric_limits<double>::infinity());

    const std::string new_filename{"FakeSpecName2"};
    test_name_map.clear();
    test_name_map[new_filename] = "Uncontrolled";
    const double highest_deriv = -1.1;
    const std::vector<std::vector<double>> test_new{
        {initial_time, initial_time, 1.0, 2.0, 1.0, 0.0, 0.0, highest_deriv}};
    test_file.close_current_object();
    auto& new_file = test_file.insert<h5::Dat>("/" + new_filename,
                                               replaced_legend, version_number);
    new_file.append(test_new);

    const Creator not_controlled_creator = std::make_unique<TestCreator>(false);
    // Next test construction without control systems but with
    // override_functions_of_time
    static_assert(
        std::is_same_v<
            fot_tag::option_tags<MetavariablesNoControlSystemsReplace>,
            tmpl::list<
                domain::OptionTags::DomainCreator<
                    MetavariablesNoControlSystemsReplace::volume_dim>,
                domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
                domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>>);
    auto no_control_sys_replaced_fot =
        fot_tag::create_from_options<MetavariablesNoControlSystemsReplace>(
            not_controlled_creator, {test_filename}, test_name_map);

    CHECK(no_control_sys_replaced_fot.size() == 1);
    CHECK(no_control_sys_replaced_fot.count("Uncontrolled") == 1);
    CHECK(no_control_sys_replaced_fot.at("Uncontrolled")
              ->func_and_2_derivs(initial_time)[2][0] == highest_deriv);
    CHECK(no_control_sys_replaced_fot.count("Controlled2") == 0);

    // Last test construction without control systems and without
    // override_functions_of_time
    static_assert(
        std::is_same_v<fot_tag::option_tags<MetavariablesNoControlSystems>,
                       tmpl::list<domain::OptionTags::DomainCreator<
                           MetavariablesNoControlSystems::volume_dim>>>);
    auto no_control_sys_fot =
        fot_tag::create_from_options<MetavariablesNoControlSystems>(
            not_controlled_creator);
    CHECK(no_control_sys_fot.size() == 1);
    CHECK(no_control_sys_fot.count("Uncontrolled") == 1);
    // -3.0 comes from the TestCreator above
    CHECK(no_control_sys_fot.at("Uncontrolled")
              ->func_and_2_derivs(initial_time)[2][0] == -3.0);
    CHECK(no_control_sys_fot.count("Controlled2") == 0);

    // Clean up
    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }
  }
}

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.FunctionsOfTimeInitialize",
                  "[ControlSystem][Unit]") {
  test_functions_of_time_tag();

  using fot_tag = control_system::Tags::FunctionsOfTimeInitialize;
  using Creator = tmpl::front<fot_tag::option_tags<Metavariables>>::type;

  CHECK_THROWS_WITH(
      []() {
        const Creator creator = std::make_unique<TestCreator>(true);

        const TimescaleTuner tuner({1.0}, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01,
                                   0.99);
        const Averager<1> averager(0.25, true);
        const double update_fraction = 0.3;
        const Controller<2> controller(update_fraction);
        const control_system::TestHelpers::ControlError control_error{};

        OptionHolder<1> option_holder1(averager, controller, tuner,
                                       control_error);
        OptionHolder<2> option_holder2(averager, controller, tuner,
                                       control_error);
        OptionHolder<3> option_holder3(averager, controller, tuner,
                                       control_error);
        OptionHolder<4> option_holder4(averager, controller, tuner,
                                       control_error);

        const double initial_time_step = 1.0;
        fot_tag::type functions_of_time =
            fot_tag::create_from_options<Metavariables>(
                creator, initial_time, initial_time_step, option_holder1,
                option_holder2, option_holder3, option_holder4);
      }(),
      Catch::Contains(
          "is not controlling a function of time. Check that the DomainCreator "
          "you have chosen uses all of the control systems in the executable. "
          "The existing functions of time are"));

  CHECK_THROWS_WITH(
      []() {
        const Creator creator = std::make_unique<BadCreator>();

        const TimescaleTuner tuner({1.0}, 10.0, 1.0e-3, 1.0e-2, 1.0e-4, 1.01,
                                   0.99);
        const Averager<1> averager(0.25, true);
        const double update_fraction = 0.3;
        const Controller<2> controller(update_fraction);
        const control_system::TestHelpers::ControlError control_error{};

        OptionHolder<1> option_holder1(averager, controller, tuner,
                                       control_error);
        OptionHolder<2> option_holder2(averager, controller, tuner,
                                       control_error);
        OptionHolder<3> option_holder3(averager, controller, tuner,
                                       control_error);

        const double initial_time_step = 1.0;
        fot_tag::type functions_of_time =
            fot_tag::create_from_options<Metavariables>(
                creator, initial_time, initial_time_step, option_holder1,
                option_holder2, option_holder3);
      }(),
      Catch::Contains("It is possible that the DomainCreator you are using "
                      "isn't compatible with the control systems"));
}
}  // namespace
