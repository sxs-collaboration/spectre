// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Importers/ReadSpecThirdOrderPiecewisePolynomial.hpp"
#include "IO/Importers/Tags.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = tmpl::list<::domain::Tags::FunctionsOfTime>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              importers::Actions::ReadSpecThirdOrderPiecewisePolynomial>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

void test_options() noexcept {
  CHECK(db::tag_name<importers::Tags::FunctionOfTimeFile>() ==
        "FunctionOfTimeFile");
  CHECK(db::tag_name<importers::Tags::FunctionOfTimeNameMap>() ==
        "FunctionOfTimeNameMap");

  const std::string option_string{
      "SpecFuncOfTimeReader:\n"
      "  FunctionOfTimeFile: TestFile.h5\n"
      "  FunctionOfTimeNameMap: {Set1: Name1, Set2: Name2}"};
  using option_tags = tmpl::list<importers::OptionTags::FunctionOfTimeFile,
                                 importers::OptionTags::FunctionOfTimeNameMap>;
  Options<option_tags> options{""};
  options.parse(option_string);
  CHECK(options.get<importers::OptionTags::FunctionOfTimeFile>() ==
        "TestFile.h5");
  const auto& set_names =
      options.get<importers::OptionTags::FunctionOfTimeNameMap>();
  const std::map<std::string, std::string> expected_set_names{
      {"Set1", "Name1"}, {"Set2", "Name2"}};
  CHECK(set_names == expected_set_names);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.ReadSpecThirdOrderPiecewisePolynomial",
                  "[Unit][Evolution][Actions]") {
  domain::FunctionsOfTime::register_derived_with_charm();

  test_options();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  const size_t self_id{1};

  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestSpecFuncOfTimeData.h5"};
  constexpr uint32_t version_number = 4;
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

  const std::array<double, 2> expected_times{{0.0, 0.00599999974179903}};
  const std::array<std::string, 2> expected_names{
      {"ExpansionFactor", "RotationAngle"}};

  const std::vector<std::vector<double>> test_expansion{
      {5.9999997417990300e-03, 0.0, 1.0, 3.0, 1.0, 1.0, -1.4268296756999999e-06,
       0.0, 0.0},
      {5.9999997417990300e-03, 5.9999997417990300e-03, 1.0, 3.0, 1.0,
       9.9999999143902230e-01, -1.4268296756999999e-06, 0.0,
       -3.5943676411727592e-05}};
  const std::vector<std::string> expansion_legend{
      "Time", "TLastUpdate", "Nc",  "DerivOrder", "Version",
      "a",    "da",          "d2a", "d3a"};
  auto& expansion_file = test_file.insert<h5::Dat>(
      "/" + expected_names[0], expansion_legend, version_number);
  expansion_file.append(test_expansion);

  const std::vector<std::vector<double>> test_rotation{
      {5.9999997417990300e-03, 0.0, 1.0, 3.0, 1.0, 0.0, 1.3472907726000001e-02,
       0.0, 0.0},
      {5.9999997417990300e-03, 5.9999997417990300e-03, 1.0, 3.0, 1.0,
       8.0837442877282155e-05, 1.3472907726000001e-02, 0.0,
       1.4799266358888008e-04}};
  const std::vector<std::string> rotation_legend{
      "Time", "TLastUpdate", "Nc",    "DerivOrder", "Version",
      "Phi",  "dPhi",        "d2Phi", "d3Phi"};
  auto& rotation_file = test_file.insert<h5::Dat>(
      "/" + expected_names[1], rotation_legend, version_number);
  rotation_file.append(test_rotation);

  MockRuntimeSystem runner{
      {std::string{test_filename}, std::map<std::string, std::string>{
                                       {"ExpansionFactor", "ExpansionFactor"},
                                       {"RotationAngle", "RotationAngle"}}}};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      initial_functions_of_time{};

  const std::array<DataVector, 4> initial_coefficients{
      {{0.0}, {0.0}, {0.0}, {0.0}}};
  initial_functions_of_time["ExpansionFactor"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          0.0, initial_coefficients);
  initial_functions_of_time["RotationAngle"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          0.0, initial_coefficients);

  ActionTesting::emplace_component_and_initialize<component<Metavariables>>(
      &runner, self_id, {std::move(initial_functions_of_time)});

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  ActionTesting::next_action<component<Metavariables>>(make_not_null(&runner),
                                                       self_id);

  const auto& functions_of_time =
      ActionTesting::get_databox_tag<component<Metavariables>,
                                     domain::Tags::FunctionsOfTime>(runner,
                                                                    self_id);

  // Check that the FunctionOfTime and its derivatives have the expected
  // values
  const std::array<std::array<double, 3>, 2> expected_expansion{
      {{{test_expansion[0][5], test_expansion[0][6], test_expansion[0][7]}},
       {{test_expansion[1][5], test_expansion[1][6], test_expansion[1][7]}}}};
  const std::array<std::array<double, 3>, 2> expected_rotation{
      {{{test_rotation[0][5], test_rotation[0][6], test_rotation[0][7]}},
       {{test_rotation[1][5], test_rotation[1][6], test_rotation[1][7]}}}};
  std::unordered_map<std::string, std::array<std::array<double, 3>, 2>>
      expected_functions;
  expected_functions[expected_names[0]] = expected_expansion;
  expected_functions[expected_names[1]] = expected_rotation;

  REQUIRE(functions_of_time.size() == expected_names.size());

  for (const auto& function_of_time : functions_of_time) {
    const auto& f = function_of_time.second;
    const auto& name = function_of_time.first;
    // Check if the name is one of the expected names
    CHECK(std::find(expected_names.begin(), expected_names.end(), name) !=
          expected_names.end());

    for (size_t i = 0; i < 2; ++i) {
      const auto time = gsl::at(expected_times, i);
      const auto f_and_derivs = f->func_and_2_derivs(time);
      for (size_t j = 0; j < 3; ++j) {
        CHECK(gsl::at(gsl::at(expected_functions[name], i), j) ==
              approx(gsl::at(f_and_derivs, j)[0]));
      }
    }
  }

  // Delete the temporary file created for this test
  file_system::rm(test_filename, true);
}

// [[OutputRegex, Non-monotonic time found]]
SPECTRE_TEST_CASE(
    "Unit.IO.ReadSpecThirdOrderPiecewisePolynomialNonmonotonic",
    "[Unit][Evolution][Actions]") {
  ERROR_TEST();
  domain::FunctionsOfTime::register_derived_with_charm();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  const size_t self_id{1};

  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestSpecFuncOfTimeDataNonmonotonic.h5"};
  constexpr uint32_t version_number = 4;
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

  const std::vector<std::vector<double>> test_rotation{
      {0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.3472907726000001e-02, 0.0, 0.0},
      {-1.0, -1.0, 1.0, 3.0, 1.0, 8.0837442877282155e-05,
       1.3472907726000001e-02, 0.0, 1.4799266358888008e-04}};
  const std::vector<std::string> rotation_legend{
      "Time", "TLastUpdate", "Nc",    "DerivOrder", "Version",
      "Phi",  "dPhi",        "d2Phi", "d3Phi"};
  auto& rotation_file = test_file.insert<h5::Dat>(
      "/RotationAngle", rotation_legend, version_number);
  rotation_file.append(test_rotation);

  MockRuntimeSystem runner{
      {std::string{test_filename},
       std::map<std::string, std::string>{{"RotationAngle", "RotationAngle"}}}};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      initial_functions_of_time{};

  const std::array<DataVector, 4> initial_coefficients{
      {{0.0}, {0.0}, {0.0}, {0.0}}};
  initial_functions_of_time["RotationAngle"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          0.0, initial_coefficients);

  ActionTesting::emplace_component_and_initialize<component<Metavariables>>(
      &runner, self_id, {std::move(initial_functions_of_time)});

  runner.set_phase(Metavariables::Phase::Testing);
  runner.next_action<component<Metavariables>>(self_id);
}
