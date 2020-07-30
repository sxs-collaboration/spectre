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
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Importers/ElementActions.hpp"
#include "IO/Importers/ReadSpecThirdOrderPiecewisePolynomial.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Importers/VolumeDataReader.hpp"
#include "IO/Importers/VolumeDataReaderActions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ElementIdType = ElementId<2>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIdType;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
                     tmpl::list<domain::Tags::FunctionsOfTime>>,
                 importers::Actions::RegisterWithVolumeDataReader>>>;
};

template <typename Metavariables>
struct MockVolumeDataReader {
  using component_being_mocked = importers::VolumeDataReader<Metavariables>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<importers::detail::InitializeVolumeDataReader>>>;
};

struct Metavariables {
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<importers::Tags::FunctionOfTimeFile,
                 importers::Tags::FunctionOfTimeNameMap>;
  enum class Phase { Initialization, Testing };
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

struct TestCallback {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<domain::Tags::FunctionsOfTime,
                                              DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementIdType& /*array_index*/,
                    tuples::tagged_tuple_from_typelist<
                        tmpl::list<domain::Tags::FunctionsOfTime>>
                        functions_of_time_data) noexcept {
    // Get the expected FunctionsOfTime from the DataBox.
    const auto& expected_functions_of_time =
        get<domain::Tags::FunctionsOfTime>(box);

    const std::array<std::string, 2> expected_names{
        {"ExpansionFactor", "RotationAngle"}};
    constexpr size_t number_of_times = 3;
    const std::array<double, number_of_times> expected_times{{0.0, 0.1, 0.2}};

    // The callback action receives the imported functions of time as a
    // TaggedTuple.
    const auto& functions_of_time =
        get<domain::Tags::FunctionsOfTime>(functions_of_time_data);

    REQUIRE(expected_functions_of_time.size() == functions_of_time.size());
    REQUIRE(functions_of_time.size() == expected_names.size());

    // Check that the imported and expected FunctionsOfTime match
    for (const auto& function_of_time : functions_of_time) {
      const auto& f = function_of_time.second;
      const auto& name = function_of_time.first;
      // Check if the name is one of the expected names
      CHECK(std::find(expected_names.begin(), expected_names.end(), name) !=
            expected_names.end());

      for (size_t i = 0; i < number_of_times; ++i) {
        const auto time = gsl::at(expected_times, i);
        const auto f_and_derivs = f->func_and_2_derivs(time);
        const auto expected_f_and_derivs =
            expected_functions_of_time.at(name)->func_and_2_derivs(time);
        for (size_t j = 0; j < 3; ++j) {
          CHECK(gsl::at(expected_f_and_derivs, j)[0] ==
                approx(gsl::at(f_and_derivs, j)[0]));
        }
      }
    }
  }
};

template <bool TestNonmonotonic>
void test_reader() noexcept {
  constexpr size_t number_of_times = 3;
  const std::array<double, number_of_times> expected_times{{0.0, 0.1, 0.2}};
  const std::array<double, number_of_times> output_times =
      TestNonmonotonic ? std::array<double, number_of_times>{{0.0, -0.1, 0.2}}
                       : expected_times;
  const std::array<std::string, 2> expected_names{
      {"ExpansionFactor", "RotationAngle"}};

  std::array<DataVector, 4> initial_expansion{
      {{{1.0}}, {{0.2}}, {{0.03}}, {{0.004}}}};
  const std::array<DataVector, number_of_times - 1> next_expansion_third_deriv{
      {{{0.5}}, {{0.75}}}};
  domain::FunctionsOfTime::PiecewisePolynomial<3> expansion(expected_times[0],
                                                            initial_expansion);
  expansion.update(expected_times[1], next_expansion_third_deriv[0]);
  expansion.update(expected_times[2], next_expansion_third_deriv[1]);
  const std::array<std::array<DataVector, 3>, number_of_times - 1>&
      expansion_func_and_2_derivs_next{
          {expansion.func_and_2_derivs(expected_times[1]),
           expansion.func_and_2_derivs(expected_times[2])}};

  const std::array<DataVector, 4> initial_rotation{
      {{{2.0}}, {{-0.1}}, {{-0.02}}, {{-0.003}}}};
  const std::array<DataVector, number_of_times - 1> next_rotation_third_deriv{
      {{{-0.5}}, {{-0.75}}}};
  domain::FunctionsOfTime::PiecewisePolynomial<3> rotation(expected_times[0],
                                                           initial_rotation);
  rotation.update(expected_times[1], {{next_rotation_third_deriv[0]}});
  rotation.update(expected_times[2], {{next_rotation_third_deriv[1]}});
  const std::array<std::array<DataVector, 3>, number_of_times - 1>&
      rotation_func_and_2_derivs_next{
          {rotation.func_and_2_derivs(expected_times[1]),
           rotation.func_and_2_derivs(expected_times[2])}};

  const std::vector<std::vector<double>> test_expansion{
      {output_times[0], output_times[0], 1.0, 3.0, 1.0, initial_expansion[0][0],
       initial_expansion[1][0], initial_expansion[2][0],
       initial_expansion[3][0]},
      {output_times[1], output_times[1], 1.0, 3.0, 1.0,
       expansion_func_and_2_derivs_next[0][0][0],
       expansion_func_and_2_derivs_next[0][1][0],
       expansion_func_and_2_derivs_next[0][2][0],
       next_expansion_third_deriv[0][0]},
      {output_times[2], output_times[2], 1.0, 3.0, 1.0,
       expansion_func_and_2_derivs_next[1][0][0],
       expansion_func_and_2_derivs_next[1][1][0],
       expansion_func_and_2_derivs_next[1][2][0],
       next_expansion_third_deriv[1][0]}};
  const std::vector<std::string> expansion_legend{
      "Time", "TLastUpdate", "Nc",  "DerivOrder", "Version",
      "a",    "da",          "d2a", "d3a"};

  const std::vector<std::vector<double>> test_rotation{
      {output_times[0], output_times[0], 1.0, 3.0, 1.0, initial_rotation[0][0],
       initial_rotation[1][0], initial_rotation[2][0], initial_rotation[3][0]},
      {output_times[1], output_times[1], 1.0, 3.0, 1.0,
       rotation_func_and_2_derivs_next[0][0][0],
       rotation_func_and_2_derivs_next[0][1][0],
       rotation_func_and_2_derivs_next[0][2][0],
       next_rotation_third_deriv[0][0]},
      {output_times[2], output_times[2], 1.0, 3.0, 1.0,
       rotation_func_and_2_derivs_next[1][0][0],
       rotation_func_and_2_derivs_next[1][1][0],
       rotation_func_and_2_derivs_next[1][2][0],
       next_rotation_third_deriv[1][0]}};
  const std::vector<std::string> rotation_legend{
      "Time", "TLastUpdate", "Nc",    "DerivOrder", "Version",
      "Phi",  "dPhi",        "d2Phi", "d3Phi"};

  std::string test_filename{"TestSpecFuncOfTime.h5"};
  if (TestNonmonotonic) {
    test_filename = std::string{"TestSpecFuncOfTimeDataNonmonotonic.h5"};
  }
  constexpr uint32_t version_number = 4;
  {
    // Create a temporary file with test data to read in
    // First, check if the file exists, and delete it if so
    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }

    h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);
    auto& expansion_file = test_file.insert<h5::Dat>(
        "/" + expected_names[0], expansion_legend, version_number);
    expansion_file.append(test_expansion);
    auto& rotation_file = test_file.insert<h5::Dat>(
        "/" + expected_names[1], rotation_legend, version_number);
    rotation_file.append(test_rotation);
  }

  // Create the expected function_vs_time, for comparing to the imported data
  const std::array<std::array<double, 3>, number_of_times> expected_expansion{
      {{{test_expansion[0][5], test_expansion[0][6], test_expansion[0][7]}},
       {{test_expansion[1][5], test_expansion[1][6], test_expansion[1][7]}},
       {{test_expansion[2][5], test_expansion[2][6], test_expansion[2][7]}}}};
  const std::array<std::array<double, 3>, number_of_times> expected_rotation{
      {{{test_rotation[0][5], test_rotation[0][6], test_rotation[0][7]}},
       {{test_rotation[1][5], test_rotation[1][6], test_rotation[1][7]}},
       {{test_rotation[2][5], test_rotation[2][6], test_rotation[2][7]}}}};
  std::unordered_map<std::string,
                     std::array<std::array<double, 3>, number_of_times>>
      expected_functions;
  expected_functions[expected_names[0]] = expected_expansion;
  expected_functions[expected_names[1]] = expected_rotation;

  // Set up the MockRuntimeSystem runner
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{
      {std::string{test_filename}, std::map<std::string, std::string>{
                                       {"ExpansionFactor", "ExpansionFactor"},
                                       {"RotationAngle", "RotationAngle"}}}};
  // Setup mock data file reader
  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;
  ActionTesting::emplace_component<reader_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<reader_component>(make_not_null(&runner), 0);

  // Create a few elements with sample data
  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};

  for (const auto& id : element_ids) {
    // Create a FunctionsOfTime storing the expected piecewise polynomials
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        expected_functions_of_time;
    expected_functions_of_time["ExpansionFactor"] =
        static_cast<std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>(
            std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
                expansion));
    expected_functions_of_time["RotationAngle"] =
        static_cast<std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>(
            std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
                rotation));

    // Initialize the FunctionsOfTime on each element
    ActionTesting::emplace_component_and_initialize<element_array>(
        make_not_null(&runner), ElementIdType{id},
        {std::move(expected_functions_of_time)});

    // Register each element
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);

    // Invoke the simple_action RegisterElementWithSelf that was called on the
    // reader component by the RegisterWithVolumeDataReader action.
    runner.invoke_queued_simple_action<reader_component>(0);
  }

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  // Have the importer read the file and pass it to the callback
  runner.algorithms<reader_component>()
      .at(0)
      .template threaded_action<
          importers::ThreadedActions::ReadSpecThirdOrderPiecewisePolynomial<
              TestCallback, element_array>>();
  runner.invoke_queued_threaded_action<reader_component>(0);

  // Invoke the queued callbacks on the elements that test if the data is
  // correct
  for (const auto& id : element_ids) {
    runner.invoke_queued_simple_action<element_array>(id);
  }

  // Delete the temporary file created for this test
  file_system::rm(test_filename, true);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.ReadSpecThirdOrderPiecewisePolynomial",
                  "[Unit][Evolution][Actions]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_options();
  test_reader<false>();
}

// [[OutputRegex, Non-monotonic time found]]
SPECTRE_TEST_CASE("Unit.IO.ReadSpecThirdOrderPiecewisePolynomialNonmonotonic",
                  "[Unit][Evolution][Actions]") {
  ERROR_TEST();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_reader<true>();
}
