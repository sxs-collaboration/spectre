// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/OptionTags.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/ReadSpecPiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct Metavariables {
  static constexpr size_t volume_dim = 3;
};

class FakeCreator : public DomainCreator<3> {
 public:
  explicit FakeCreator() {}
  Domain<3> create_domain() const override { ERROR(""); }
  std::vector<std::array<size_t, 3>> initial_extents() const override {
    ERROR("");
  }
  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const override {
    ERROR("");
  }
  auto functions_of_time(const std::unordered_map<std::string, double>&
                         /*initial_expiration_times*/
                         = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        result{};

    // All initial parameters must match the values below
    const double initial_time = 0.0;
    std::array<DataVector, 4> initial_expansion{
        {{{1.0}}, {{0.2}}, {{0.03}}, {{0.004}}}};
    std::array<DataVector, 4> initial_unity{
        {{{1.0}}, {{0.0}}, {{0.0}}, {{0.0}}}};
    const std::array<DataVector, 4> initial_rotation{{{3, 0.0},
                                                      {{0.0, 0.0, -0.1}},
                                                      {{0.0, 0.0, -0.02}},
                                                      {{0.0, 0.0, -0.003}}}};
    const std::array<DataVector, 1> initial_quaternion{{{1.0, 0.0, 0.0, 0.0}}};

    result.insert(
        {"CubicScaleA",
         std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
             initial_time, initial_expansion,
             std::numeric_limits<double>::infinity())});
    result.insert(
        {"CubicScaleB",
         std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
             initial_time, initial_unity,
             std::numeric_limits<double>::infinity())});
    result.insert(
        {"Rotation",
         std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
             initial_time, initial_quaternion, initial_rotation,
             std::numeric_limits<double>::infinity())});
    return result;
  }
};

void test_options() {
  CHECK(
      db::tag_name<domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile>() ==
      "FunctionOfTimeFile");
  CHECK(db::tag_name<
            domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>() ==
        "FunctionOfTimeNameMap");

  using option_tags =
      tmpl::list<domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
                 domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>;
  {
    const std::string option_string{
        "CubicFunctionOfTimeOverride:\n"
        "  FunctionOfTimeFile: TestFile.h5\n"
        "  FunctionOfTimeNameMap: {Set1: Name1, Set2: Name2}"};
    Options::Parser<option_tags> options{""};
    options.parse(option_string);
    CHECK(*std::optional<std::string>(
              options.get<
                  domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile>()) ==
          "TestFile.h5"s);
    const std::map<std::string, std::string> expected_set_names{
        {"Set1", "Name1"}, {"Set2", "Name2"}};
    CHECK(options.get<
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>() ==
          expected_set_names);
  }
  {
    const std::string option_string{
        "CubicFunctionOfTimeOverride:\n"
        "  FunctionOfTimeFile: None\n"
        "  FunctionOfTimeNameMap: {}"};
    Options::Parser<option_tags> options{""};
    options.parse(option_string);
    CHECK(std::optional<std::string>(
              options.get<
                  domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile>()) ==
          std::nullopt);
    const std::map<std::string, std::string> expected_set_names{};
    CHECK(options.get<
              domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>() ==
          expected_set_names);
  }
}

void test_errors();
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.ReadSpecPiecewisePolynomial",
                  "[Unit][Evolution][Actions]") {
  test_options();
  test_errors();

  // Create a temporary file with test data to read in
  // First, check if the file exists, and delete it if so
  const std::string test_filename{"TestSpecFuncOfTimeData.h5"};
  constexpr uint32_t version_number = 4;
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

  constexpr size_t number_of_times = 3;
  const std::array<double, number_of_times> expected_times{{0.0, 0.1, 0.2}};
  const std::array<std::string, 3> source_names{
      {"ExpansionFactor", "RotationAngle", "Unity"}};

  std::array<DataVector, 4> initial_expansion{
      {{{1.0}}, {{0.2}}, {{0.03}}, {{0.004}}}};
  const std::array<DataVector, number_of_times - 1> next_expansion_third_deriv{
      {{{0.5}}, {{0.75}}}};
  domain::FunctionsOfTime::PiecewisePolynomial<3> expansion(
      expected_times[0], initial_expansion, expected_times[0]);
  expansion.update(expected_times[1], next_expansion_third_deriv[0],
                   expected_times[1]);
  expansion.update(expected_times[2], next_expansion_third_deriv[1],
                   expected_times[2]);
  const std::array<std::array<DataVector, 3>, number_of_times - 1>&
      expansion_func_and_2_derivs_next{
          {expansion.func_and_2_derivs(expected_times[1]),
           expansion.func_and_2_derivs(expected_times[2])}};

  const std::array<DataVector, 4> initial_rotation{
      {{{2.0}}, {{-0.1}}, {{-0.02}}, {{-0.003}}}};
  const std::array<DataVector, number_of_times - 1> next_rotation_fourth_deriv{
      {{{-0.5}}, {{-0.75}}}};
  domain::FunctionsOfTime::PiecewisePolynomial<3> rotation(
      expected_times[0], initial_rotation, expected_times[0]);
  rotation.update(expected_times[1], {{next_rotation_fourth_deriv[0]}},
                  expected_times[1]);
  rotation.update(expected_times[2], {{next_rotation_fourth_deriv[1]}},
                  expected_times[2]);
  const std::array<std::array<DataVector, 3>, number_of_times - 1>&
      rotation_func_and_2_derivs_next{
          {rotation.func_and_2_derivs(expected_times[1]),
           rotation.func_and_2_derivs(expected_times[2])}};

  const std::vector<std::vector<double>> test_expansion{
      {expected_times[0], expected_times[0], 1.0, 3.0, 1.0,
       initial_expansion[0][0], initial_expansion[1][0],
       initial_expansion[2][0], initial_expansion[3][0]},
      {expected_times[1], expected_times[1], 1.0, 3.0, 1.0,
       expansion_func_and_2_derivs_next[0][0][0],
       expansion_func_and_2_derivs_next[0][1][0],
       expansion_func_and_2_derivs_next[0][2][0],
       next_expansion_third_deriv[0][0]},
      {expected_times[2], expected_times[2], 1.0, 3.0, 1.0,
       expansion_func_and_2_derivs_next[1][0][0],
       expansion_func_and_2_derivs_next[1][1][0],
       expansion_func_and_2_derivs_next[1][2][0],
       next_expansion_third_deriv[1][0]}};
  const std::vector<std::string> expansion_legend{
      "Time", "TLastUpdate", "Nc",  "DerivOrder", "Version",
      "a",    "da",          "d2a", "d3a"};
  test_file.close_current_object();
  auto& expansion_file = test_file.insert<h5::Dat>(
      "/" + source_names[0], expansion_legend, version_number);
  expansion_file.append(test_expansion);

  const std::vector<std::vector<double>> test_rotation{
      {expected_times[0], expected_times[0], 1.0, 3.0, 1.0,
       initial_rotation[0][0], initial_rotation[1][0], initial_rotation[2][0],
       initial_rotation[3][0]},
      {expected_times[1], expected_times[1], 1.0, 3.0, 1.0,
       rotation_func_and_2_derivs_next[0][0][0],
       rotation_func_and_2_derivs_next[0][1][0],
       rotation_func_and_2_derivs_next[0][2][0],
       next_rotation_fourth_deriv[0][0]},
      {expected_times[2], expected_times[2], 1.0, 3.0, 1.0,
       rotation_func_and_2_derivs_next[1][0][0],
       rotation_func_and_2_derivs_next[1][1][0],
       rotation_func_and_2_derivs_next[1][2][0],
       next_rotation_fourth_deriv[1][0]}};
  const std::vector<std::string> rotation_legend{
      "Time", "TLastUpdate", "Nc",    "DerivOrder", "Version",
      "Phi",  "dPhi",        "d2Phi", "d3Phi"};
  test_file.close_current_object();
  auto& rotation_file = test_file.insert<h5::Dat>(
      "/" + source_names[1], rotation_legend, version_number);
  rotation_file.append(test_rotation);

  const std::vector<std::vector<double>> test_unity{
      {expected_times[0], expected_times[0], 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0},
      {expected_times[1], expected_times[1], 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0},
      {expected_times[2], expected_times[2], 1.0, 3.0, 1.0, 1.0, 0.0, 0.0,
       0.0}};
  const std::vector<std::string> unity_legend{
      "Time",  "TLastUpdate", "Nc",      "DerivOrder", "Version",
      "Unity", "dUnity",      "d2Unity", "d3Unity"};
  test_file.close_current_object();
  auto& unity_file = test_file.insert<h5::Dat>("/" + source_names[2],
                                               unity_legend, version_number);
  unity_file.append(test_unity);

  // These names must match the hard coded ones in their respective
  // TimeDependences
  const std::string expansion_name = "CubicScaleA";
  const std::string unity_name = "CubicScaleB";
  const std::string rotation_name = "Rotation";

  const std::map<std::string, std::string> expected_set_names{
      {"ExpansionFactor", expansion_name},
      {"Unity", unity_name},
      {"RotationAngle", rotation_name}};

  const auto created_domain_creator = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<3, domain::creators::Brick>>(
      "Brick:\n"
      "  LowerBound: [-4.0, -5.0, -6.0]\n"
      "  UpperBound: [6.0, 5.0, 4.0]\n"
      "  IsPeriodicIn: [false, false, false]\n"
      "  InitialRefinement: [0, 0, 0]\n"
      "  InitialGridPoints: [5, 5, 5]\n"
      "  TimeDependence:\n"
      "    ScalingAndZRotation:\n"
      "      InitialTime: 0.0\n"
      "      InitialExpansion: [1.0, 1.0]\n"
      "      UseLinearScaling: false\n"
      "      Velocity: [0.0, 0.0]\n"
      "      Acceleration: [0.0, 0.0]\n"
      "      OuterBoundary: 839.661030811156\n"
      "      AngularVelocity: 0.0\n");
  const auto created_function_of_time_file = TestHelpers::test_option_tag<
      domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile>(test_filename);
  const auto created_function_of_time_name_map = TestHelpers::test_option_tag<
      domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>(
      "{ExpansionFactor: " + expansion_name + ", Unity: " + unity_name +
      ", RotationAngle: " + rotation_name + "}");

  // Check that the FunctionOfTime and its derivatives have the expected
  // values
  const std::array<std::array<double, 3>, number_of_times> expected_expansion{
      {{{test_expansion[0][5], test_expansion[0][6], test_expansion[0][7]}},
       {{test_expansion[1][5], test_expansion[1][6], test_expansion[1][7]}},
       {{test_expansion[2][5], test_expansion[2][6], test_expansion[2][7]}}}};
  const std::array<std::array<double, 3>, number_of_times> expected_rotation{
      {{{test_rotation[0][5], test_rotation[0][6], test_rotation[0][7]}},
       {{test_rotation[1][5], test_rotation[1][6], test_rotation[1][7]}},
       {{test_rotation[2][5], test_rotation[2][6], test_rotation[2][7]}}}};
  const std::array<std::array<double, 3>, number_of_times> expected_unity{
      {{{test_unity[0][5], test_unity[0][6], test_unity[0][7]}},
       {{test_unity[1][5], test_unity[1][6], test_unity[1][7]}},
       {{test_unity[2][5], test_unity[2][6], test_unity[2][7]}}}};
  std::unordered_map<std::string,
                     std::array<std::array<double, 3>, number_of_times>>
      expected_functions;
  expected_functions[expansion_name] = expected_expansion;
  expected_functions[rotation_name] = expected_rotation;
  expected_functions[unity_name] = expected_unity;

  const std::array<std::string, 3> expected_names{
      {expansion_name, rotation_name, unity_name}};

  const auto check_read_functions_of_time =
      [&expected_names, &expected_times](
          const std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
              functions_of_time,
          const std::unordered_map<
              std::string, std::array<std::array<double, 3>, number_of_times>>&
              expected_funcs,
          const bool quaternion_rotation = false) {
        REQUIRE(functions_of_time.size() == expected_names.size());
        for (const auto& function_of_time : functions_of_time) {
          const auto& f = function_of_time.second;
          const auto& name = function_of_time.first;
          // Check if the name is one of the expected names
          CHECK(std::find(expected_names.begin(), expected_names.end(), name) !=
                expected_names.end());

          // Only convert to QuatFot for specific name
          const bool using_quaternion_fot =
              quaternion_rotation and name == "Rotation";
          // Check if the read FunctionOfTimes have the expected values, derivs
          for (size_t i = 0; i < number_of_times; ++i) {
            const auto time = gsl::at(expected_times, i);
            std::array<DataVector, 3> f_and_derivs{};
            if (using_quaternion_fot) {
              // Need the angle func, not the quaternion to check
              auto* quaternion_fot = dynamic_cast<
                  domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
                  f.get());
              f_and_derivs = quaternion_fot->angle_func_and_2_derivs(time);
            } else {
              f_and_derivs = f->func_and_2_derivs(time);
            }
            // z-rotation if using quaternions
            const size_t index = using_quaternion_fot ? 2 : 0;
            for (size_t j = 0; j < 3; ++j) {
              CHECK(gsl::at(gsl::at(expected_funcs.at(name), i), j) ==
                    approx(gsl::at(f_and_derivs, j)[index]));
            }
          }
        }
      };

  {
    INFO("Override all FunctionsOfTime");
    const auto& functions_of_time =
        domain::Tags::FunctionsOfTimeInitialize::create_from_options<
            Metavariables>(created_domain_creator,
                           created_function_of_time_file,
                           created_function_of_time_name_map);
    check_read_functions_of_time(functions_of_time, expected_functions);
  }

  {
    INFO("Override with QuaternionFunctionOfTime")
    const std::unique_ptr<DomainCreator<3>> fake_creator =
        std::make_unique<FakeCreator>();
    const auto& functions_of_time =
        domain::Tags::FunctionsOfTimeInitialize::create_from_options<
            Metavariables>(fake_creator, created_function_of_time_file,
                           created_function_of_time_name_map);
    check_read_functions_of_time(functions_of_time, expected_functions, true);
  }

  {
    INFO("Override only expansion");
    // Read the file again, but this time, only override one FunctionOfTime
    const auto created_function_of_time_name_map_expansion =
        TestHelpers::test_option_tag<
            domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>(
            "{ExpansionFactor: " + expansion_name + "}");
    const auto& functions_of_time_expansion =
        domain::Tags::FunctionsOfTimeInitialize::create_from_options<
            Metavariables>(created_domain_creator,
                           created_function_of_time_file,
                           created_function_of_time_name_map_expansion);

    // Only override ExpansionFactor this time, so change the expected Rotation
    // and Unity FunctionsOfTime to their initial values
    expected_functions[rotation_name] =
        std::array<std::array<double, 3>, number_of_times>{
            {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}};
    expected_functions[unity_name] =
        std::array<std::array<double, 3>, number_of_times>{
            {{{1.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}}};
    check_read_functions_of_time(functions_of_time_expansion,
                                 expected_functions);
  }

  // Delete the temporary file created for this test
  file_system::rm(test_filename, true);
}

namespace {
void test_errors() {
  CHECK_THROWS_WITH(
      ([]() {
        // Create a temporary file with test data to read in
        // First, check if the file exists, and delete it if so
        const std::string test_filename{
            "TestSpecFuncOfTimeDataNonmonotonic.h5"};
        const std::map<std::string, std::string> test_name_map{
            {{"RotationAngle", "RotationAngle"}}};
        constexpr uint32_t version_number = 4;
        if (file_system::check_if_file_exists(test_filename)) {
          file_system::rm(test_filename, true);
        }

        h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

        constexpr size_t number_of_times = 3;
        const std::array<double, number_of_times> expected_times{
            {0.0, 0.1, 0.2}};
        const std::string expected_name{"RotationAngle"};

        const std::array<DataVector, 4> initial_rotation{
            {{{2.0}}, {{-0.1}}, {{-0.02}}, {{-0.003}}}};
        const std::array<DataVector, number_of_times - 1>
            next_rotation_fourth_deriv{{{{-0.5}}, {{-0.75}}}};
        domain::FunctionsOfTime::PiecewisePolynomial<3> rotation(
            expected_times[0], initial_rotation, expected_times[0]);
        rotation.update(expected_times[1], {{next_rotation_fourth_deriv[0]}},
                        expected_times[1]);
        rotation.update(expected_times[2], {{next_rotation_fourth_deriv[1]}},
                        expected_times[2]);
        const std::array<std::array<DataVector, 3>, number_of_times - 1>&
            rotation_func_and_2_derivs_next{
                {rotation.func_and_2_derivs(expected_times[1]),
                 rotation.func_and_2_derivs(expected_times[2])}};

        const std::vector<std::vector<double>> test_rotation{
            {expected_times[0], expected_times[0], 1.0, 3.0, 1.0,
             initial_rotation[0][0], initial_rotation[1][0],
             initial_rotation[2][0], initial_rotation[3][0]},
            {expected_times[2], expected_times[2], 1.0, 3.0, 1.0,
             rotation_func_and_2_derivs_next[0][0][0],
             rotation_func_and_2_derivs_next[0][1][0],
             rotation_func_and_2_derivs_next[0][2][0],
             next_rotation_fourth_deriv[0][0]},
            {expected_times[1], expected_times[1], 1.0, 3.0, 1.0,
             rotation_func_and_2_derivs_next[1][0][0],
             rotation_func_and_2_derivs_next[1][1][0],
             rotation_func_and_2_derivs_next[1][2][0],
             next_rotation_fourth_deriv[1][0]}};
        const std::vector<std::string> rotation_legend{
            "Time", "TLastUpdate", "Nc",    "DerivOrder", "Version",
            "Phi",  "dPhi",        "d2Phi", "d3Phi"};
        auto& rotation_file = test_file.insert<h5::Dat>(
            "/" + expected_name, rotation_legend, version_number);
        rotation_file.append(test_rotation);

        std::unordered_map<std::string,
                           domain::FunctionsOfTime::PiecewisePolynomial<3>>
            spec_functions_of_time{};
        // Attempt to read in the file: will trigger error that file contains
        // a non-monotonic time step
        domain::FunctionsOfTime::read_spec_piecewise_polynomial(
            make_not_null(&spec_functions_of_time), test_filename,
            test_name_map);
      }()),
      Catch::Contains("Non-monotonic time found"));

  CHECK_THROWS_WITH(
      ([]() {
        // Each row in the SpEC data read by
        // read_spec_piecewise_polynomial() should have the same values in
        // column 2 (number of components), column 3 (max deriv order), or
        // column 4 (version). This test checks that an appropriate ERROR()
        // triggers if one of these (randomly chosen) is not satisfied.

        // Create a temporary file with test data to read in
        // First, check if the file exists, and delete it if so
        const std::string test_filename{"TestSpecFuncOfTimeDataCols234.h5"};
        const std::map<std::string, std::string> test_name_map{
            {{"RotationAngle", "RotationAngle"}}};
        constexpr uint32_t version_number = 4;
        if (file_system::check_if_file_exists(test_filename)) {
          file_system::rm(test_filename, true);
        }

        h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename);

        constexpr size_t number_of_times = 3;
        const std::array<double, number_of_times> expected_times{
            {0.0, 0.1, 0.2}};
        const std::string expected_name{"RotationAngle"};

        const std::array<DataVector, 4> initial_rotation{
            {{{2.0}}, {{-0.1}}, {{-0.02}}, {{-0.003}}}};
        const std::array<DataVector, number_of_times - 1>
            next_rotation_third_deriv{{{{-0.5}}, {{-0.75}}}};
        domain::FunctionsOfTime::PiecewisePolynomial<3> rotation(
            expected_times[0], initial_rotation, expected_times[0]);
        rotation.update(expected_times[1], {{next_rotation_third_deriv[0]}},
                        expected_times[1]);
        rotation.update(expected_times[2], {{next_rotation_third_deriv[1]}},
                        expected_times[2]);
        const std::array<std::array<DataVector, 3>, number_of_times - 1>&
            rotation_func_and_2_derivs_next{
                {rotation.func_and_2_derivs(expected_times[1]),
                 rotation.func_and_2_derivs(expected_times[2])}};

        // Select which of columns 2, 3, 4 to make unequal.
        const std::array<double, 3> columns_234{{1.0, 3.0, 1.0}};
        std::array<double, 3> columns_234_row1{columns_234};
        MAKE_GENERATOR(gen);
        std::uniform_int_distribution<> dist(0, 2);
        gsl::at(columns_234_row1, dist(gen)) += 1.0;

        const std::vector<std::vector<double>> test_rotation{
            {expected_times[0], expected_times[0], columns_234[0],
             columns_234[1], columns_234[2], initial_rotation[0][0],
             initial_rotation[1][0], initial_rotation[2][0],
             initial_rotation[3][0]},
            {expected_times[1], expected_times[1], columns_234_row1[0],
             columns_234_row1[1], columns_234_row1[2],
             rotation_func_and_2_derivs_next[0][0][0],
             rotation_func_and_2_derivs_next[0][1][0],
             rotation_func_and_2_derivs_next[0][2][0],
             next_rotation_third_deriv[0][0]},
            {expected_times[2], expected_times[2], columns_234[0],
             columns_234[1], columns_234[2],
             rotation_func_and_2_derivs_next[1][0][0],
             rotation_func_and_2_derivs_next[1][1][0],
             rotation_func_and_2_derivs_next[1][2][0],
             next_rotation_third_deriv[1][0]}};
        const std::vector<std::string> rotation_legend{
            "Time", "TLastUpdate", "Nc",    "DerivOrder", "Version",
            "Phi",  "dPhi",        "d2Phi", "d3Phi"};
        auto& rotation_file = test_file.insert<h5::Dat>(
            "/" + expected_name, rotation_legend, version_number);
        rotation_file.append(test_rotation);

        std::unordered_map<std::string,
                           domain::FunctionsOfTime::PiecewisePolynomial<3>>
            spec_functions_of_time{};
        // Attempt to read in the file: will trigger error that column 2, 3, or
        // 4 should be constant, but one isn't
        domain::FunctionsOfTime::read_spec_piecewise_polynomial(
            make_not_null(&spec_functions_of_time), test_filename,
            test_name_map);
      }()),
      Catch::Contains("Values in column 2"));
}
}  // namespace
