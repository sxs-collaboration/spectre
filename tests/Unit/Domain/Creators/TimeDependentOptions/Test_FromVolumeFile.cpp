// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void test(const std::string& function_of_time_name) {
  const std::string filename{"HorseRadish.h5"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  const std::string subfile_name{"VolumeData"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[function_of_time_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{3, 0.0}, DataVector{3, 1.0},
                     DataVector{3, 0.0}},
          100.0);

  TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                   functions_of_time);

  const double time = 50.0;

  const auto from_volume_file = TestHelpers::test_creation<
      domain::creators::time_dependent_options::FromVolumeFile>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name);

  const domain::FunctionsOfTimeMap fot_from_file =
      from_volume_file.retrieve_function_of_time({function_of_time_name}, time);

  std::array<DataVector, 3> expected_values{
      DataVector{3, 1.0 * time}, DataVector{3, 1.0}, DataVector{3, 0.0}};

  CHECK(dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>&>(
            *fot_from_file.at(function_of_time_name)) ==
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>&>(
            *functions_of_time.at(function_of_time_name)));

  const auto from_volume_file_replay =
      domain::creators::time_dependent_options::FromVolumeFile(
          filename, subfile_name, true);

  const domain::FunctionsOfTimeMap fot_from_file_replay =
      from_volume_file.retrieve_function_of_time({function_of_time_name}, time);

  CHECK(dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>&>(
            *fot_from_file_replay.at(function_of_time_name)) ==
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>&>(
            *functions_of_time.at(function_of_time_name)));

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

void test_errors() {
  const std::string filename{"HorseRadishErrors.h5"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  std::string subfile_name{"VolumeData"};

  {
    h5::H5File<h5::AccessType::ReadWrite> h5_file{filename};
    h5_file.insert<h5::VolumeData>(subfile_name);
  }

  using FromVolumeFile =
      domain::creators::time_dependent_options::FromVolumeFile;

  auto from_volume_file = TestHelpers::test_creation<FromVolumeFile>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name);

  // Need new subfile to write to
  subfile_name += "0";
  TestHelpers::domain::creators::write_volume_data(filename, subfile_name);

  from_volume_file = TestHelpers::test_creation<FromVolumeFile>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name);

  CHECK_THROWS_WITH(
      (from_volume_file.retrieve_function_of_time({"Expansion"}, 0.0)),
      Catch::Matchers::ContainsSubstring(
          "There are no functions of time in the subfile "));

  subfile_name += "0";
  from_volume_file = TestHelpers::test_creation<FromVolumeFile>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name);
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Translation"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{3, 0.0}, DataVector{3, 1.0},
                     DataVector{3, 0.0}},
          100.0);

  TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                   functions_of_time);

  CHECK_THROWS_WITH(
      (from_volume_file.retrieve_function_of_time({"Expansion"}, 0.0)),
      Catch::Matchers::ContainsSubstring(
          "No function of time named Expansion in the subfile "));

  subfile_name += "0";
  from_volume_file = TestHelpers::test_creation<FromVolumeFile>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name);
  functions_of_time.clear();
  functions_of_time["Expansion"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{3, 0.0}, DataVector{3, 1.0},
                     DataVector{3, 0.0}},
          1.0);

  TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                   functions_of_time);

  CHECK_THROWS_WITH(
      (from_volume_file.retrieve_function_of_time({"Expansion"}, 10.0)),
      Catch::Matchers::ContainsSubstring("Expansion: The requested time") and
          Catch::Matchers::ContainsSubstring(
              "is out of the range of the function of time"));

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.FromVolumeFile",
                  "[Unit][Domain]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test("Translation");
  test_errors();
}
}  // namespace
