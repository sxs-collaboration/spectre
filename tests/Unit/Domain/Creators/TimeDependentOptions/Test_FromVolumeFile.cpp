// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestingFramework.hpp"

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <memory>
#include <string>
#include <unordered_map>

#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestCreation.hpp"
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
void write_volume_data(
    const std::string& filename, const std::string& subfile_name,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {}) {
  h5::H5File<h5::AccessType::ReadWrite> h5_file{filename, true};
  auto& vol_file = h5_file.insert<h5::VolumeData>(subfile_name);

  // We don't care about the volume data here, just the functions of time
  vol_file.write_volume_data(
      0, 0.0,
      {ElementVolumeData{"blah",
                         {TensorComponent{"RandomTensor", DataVector{3, 0.0}}},
                         {3},
                         {Spectral::Basis::Legendre},
                         {Spectral::Quadrature::GaussLobatto}}},
      std::nullopt,
      functions_of_time.empty() ? std::nullopt
                                : std::optional{serialize(functions_of_time)});
}

template <typename FoTName>
void test() {
  const std::string filename{"HorseRadish.h5"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  const std::string subfile_name{"VolumeData"};
  const std::string function_of_time_name = pretty_type::name<FoTName>();

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[function_of_time_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{3, 0.0}, DataVector{3, 1.0},
                     DataVector{3, 0.0}},
          100.0);

  write_volume_data(filename, subfile_name, functions_of_time);

  const double time = 50.0;

  const auto from_volume_file = TestHelpers::test_creation<
      domain::creators::time_dependent_options::FromVolumeFile<FoTName>>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name +
      "\nTime: 50.0\n");

  std::array<DataVector, 3> expected_values{
      DataVector{3, 1.0 * time}, DataVector{3, 1.0}, DataVector{3, 0.0}};

  CHECK(from_volume_file.values == expected_values);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

template <>
void test<domain::creators::time_dependent_options::names::Expansion>() {
  const std::string filename{"SpicyHorseRadish.h5"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  const std::string subfile_name{"VolumeData"};
  const std::string function_of_time_name{"Expansion"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[function_of_time_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{1, 0.0}, DataVector{1, 1.0},
                     DataVector{1, 0.0}},
          100.0);
  const double velocity = -1e-5;
  const double decay_timescale = 50.0;
  functions_of_time[function_of_time_name + "OuterBoundary"] =
      std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
          1.0, 0.0, velocity, decay_timescale);

  write_volume_data(filename, subfile_name, functions_of_time);

  // Makes things easier
  const double time = decay_timescale;

  const auto from_volume_file = TestHelpers::test_creation<
      domain::creators::time_dependent_options::FromVolumeFile<
          domain::creators::time_dependent_options::names::Expansion>>(
      "H5Filename: " + filename + "\nSubfileName: " + subfile_name +
      "\nTime: 50.0\n");

  std::array<DataVector, 3> expected_values{DataVector{1.0 * time},
                                            DataVector{1.0}, DataVector{0.0}};
  // Comes from the FixedSpeedCubic formula
  std::array<DataVector, 3> expected_values_outer_boundary{
      DataVector{1.0 + velocity * cube(time) /
                           (square(decay_timescale) + square(time))},
      DataVector{velocity}, DataVector{0.01 * velocity}};

  CHECK(from_volume_file.expansion_values == expected_values);
  CHECK_ITERABLE_APPROX(from_volume_file.expansion_values_outer_boundary,
                        expected_values_outer_boundary);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

template <>
void test<domain::creators::time_dependent_options::names::Rotation>() {
  const std::string filename{"ExtraSpicyHorseRadish.h5"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  const std::string subfile_name{"VolumeData"};
  const std::string function_of_time_name{"Rotation"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[function_of_time_name] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          0.0, std::array{DataVector{1.0, 0.0, 0.0, 0.0}},
          std::array{DataVector{3, 2.0}, DataVector{3, 1.0}, DataVector{3, 0.0},
                     DataVector{3, 0.0}},
          100.0);

  write_volume_data(filename, subfile_name, functions_of_time);

  {
    INFO("Is a QuaternionFunctionOfTime");
    // Going at t=0 is easier for checking quaternions
    const auto from_volume_file = TestHelpers::test_creation<
        domain::creators::time_dependent_options::FromVolumeFile<
            domain::creators::time_dependent_options::names::Rotation>>(
        "H5Filename: " + filename + "\nSubfileName: " + subfile_name +
        "\nTime: 0.0\n");

    // q
    // dtq = 0.5 * q * omega
    // d2tq = 0.5 * (dtq * omega + q * dtomega)
    std::array<DataVector, 3> expected_quaternion{
        DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{0.0, 0.5, 0.5, 0.5},
        DataVector{-0.75, 0.0, 0.0, 0.0}};
    std::array<DataVector, 4> expected_angle{
        DataVector{3, 2.0}, DataVector{3, 1.0}, DataVector{3, 0.0},
        DataVector{3, 0.0}};

    CHECK(from_volume_file.quaternions == expected_quaternion);
    CHECK(from_volume_file.angle_values == expected_angle);
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  // Write new function of time
  {
    auto quat_and_derivs =
        functions_of_time.at(function_of_time_name)->func_and_2_derivs(0.0);
    functions_of_time.clear();
    functions_of_time[function_of_time_name] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0, std::move(quat_and_derivs), 100.0);
    write_volume_data(filename, subfile_name, functions_of_time);
  }

  {
    INFO("Is not a QuaternionFunctionOfTime");
    // Going at t=0 is easier for checking quaternions
    const auto from_volume_file = TestHelpers::test_creation<
        domain::creators::time_dependent_options::FromVolumeFile<
            domain::creators::time_dependent_options::names::Rotation>>(
        "H5Filename: " + filename + "\nSubfileName: " + subfile_name +
        "\nTime: 0.0\n");

    // q
    // dtq = 0.5 * q * omega
    // d2tq = 0.5 * (dtq * omega + q * dtomega)
    std::array<DataVector, 3> expected_quaternion{
        DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{0.0, 0.5, 0.5, 0.5},
        DataVector{-0.75, 0.0, 0.0, 0.0}};
    std::array<DataVector, 4> expected_angle{
        DataVector{3, 0.0}, DataVector{3, 1.0}, DataVector{3, 0.0},
        DataVector{3, 0.0}};

    CHECK(from_volume_file.quaternions == expected_quaternion);
    CHECK(from_volume_file.angle_values == expected_angle);
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

template <>
void test<domain::creators::time_dependent_options::names::ShapeSize<
    domain::ObjectLabel::B>>() {
  const std::string filename{"BlandHorseRadish.h5"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  const std::string subfile_name{"VolumeData"};
  const std::string shape_name{"ShapeB"};
  const std::string size_name{"SizeB"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  // For reading these in, we don't care how many components of the DataVector
  // there are. Normally there'd be a lot more.
  functions_of_time[shape_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{1, 0.0}, DataVector{1, 0.0},
                     DataVector{1, 2.0}},
          100.0);
  functions_of_time[size_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          0.0,
          std::array{DataVector{1, 0.1}, DataVector{1, 0.0}, DataVector{1, 0.3},
                     DataVector{1, 0.0}},
          100.0);

  write_volume_data(filename, subfile_name, functions_of_time);

  const auto from_volume_file = TestHelpers::test_creation<
      domain::creators::time_dependent_options::FromVolumeFile<
          domain::creators::time_dependent_options::names::ShapeSize<
              domain::ObjectLabel::B>>>("H5Filename: " + filename +
                                        "\nSubfileName: " + subfile_name +
                                        "\nTime: 50.0\n");

  std::array<DataVector, 3> expected_shape_values{
      DataVector{2500.0}, DataVector{100.0}, DataVector{2.0}};
  std::array<DataVector, 4> expected_size_values{
      DataVector{0.1 + 0.5 * 2500.0 * 0.3}, DataVector{50.0 * 0.3},
      DataVector{0.3}, DataVector{0.0}};

  CHECK(from_volume_file.shape_values == expected_shape_values);
  CHECK_ITERABLE_APPROX(from_volume_file.size_values, expected_size_values);

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
      domain::creators::time_dependent_options::FromVolumeFile<
          domain::creators::time_dependent_options::names::Expansion>;

  CHECK_THROWS_WITH(
      (FromVolumeFile{filename, subfile_name, 0.0}),
      Catch::Matchers::ContainsSubstring(
          "Expansion: There are no observation IDs in the subfile "));

  // Need new subfile to write to
  subfile_name += "0";
  write_volume_data(filename, subfile_name);

  CHECK_THROWS_WITH(
      (FromVolumeFile{filename, subfile_name, 0.0}),
      Catch::Matchers::ContainsSubstring(
          "Expansion: There are no functions of time in the subfile "));

  subfile_name += "0";
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Translation"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{3, 0.0}, DataVector{3, 1.0},
                     DataVector{3, 0.0}},
          100.0);

  write_volume_data(filename, subfile_name, functions_of_time);

  CHECK_THROWS_WITH((FromVolumeFile{filename, subfile_name, 0.1}),
                    Catch::Matchers::ContainsSubstring(
                        "No function of time named Expansion in the subfile "));

  subfile_name += "0";
  functions_of_time.clear();
  functions_of_time["Expansion"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0,
          std::array{DataVector{3, 0.0}, DataVector{3, 1.0},
                     DataVector{3, 0.0}},
          1.0);

  write_volume_data(filename, subfile_name, functions_of_time);

  CHECK_THROWS_WITH(
      (FromVolumeFile{filename, subfile_name, 10.0}),
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
  test<domain::creators::time_dependent_options::names::Translation>();
  test<domain::creators::time_dependent_options::names::Expansion>();
  test<domain::creators::time_dependent_options::names::Rotation>();
  test<domain::creators::time_dependent_options::names::ShapeSize<
      domain::ObjectLabel::B>>();
  test_errors();
}
}  // namespace
