// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
template <size_t Dim>
std::string make_array_str(const double value) {
  std::stringstream ss{};
  ss << "[" << value;
  if constexpr (Dim > 1) {
    ss << ", " << value;
    if constexpr (Dim > 2) {
      ss << ", " << value;
    }
  }
  ss << "]";

  return ss.str();
}

void test_rotation_map_options() {
  {
    const auto rotation_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::RotationMapOptions<2>>(
        "InitialQuaternions: [[1.0, 0.0, 0.0, 0.0]]\n"
        "InitialAngles: Auto\n"
        "DecayTimescale: Auto\n");
    CHECK(rotation_map_options.name() == "RotationMap");
    std::array<DataVector, 3> expected_quat_func =
        make_array<3>(DataVector{4, 0.0});
    expected_quat_func[0][0] = 1.0;
    const std::array<DataVector, 3> expected_angle_func =
        make_array<3>(DataVector{3, 0.0});
    CHECK(expected_quat_func == rotation_map_options.quaternions);
    CHECK(expected_angle_func == rotation_map_options.angles);
    CHECK_FALSE(rotation_map_options.decay_timescale.has_value());
  }
  {
    const auto rotation_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::RotationMapOptions<3>>(
        "InitialQuaternions: [[1.0, 0.0, 0.0, 0.0]]\n"
        "InitialAngles: [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]\n"
        "DecayTimescale: 50.0\n");
    CHECK(rotation_map_options.name() == "RotationMap");
    std::array<DataVector, 4> expected_quat_func =
        make_array<4>(DataVector{4, 0.0});
    expected_quat_func[0][0] = 1.0;
    const std::array<DataVector, 4> expected_angle_func{
        DataVector{0.1, 0.2, 0.3}, DataVector{1.1, 1.2, 1.3},
        DataVector{3, 0.0}, DataVector{3, 0.0}};
    CHECK(expected_quat_func == rotation_map_options.quaternions);
    CHECK(expected_angle_func == rotation_map_options.angles);
    CHECK(rotation_map_options.decay_timescale.has_value());
    CHECK(rotation_map_options.decay_timescale.value() == 50.0);
  }

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Rotation"] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          0.0, std::array{DataVector{1.0, 0.0, 0.0, 0.0}},
          std::array{DataVector{3, 0.1}, DataVector{3, 0.2}, DataVector{3, 0.3},
                     DataVector{3, 0.4}},
          100.0);
  const std::string filename{"GoatCheese.h5"};
  const std::string subfile_name{"VolumeData"};
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  {
    h5::H5File<h5::AccessType::ReadWrite> h5_file{filename};
    auto& vol_file = h5_file.insert<h5::VolumeData>(subfile_name);

    // We don't care about the volume data here, just the functions of time
    vol_file.write_volume_data(
        0, 0.0,
        {ElementVolumeData{
            "blah",
            {TensorComponent{"RandomTensor", DataVector{3, 0.0}}},
            {3},
            {Spectral::Basis::Legendre},
            {Spectral::Quadrature::GaussLobatto}}},
        std::nullopt, serialize(functions_of_time));
  }

  {
    const auto rotation_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::RotationMapOptions<2>>(
        "InitialQuaternions:\n"
        "  H5Filename: " +
        filename + "\n  SubfileName: " + subfile_name +
        "\n  Time: 0.0\n"
        "InitialAngles: Auto\n"
        "DecayTimescale: Auto\n");
    CHECK(rotation_map_options.name() == "RotationMap");
    // q
    // dtq = 0.5 * q * omega
    // d2tq = 0.5 * (dtq * omega + q * dtomega)
    std::array<DataVector, 3> expected_quaternion{
        DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{0.0, 0.1, 0.1, 0.1},
        DataVector{-0.03, 0.15, 0.15, 0.15}};
    std::array<DataVector, 3> expected_angle{
        DataVector{3, 0.1}, DataVector{3, 0.2}, DataVector{3, 0.3}};

    CHECK_ITERABLE_APPROX(expected_quaternion,
                          rotation_map_options.quaternions);
    CHECK(expected_angle == rotation_map_options.angles);
    CHECK_FALSE(rotation_map_options.decay_timescale.has_value());
  }
  {
    const auto rotation_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::RotationMapOptions<3>>(
        "InitialQuaternions:\n"
        "  H5Filename: " +
        filename + "\n  SubfileName: " + subfile_name +
        "\n  Time: 0.0\n"
        "InitialAngles: [[0.11, 0.22, 0.33]]\n"
        "DecayTimescale: Auto\n");
    CHECK(rotation_map_options.name() == "RotationMap");
    // q
    // dtq = 0.5 * q * omega
    // d2tq = 0.5 * (dtq * omega + q * dtomega)
    std::array<DataVector, 4> expected_quaternion{
        DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{0.0, 0.1, 0.1, 0.1},
        DataVector{-0.03, 0.15, 0.15, 0.15}, DataVector{4, 0.0}};
    std::array<DataVector, 4> expected_angle{
        DataVector{0.11, 0.22, 0.33}, DataVector{3, 0.0}, DataVector{3, 0.0},
        DataVector{3, 0.0}};

    CHECK_ITERABLE_APPROX(expected_quaternion,
                          rotation_map_options.quaternions);
    CHECK(expected_angle == rotation_map_options.angles);
    CHECK_FALSE(rotation_map_options.decay_timescale.has_value());
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.RotationMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_rotation_map_options();
}
