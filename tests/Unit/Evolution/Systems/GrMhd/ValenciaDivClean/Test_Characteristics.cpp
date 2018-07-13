// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

void test_characteristic_speeds(const DataVector& used_for_size) noexcept {
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  const double max_value = 1.0 / sqrt(3);
  pypp::check_with_random_values<7>(
      &grmhd::ValenciaDivClean::characteristic_speeds, "TestFunctions",
      "characteristic_speeds",
      {{{0.0, 1.0},
        {-1.0, 1.0},
        {-max_value, max_value},
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
        {-max_value, max_value}}},
      used_for_size);
}

void test_with_normal_along_coordinate_axes(
    const DataVector& used_for_size) noexcept {
  const auto seed = std::random_device{}();
  CAPTURE(seed);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  const auto spatial_velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  const auto spatial_velocity_squared =
      make_with_random_values<Scalar<DataVector>>(nn_generator, nn_distribution,
                                                  used_for_size);
  const auto sound_speed_squared = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto alfven_speed_squared = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto normal = euclidean_basis_vector(direction, used_for_size);

    CHECK_ITERABLE_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds(
            lapse, shift, spatial_velocity, spatial_velocity_squared,
            sound_speed_squared, alfven_speed_squared, normal),
        (pypp::call<std::array<DataVector, 9>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            alfven_speed_squared, normal)));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  const DataVector dv(5);
  test_characteristic_speeds(dv);
  // Test with aligned normals to check the code works
  // with vector components being 0.
  test_with_normal_along_coordinate_axes(dv);
}
