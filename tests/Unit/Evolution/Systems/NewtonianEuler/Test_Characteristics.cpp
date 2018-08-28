// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
void test_characteristic_speeds(const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<3>(&NewtonianEuler::characteristic_speeds<Dim>,
                                    "TestFunctions", "characteristic_speeds",
                                    {{{-1.0, 1.0}, {0.0, 1.0}, {-1.0, 1.0}}},
                                    used_for_size);
}

template <size_t Dim>
void test_with_normal_along_coordinate_axes(
    const DataVector& used_for_size) noexcept {
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed = " << seed);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto sound_speed_squared = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);

  for (const auto& direction : domain::Direction<Dim>::all_directions()) {
    const auto normal = euclidean_basis_vector(direction, used_for_size);

    CHECK_ITERABLE_APPROX(NewtonianEuler::characteristic_speeds(
                              velocity, sound_speed_squared, normal),
                          (pypp::call<std::array<DataVector, Dim + 2>>(
                              "TestFunctions", "characteristic_speeds",
                              velocity, sound_speed_squared, normal)));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_characteristic_speeds, (1, 2, 3))
  CHECK_FOR_DATAVECTORS(test_with_normal_along_coordinate_axes, (1, 2, 3))
}
