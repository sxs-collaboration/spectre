// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"  // delete when py
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim> expected_velocity(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& momentum_density) noexcept {
  auto result = make_with_value<tnsr::I<DataType, Dim>>(mass_density, 0.0);
  const DataType one_over_mass_density = 1.0 / get(mass_density);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = momentum_density.get(i) * one_over_mass_density;
  }
  return result;
}

template <size_t Dim, typename DataType>
Scalar<DataType> expected_specific_internal_energy(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& momentum_density,
    const Scalar<DataType>& energy_density) noexcept {
  const auto velocity =
      NewtonianEuler::velocity(mass_density, momentum_density);
  return Scalar<DataType>{-0.5 * get(dot_product(velocity, velocity)) +
                          get(energy_density) / get(mass_density)};
}

template <size_t Dim, typename DataType>
void test_velocity(
    const gsl::not_null<std::mt19937*> generator,
    const gsl::not_null<std::uniform_real_distribution<>*> distribution,
    const DataType& used_for_size) noexcept {
  const auto mass_density = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto momentum_density = make_with_random_values<tnsr::I<DataType, Dim>>(
      generator, distribution, used_for_size);

  const auto velocity =
      NewtonianEuler::velocity(mass_density, momentum_density);
  const auto exp_velocity = expected_velocity(mass_density, momentum_density);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(exp_velocity.get(i), velocity.get(i));
  }
}

template <size_t Dim, typename DataType>
void test_primitive_from_conservative(
    const gsl::not_null<std::mt19937*> generator,
    const gsl::not_null<std::uniform_real_distribution<>*> distribution,
    const DataType& used_for_size) noexcept {
  const auto mass_density = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto momentum_density = make_with_random_values<tnsr::I<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto energy_density = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);

  auto velocity = make_with_value<tnsr::I<DataType, Dim>>(used_for_size, 0.0);
  auto specific_internal_energy =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  NewtonianEuler::primitive_from_conservative(
      make_not_null(&velocity), make_not_null(&specific_internal_energy),
      mass_density, momentum_density, energy_density);

  const auto exp_velocity = expected_velocity(mass_density, momentum_density);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(exp_velocity.get(i), velocity.get(i));
  }
  CHECK_ITERABLE_APPROX(expected_specific_internal_energy(
                            mass_density, momentum_density, energy_density),
                        specific_internal_energy);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.PrimitiveFromConservative",
    "[Unit][Evolution]") {
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed = " << seed);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const double d = std::numeric_limits<double>::signaling_NaN();
  test_velocity<1>(nn_generator, nn_distribution, d);
  test_velocity<2>(nn_generator, nn_distribution, d);
  test_velocity<3>(nn_generator, nn_distribution, d);
  test_primitive_from_conservative<1>(nn_generator, nn_distribution, d);
  test_primitive_from_conservative<2>(nn_generator, nn_distribution, d);
  test_primitive_from_conservative<3>(nn_generator, nn_distribution, d);

  const DataVector dv(5);
  test_velocity<1>(nn_generator, nn_distribution, dv);
  test_velocity<2>(nn_generator, nn_distribution, dv);
  test_velocity<3>(nn_generator, nn_distribution, dv);
  test_primitive_from_conservative<1>(nn_generator, nn_distribution, dv);
  test_primitive_from_conservative<2>(nn_generator, nn_distribution, dv);
  test_primitive_from_conservative<3>(nn_generator, nn_distribution, dv);
}
