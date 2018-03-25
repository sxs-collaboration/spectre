// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {
template <typename DataType>
Scalar<DataType> expected_tilde_d(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.);
  get(result) = get(lorentz_factor) * get(rest_mass_density) *
                get(sqrt_det_spatial_metric);
  return result;
}

template <typename DataType>
Scalar<DataType> expected_tilde_tau(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& spatial_velocity_squared,
    const Scalar<DataType>& lorentz_factor, const Scalar<DataType>& pressure,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.);
  get(result) = (get(pressure) * get(spatial_velocity_squared) +
                 (get(lorentz_factor) / (1.0 + get(lorentz_factor)) *
                      get(spatial_velocity_squared) +
                  get(specific_internal_energy)) *
                     get(rest_mass_density)) *
                square(get(lorentz_factor)) * get(sqrt_det_spatial_metric);
  return result;
}

template <typename DataType, size_t Dim>
tnsr::i<DataType, Dim> expected_tilde_s(
    const Scalar<DataType>& rest_mass_density,
    const tnsr::i<DataType, Dim, Frame::Inertial>& spatial_velocity_oneform,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& specific_enthalpy,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  auto result = make_with_value<tnsr::i<DataType, Dim>>(rest_mass_density, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = spatial_velocity_oneform.get(i) *
                    square(get(lorentz_factor)) * get(specific_enthalpy) *
                    get(rest_mass_density) * get(sqrt_det_spatial_metric);
  }
  return result;
}

template <size_t Dim, typename DataType>
void test_conservative_from_primitive(
    const gsl::not_null<std::mt19937*> generator,
    const gsl::not_null<std::uniform_real_distribution<>*> distribution,
    const DataType& used_for_size) noexcept {
  const auto rest_mass_density = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataType>>(generator, distribution,
                                                used_for_size);
  const auto spatial_velocity_squared =
      make_with_random_values<Scalar<DataType>>(generator, distribution,
                                                used_for_size);
  const auto lorentz_factor = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto specific_enthalpy = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto pressure = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto sqrt_det_spatial_metric =
      make_with_random_values<Scalar<DataType>>(generator, distribution,
                                                used_for_size);
  const auto spatial_velocity_oneform =
      make_with_random_values<tnsr::i<DataType, Dim>>(generator, distribution,
                                                      used_for_size);

  auto tilde_d = make_with_value<Scalar<DataType>>(used_for_size, 0.);
  auto tilde_tau = make_with_value<Scalar<DataType>>(used_for_size, 0.);
  auto tilde_s = make_with_value<tnsr::i<DataType, Dim>>(used_for_size, 0.);

  RelativisticEuler::Valencia::conservative_from_primitive(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), rest_mass_density, specific_internal_energy,
      spatial_velocity_oneform, spatial_velocity_squared, lorentz_factor,
      specific_enthalpy, pressure, sqrt_det_spatial_metric);
  CHECK_ITERABLE_APPROX(expected_tilde_d(rest_mass_density, lorentz_factor,
                                         sqrt_det_spatial_metric),
                        tilde_d);
  CHECK_ITERABLE_APPROX(
      expected_tilde_tau(rest_mass_density, specific_internal_energy,
                         spatial_velocity_squared, lorentz_factor, pressure,
                         sqrt_det_spatial_metric),
      tilde_tau);
  CHECK_ITERABLE_APPROX(
      expected_tilde_s(rest_mass_density, spatial_velocity_oneform,
                       lorentz_factor, specific_enthalpy,
                       sqrt_det_spatial_metric),
      tilde_s);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.ConservativeFromPrimitive",
                  "[Unit][RelativisticEuler]") {
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed = " << seed);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const double d = std::numeric_limits<double>::signaling_NaN();
  test_conservative_from_primitive<1>(nn_generator, nn_distribution, d);
  test_conservative_from_primitive<2>(nn_generator, nn_distribution, d);
  test_conservative_from_primitive<3>(nn_generator, nn_distribution, d);

  const DataVector dv(5);
  test_conservative_from_primitive<1>(nn_generator, nn_distribution, dv);
  test_conservative_from_primitive<2>(nn_generator, nn_distribution, dv);
  test_conservative_from_primitive<3>(nn_generator, nn_distribution, dv);
}
