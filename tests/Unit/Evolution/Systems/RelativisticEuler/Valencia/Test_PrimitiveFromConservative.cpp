// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/PrimitiveFromConservative.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"
#include "tests/Utilities/RandomUnitNormal.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace {

template <typename DataType>
Scalar<DataType> random_density(const gsl::not_null<std::mt19937*> generator,
                                const DataType& used_for_size) noexcept {
  // 1 g/cm^3 = 1.62e-18 in geometric units
  constexpr double minimum_density = 1.0e-5;
  constexpr double maximum_density = 1.0e-3;
  std::uniform_real_distribution<> distribution(log(minimum_density),
                                                log(maximum_density));
  return Scalar<DataType>{exp(make_with_random_values<DataType>(
      generator, make_not_null(&distribution), used_for_size))};
}

template <typename DataType>
Scalar<DataType> random_lorentz_factor(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-10.0, 3.0);
  return Scalar<DataType>{
      1.0 + exp(make_with_random_values<DataType>(
                generator, make_not_null(&distribution), used_for_size))};
}

template <typename DataType, size_t Dim>
tnsr::I<DataType, Dim> random_velocity(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::ii<DataType, Dim>& spatial_metric) noexcept {
  tnsr::I<DataType, Dim> spatial_velocity =
      random_unit_normal(generator, spatial_metric);
  const DataType v = sqrt(1.0 - 1.0 / square(get(lorentz_factor)));
  for (size_t d = 0; d < Dim; ++d) {
    spatial_velocity.get(d) *= v;
  }
  return spatial_velocity;
}

// temperature in MeV
template <typename DataType>
Scalar<DataType> random_temperature(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  constexpr double minimum_temperature = 1.0;
  constexpr double maximum_temperature = 50.0;
  std::uniform_real_distribution<> distribution(log(minimum_temperature),
                                                log(maximum_temperature));
  return Scalar<DataType>{exp(make_with_random_values<DataType>(
      generator, make_not_null(&distribution), used_for_size))};
}

template <typename DataType>
Scalar<DataType> random_specific_internal_energy(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  // assumes Ideal gas with gamma = 4/3
  // For ideal fluid T = (m/k_b)(gamma - 1) epsilon
  // where m = atomic mass unit, k_b = Boltzmann constant
  return Scalar<DataType>{3.21e-3 *
                          get(random_temperature(generator, used_for_size))};
}

template <typename DataType>
Scalar<DataType> specific_enthalpy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept {
  return Scalar<DataType>{1.0 + get(specific_internal_energy) +
                          get(pressure) / get(rest_mass_density)};
}

template <typename DataType, size_t Dim>
tnsr::ii<DataType, Dim> random_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto spatial_metric = make_with_random_values<tnsr::ii<DataType, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < Dim; ++d) {
    spatial_metric.get(d, d) += 1.0;
  }
  return spatial_metric;
}

template <size_t Dim, size_t ThermodynamicDim, typename DataType>
void test_primitive_from_conservative(
    const gsl::not_null<std::mt19937*> generator,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const DataType& used_for_size) noexcept {
  const auto expected_rest_mass_density =
      random_density(generator, used_for_size);
  const auto expected_lorentz_factor =
      random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric =
      random_spatial_metric<DataType, Dim>(generator, used_for_size);
  const auto expected_spatial_velocity =
      random_velocity(generator, expected_lorentz_factor, spatial_metric);
  const auto expected_specific_internal_energy = make_overloader(
      [&expected_rest_mass_density](
          const EquationsOfState::EquationOfState<true, 1>&
              the_equation_of_state) noexcept {
        return the_equation_of_state.specific_internal_energy_from_density(
            expected_rest_mass_density);
      },
      [&generator,
       &used_for_size ](const EquationsOfState::EquationOfState<true, 2>&
                        /*the_equation_of_state*/) noexcept {
        // note this call assumes an ideal fluid
        return random_specific_internal_energy(generator, used_for_size);
      })(equation_of_state);
  const auto expected_pressure = make_overloader(
      [&expected_rest_mass_density](
          const EquationsOfState::EquationOfState<true, 1>&
              the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density(
            expected_rest_mass_density);
      },
      [&expected_rest_mass_density, &expected_specific_internal_energy ](
          const EquationsOfState::EquationOfState<true, 2>&
              the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density_and_energy(
            expected_rest_mass_density, expected_specific_internal_energy);
      })(equation_of_state);

  const auto expected_specific_enthalpy =
      specific_enthalpy(expected_rest_mass_density,
                        expected_specific_internal_energy, expected_pressure);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataType> sqrt_det_spatial_metric =
      Scalar<DataType>{sqrt(get(det_and_inv.first))};
  const auto spatial_velocity_oneform =
      raise_or_lower_index(expected_spatial_velocity, spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity_oneform, expected_spatial_velocity);

  auto tilde_d = make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto tilde_tau = make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto tilde_s = make_with_value<tnsr::i<DataType, Dim>>(used_for_size, 0.0);

  RelativisticEuler::Valencia::conservative_from_primitive(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), expected_rest_mass_density,
      expected_specific_internal_energy, spatial_velocity_oneform,
      spatial_velocity_squared, expected_lorentz_factor,
      expected_specific_enthalpy, expected_pressure, sqrt_det_spatial_metric);

  auto rest_mass_density =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto specific_internal_energy =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto lorentz_factor = make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto specific_enthalpy =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto pressure = make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  auto spatial_velocity =
      make_with_value<tnsr::I<DataType, Dim>>(used_for_size, 0.0);

  RelativisticEuler::Valencia::primitive_from_conservative<ThermodynamicDim>(
      make_not_null(&rest_mass_density),
      make_not_null(&specific_internal_energy), make_not_null(&lorentz_factor),
      make_not_null(&specific_enthalpy), make_not_null(&pressure),
      make_not_null(&spatial_velocity), tilde_d, tilde_tau, tilde_s,
      inv_spatial_metric, sqrt_det_spatial_metric, equation_of_state);

  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e6);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_rest_mass_density, rest_mass_density,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_specific_internal_energy,
                               specific_internal_energy, larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_lorentz_factor, lorentz_factor,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_specific_enthalpy, specific_enthalpy,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_pressure, pressure, larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_spatial_velocity, spatial_velocity,
                               larger_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.PrimitiveFromConservative",
                  "[Unit][RelativisticEuler]") {
  MAKE_GENERATOR(generator);

  EquationsOfState::PolytropicFluid<true> polytropic_fluid(100.0, 2.0);
  EquationsOfState::IdealFluid<true> ideal_fluid(4.0 / 3.0);
  const double d = std::numeric_limits<double>::signaling_NaN();
  test_primitive_from_conservative<1, 1>(&generator, polytropic_fluid, d);
  test_primitive_from_conservative<2, 1>(&generator, polytropic_fluid, d);
  test_primitive_from_conservative<3, 1>(&generator, polytropic_fluid, d);
  test_primitive_from_conservative<1, 2>(&generator, ideal_fluid, d);
  test_primitive_from_conservative<2, 2>(&generator, ideal_fluid, d);
  test_primitive_from_conservative<3, 2>(&generator, ideal_fluid, d);

  const DataVector dv(5);
  test_primitive_from_conservative<1, 1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative<2, 1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative<3, 1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative<1, 2>(&generator, ideal_fluid, dv);
  test_primitive_from_conservative<2, 2>(&generator, ideal_fluid, dv);
  test_primitive_from_conservative<3, 2>(&generator, ideal_fluid, dv);
}
