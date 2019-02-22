// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"
#include "tests/Utilities/RandomUnitNormal.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace grmhd {
namespace ValenciaDivClean {
namespace PrimitiveRecoverySchemes {
class NewmanHamlin;
class PalenzuelaEtAl;
}  // namespace PrimitiveRecoverySchemes
}  // namespace ValenciaDivClean
}  // namespace grmhd

namespace {

Scalar<DataVector> random_density(const gsl::not_null<std::mt19937*> generator,
                                  const DataVector& used_for_size) noexcept {
  // 1 g/cm^3 = 1.62e-18 in geometric units
  constexpr double minimum_density = 1.0e-5;
  constexpr double maximum_density = 1.0e-3;
  std::uniform_real_distribution<> distribution(log(minimum_density),
                                                log(maximum_density));
  return Scalar<DataVector>{exp(make_with_random_values<DataVector>(
      generator, make_not_null(&distribution), used_for_size))};
}

Scalar<DataVector> random_lorentz_factor(
    const gsl::not_null<std::mt19937*> generator,
    const DataVector& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-10.0, 3.0);
  return Scalar<DataVector>{
      1.0 + exp(make_with_random_values<DataVector>(
                generator, make_not_null(&distribution), used_for_size))};
}

tnsr::I<DataVector, 3> random_velocity(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::ii<DataVector, 3>& spatial_metric) noexcept {
  tnsr::I<DataVector, 3> spatial_velocity =
      random_unit_normal(generator, spatial_metric);
  const DataVector v = sqrt(1.0 - 1.0 / square(get(lorentz_factor)));
  get<0>(spatial_velocity) *= v;
  get<1>(spatial_velocity) *= v;
  get<2>(spatial_velocity) *= v;
  return spatial_velocity;
}

tnsr::I<DataVector, 3> random_magnetic_field(
    const gsl::not_null<std::mt19937*> generator,
    const Scalar<DataVector>& pressure,
    const tnsr::ii<DataVector, 3>& spatial_metric) noexcept {
  tnsr::I<DataVector, 3> magnetic_field =
      random_unit_normal(generator, spatial_metric);
  std::uniform_real_distribution<> distribution(-8.0, 14.0);
  const size_t number_of_points = get(pressure).size();
  for (size_t s = 0; s < number_of_points; ++s) {
    // magnitude of B set to vary ratio of magnetic pressure to fluid pressure
    const double B = sqrt(get(pressure)[s] * exp(distribution(*generator)));
    get<0>(magnetic_field)[s] *= B;
    get<1>(magnetic_field)[s] *= B;
    get<2>(magnetic_field)[s] *= B;
  }
  return magnetic_field;
}

// temperature in MeV
Scalar<DataVector> random_temperature(
    const gsl::not_null<std::mt19937*> generator,
    const DataVector& used_for_size) noexcept {
  constexpr double minimum_temperature = 1.0;
  constexpr double maximum_temperature = 50.0;
  std::uniform_real_distribution<> distribution(log(minimum_temperature),
                                                log(maximum_temperature));
  return Scalar<DataVector>{exp(make_with_random_values<DataVector>(
      generator, make_not_null(&distribution), used_for_size))};
}

// assumes gamma = 4/3 ideal fluid
Scalar<DataVector> random_specific_internal_energy(
    const gsl::not_null<std::mt19937*> generator,
    const DataVector& used_for_size) noexcept {
  // assumes Ideal gas with gamma = 4/3
  // For ideal fluid T = (m/k_b)(gamma - 1) epsilon
  // where m = atomic mass unit, k_b = Boltzmann constant
  return Scalar<DataVector>{3.21e-3 *
                            get(random_temperature(generator, used_for_size))};
}

Scalar<DataVector> random_divergence_cleaning_field(
    const gsl::not_null<std::mt19937*> generator,
    const DataVector& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-10.0, 10.0);
  return make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
}

tnsr::ii<DataVector, 3> random_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataVector& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto spatial_metric = make_with_random_values<tnsr::ii<DataVector, 3>>(
      generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < 3; ++d) {
    spatial_metric.get(d, d) += 1.0;
  }
  return spatial_metric;
}

Scalar<DataVector> specific_enthalpy(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& pressure) noexcept {
  return Scalar<DataVector>{1.0 + get(specific_internal_energy) +
                            get(pressure) / get(rest_mass_density)};
}

template <typename OrderedListOfPrimitiveRecoverySchemes,
          size_t ThermodynamicDim>
void test_primitive_from_conservative_random(
    const gsl::not_null<std::mt19937*> generator,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const DataVector& used_for_size) noexcept {
  // generate random primitives with interesting astrophysical values
  const auto expected_rest_mass_density =
      random_density(generator, used_for_size);
  const auto expected_lorentz_factor =
      random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric = random_spatial_metric(generator, used_for_size);
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
  const auto expected_magnetic_field =
      random_magnetic_field(generator, expected_pressure, spatial_metric);
  const auto expected_divergence_cleaning_field =
      random_divergence_cleaning_field(generator, used_for_size);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  const size_t number_of_points = used_for_size.size();
  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), make_not_null(&tilde_b),
      make_not_null(&tilde_phi), expected_rest_mass_density,
      expected_specific_internal_energy, expected_specific_enthalpy,
      expected_pressure, expected_spatial_velocity, expected_lorentz_factor,
      expected_magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      expected_divergence_cleaning_field);

  Scalar<DataVector> rest_mass_density(number_of_points);
  Scalar<DataVector> specific_internal_energy(number_of_points);
  tnsr::I<DataVector, 3> spatial_velocity(number_of_points);
  tnsr::I<DataVector, 3> magnetic_field(number_of_points);
  Scalar<DataVector> divergence_cleaning_field(number_of_points);
  Scalar<DataVector> lorentz_factor(number_of_points);
  // need to zero-initialize pressure because the recovery schemes assume it is
  // not nan
  Scalar<DataVector> pressure(number_of_points, 0.0);
  Scalar<DataVector> specific_enthalpy(number_of_points);
  grmhd::ValenciaDivClean::PrimitiveFromConservative<
      OrderedListOfPrimitiveRecoverySchemes,
      ThermodynamicDim>::apply(make_not_null(&rest_mass_density),
                               make_not_null(&specific_internal_energy),
                               make_not_null(&spatial_velocity),
                               make_not_null(&magnetic_field),
                               make_not_null(&divergence_cleaning_field),
                               make_not_null(&lorentz_factor),
                               make_not_null(&pressure),
                               make_not_null(&specific_enthalpy), tilde_d,
                               tilde_tau, tilde_s, tilde_b, tilde_phi,
                               spatial_metric, inv_spatial_metric,
                               sqrt_det_spatial_metric, equation_of_state);

  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e8);
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
  CHECK_ITERABLE_CUSTOM_APPROX(expected_magnetic_field, magnetic_field,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_divergence_cleaning_field,
                               divergence_cleaning_field, larger_approx);
}

template <typename OrderedListOfPrimitiveRecoverySchemes>
void test_primitive_from_conservative_known(
    const DataVector& used_for_size) noexcept {
  const auto expected_rest_mass_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 2.0);
  const auto expected_lorentz_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.25);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3>>(used_for_size, 0.0);
  get<0, 0>(spatial_metric) = 1.0;
  get<1, 1>(spatial_metric) = 4.0;
  get<2, 2>(spatial_metric) = 16.0;
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  get<0>(expected_spatial_velocity) = 9.0 / 65.0;
  get<1>(expected_spatial_velocity) = 6.0 / 65.0;
  get<2>(expected_spatial_velocity) = 9.0 / 65.0;
  const auto expected_specific_internal_energy =
      make_with_value<Scalar<DataVector>>(used_for_size, 3.0);
  const auto expected_pressure =
      make_with_value<Scalar<DataVector>>(used_for_size, 2.0);
  const auto expected_specific_enthalpy =
      make_with_value<Scalar<DataVector>>(used_for_size, 5.0);
  auto expected_magnetic_field =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  get<0>(expected_magnetic_field) = 36.0 / 13.0;
  get<1>(expected_magnetic_field) = 9.0 / 26.0;
  get<2>(expected_magnetic_field) = 3.0 / 13.0;
  const auto expected_divergence_cleaning_field =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.5);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  const size_t number_of_points = used_for_size.size();
  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), make_not_null(&tilde_b),
      make_not_null(&tilde_phi), expected_rest_mass_density,
      expected_specific_internal_energy, expected_specific_enthalpy,
      expected_pressure, expected_spatial_velocity, expected_lorentz_factor,
      expected_magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      expected_divergence_cleaning_field);

  Scalar<DataVector> rest_mass_density(number_of_points);
  Scalar<DataVector> specific_internal_energy(number_of_points);
  tnsr::I<DataVector, 3> spatial_velocity(number_of_points);
  tnsr::I<DataVector, 3> magnetic_field(number_of_points);
  Scalar<DataVector> divergence_cleaning_field(number_of_points);
  Scalar<DataVector> lorentz_factor(number_of_points);
  // need to zero-initialize pressure because the recovery schemes assume it is
  // not nan
  Scalar<DataVector> pressure(number_of_points, 0.0);
  Scalar<DataVector> specific_enthalpy(number_of_points);
  EquationsOfState::IdealFluid<true> ideal_fluid(4.0 / 3.0);
  grmhd::ValenciaDivClean::PrimitiveFromConservative<
      OrderedListOfPrimitiveRecoverySchemes,
      2>::apply(make_not_null(&rest_mass_density),
                make_not_null(&specific_internal_energy),
                make_not_null(&spatial_velocity),
                make_not_null(&magnetic_field),
                make_not_null(&divergence_cleaning_field),
                make_not_null(&lorentz_factor), make_not_null(&pressure),
                make_not_null(&specific_enthalpy), tilde_d, tilde_tau, tilde_s,
                tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
                sqrt_det_spatial_metric, ideal_fluid);

  CHECK_ITERABLE_APPROX(expected_rest_mass_density, rest_mass_density);
  CHECK_ITERABLE_APPROX(expected_specific_internal_energy,
                        specific_internal_energy);
  CHECK_ITERABLE_APPROX(expected_lorentz_factor, lorentz_factor);
  CHECK_ITERABLE_APPROX(expected_specific_enthalpy, specific_enthalpy);
  CHECK_ITERABLE_APPROX(expected_pressure, pressure);
  CHECK_ITERABLE_APPROX(expected_spatial_velocity, spatial_velocity);
  CHECK_ITERABLE_APPROX(expected_magnetic_field, magnetic_field);
  CHECK_ITERABLE_APPROX(expected_divergence_cleaning_field,
                        divergence_cleaning_field);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.PrimitiveFromConservative",
                  "[Unit][GrMhd]") {
  MAKE_GENERATOR(generator);

  EquationsOfState::PolytropicFluid<true> polytropic_fluid(100.0, 2.0);
  EquationsOfState::IdealFluid<true> ideal_fluid(4.0 / 3.0);
  const DataVector dv(5);
  test_primitive_from_conservative_known<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>(dv);
  test_primitive_from_conservative_known<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
      1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
      2>(&generator, ideal_fluid, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
      1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
      2>(&generator, ideal_fluid, dv);
}
