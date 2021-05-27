// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {
class KastaunEtAl;
class NewmanHamlin;
class PalenzuelaEtAl;
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes

namespace {

template <typename OrderedListOfPrimitiveRecoverySchemes,
          size_t ThermodynamicDim>
void test_primitive_from_conservative_random(
    const gsl::not_null<std::mt19937*> generator,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const DataVector& used_for_size) noexcept {
  // generate random primitives with interesting astrophysical values
  const auto expected_rest_mass_density =
      TestHelpers::hydro::random_density(generator, used_for_size);
  const auto expected_lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size);
  const auto expected_spatial_velocity = TestHelpers::hydro::random_velocity(
      generator, expected_lorentz_factor, spatial_metric);
  Scalar<DataVector> expected_specific_internal_energy{};
  Scalar<DataVector> expected_pressure{};
  if constexpr (ThermodynamicDim == 1) {
    expected_specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density(
            expected_rest_mass_density);
    expected_pressure =
        equation_of_state.pressure_from_density(expected_rest_mass_density);
  } else if constexpr (ThermodynamicDim == 2) {
    // note this call assumes an ideal fluid
    expected_specific_internal_energy =
        TestHelpers::hydro::random_specific_internal_energy(generator,
                                                            used_for_size);
    expected_pressure = equation_of_state.pressure_from_density_and_energy(
        expected_rest_mass_density, expected_specific_internal_energy);
  }

  const auto expected_specific_enthalpy = hydro::relativistic_specific_enthalpy(
      expected_rest_mass_density, expected_specific_internal_energy,
      expected_pressure);
  const auto expected_magnetic_field =
      TestHelpers::hydro::random_magnetic_field(generator, expected_pressure,
                                                spatial_metric);
  const auto expected_divergence_cleaning_field =
      TestHelpers::hydro::random_divergence_cleaning_field(generator,
                                                           used_for_size);

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
  grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>::apply(
          make_not_null(&rest_mass_density),
          make_not_null(&specific_internal_energy),
          make_not_null(&spatial_velocity), make_not_null(&magnetic_field),
          make_not_null(&divergence_cleaning_field),
          make_not_null(&lorentz_factor), make_not_null(&pressure),
          make_not_null(&specific_enthalpy), tilde_d, tilde_tau, tilde_s,
          tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
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
  grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>::apply(
          make_not_null(&rest_mass_density),
          make_not_null(&specific_internal_energy),
          make_not_null(&spatial_velocity), make_not_null(&magnetic_field),
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
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>(dv);
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
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
      1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
      2>(&generator, ideal_fluid, dv);
}
