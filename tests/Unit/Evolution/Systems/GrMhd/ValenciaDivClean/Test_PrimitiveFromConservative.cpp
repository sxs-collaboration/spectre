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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAlHydro.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
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

// In certain cases the evolved variables may go out of
// bounds (for EoS evaluation) check that primitive
// recovery correctly fixes these (currently only
// used for electron fraction)
template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool UseMagneticField = false>
void test_potentially_eos_dependent_primitive_corrections(
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const Scalar<DataVector>& input_rest_mass_density,
    const Scalar<DataVector>& input_temperature,
    const Scalar<DataVector>& input_electron_fraction,
    const gsl::not_null<std::mt19937*> generator) {
  const DataVector& used_for_size = get(input_rest_mass_density);
  // Taken as references for now, but make copies if in the future
  // they are corrected as well.
  const auto& expected_rest_mass_density = input_rest_mass_density;
  const auto& expected_temperature = input_temperature;
  // The electron fraction may evolve out of bounds
  const auto expected_electron_fraction =
      Scalar<DataVector>(min(0.5, max(get(input_electron_fraction), 0.)));
  const auto expected_lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size);
  const auto expected_spatial_velocity = TestHelpers::hydro::random_velocity(
      generator, expected_lorentz_factor, spatial_metric);
  Scalar<DataVector> expected_specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density_and_temperature(
          expected_rest_mass_density, expected_temperature,
          expected_electron_fraction);

  Scalar<DataVector> expected_pressure =
      equation_of_state.pressure_from_density_and_temperature(
          expected_rest_mass_density, expected_temperature,
          expected_electron_fraction);

  auto expected_magnetic_field = TestHelpers::hydro::random_magnetic_field(
      generator, expected_pressure, spatial_metric);
  if constexpr (!UseMagneticField) {
    get<0>(expected_magnetic_field) = 0.;
    get<1>(expected_magnetic_field) = 0.;
    get<2>(expected_magnetic_field) = 0.;
  }

  const auto expected_divergence_cleaning_field =
      TestHelpers::hydro::random_divergence_cleaning_field(generator,
                                                           used_for_size);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  const size_t number_of_points = used_for_size.size();
  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_ye(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_ye),
      make_not_null(&tilde_tau), make_not_null(&tilde_s),
      make_not_null(&tilde_b), make_not_null(&tilde_phi),
      expected_rest_mass_density, expected_electron_fraction,
      expected_specific_internal_energy, expected_pressure,
      expected_spatial_velocity, expected_lorentz_factor,
      expected_magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      expected_divergence_cleaning_field);

  const double cutoff_d_for_inversion = 0.0;
  const double density_when_skipping_inversion = 0.0;
  const double kastaun_max_lorentz = std::numeric_limits<double>::max();
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      primitive_from_conservative_options(cutoff_d_for_inversion,
                                          density_when_skipping_inversion,
                                          kastaun_max_lorentz);

  Scalar<DataVector> rest_mass_density(number_of_points);
  Scalar<DataVector> electron_fraction(number_of_points);
  Scalar<DataVector> specific_internal_energy(number_of_points);
  tnsr::I<DataVector, 3> spatial_velocity(number_of_points);
  tnsr::I<DataVector, 3> magnetic_field(number_of_points);
  Scalar<DataVector> divergence_cleaning_field(number_of_points);
  Scalar<DataVector> lorentz_factor(number_of_points);
  // need to zero-initialize pressure because the recovery schemes assume it is
  // not nan
  Scalar<DataVector> pressure(number_of_points, 0.0);
  Scalar<DataVector> temperature(number_of_points);
  grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>::apply(
          make_not_null(&rest_mass_density), make_not_null(&electron_fraction),
          make_not_null(&specific_internal_energy),
          make_not_null(&spatial_velocity), make_not_null(&magnetic_field),
          make_not_null(&divergence_cleaning_field),
          make_not_null(&lorentz_factor), make_not_null(&pressure),
          make_not_null(&temperature), tilde_d, tilde_ye, tilde_tau, tilde_s,
          tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
          sqrt_det_spatial_metric, equation_of_state,
          primitive_from_conservative_options);

  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e8);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_rest_mass_density, rest_mass_density,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_electron_fraction, electron_fraction,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_temperature, temperature,
                               larger_approx);
}

template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool UseMagneticField = true, typename EosType>
void test_primitive_from_conservative_random(
    const gsl::not_null<std::mt19937*> generator,
    const EosType& equation_of_state, const DataVector& used_for_size) {
  static_assert(EosType::thermodynamic_dim == 3);
  static_assert(EosType::is_relativistic);
  constexpr bool eos_is_barotropic =
      tt::is_a_v<EquationsOfState::Barotropic3D, EosType>;
  // generate random primitives with interesting astrophysical values
  const auto expected_rest_mass_density =
      TestHelpers::hydro::random_density(generator, used_for_size);
  const auto expected_electron_fraction =
      TestHelpers::hydro::random_electron_fraction(generator, used_for_size);
  const auto expected_lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(generator, used_for_size);
  const auto expected_spatial_velocity = TestHelpers::hydro::random_velocity(
      generator, expected_lorentz_factor, spatial_metric);
  Scalar<DataVector> expected_specific_internal_energy{};
  Scalar<DataVector> expected_temperature{};
  Scalar<DataVector> expected_pressure{};
  if constexpr (eos_is_barotropic) {
    expected_specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density_and_temperature(
            expected_rest_mass_density, Scalar<DataVector>{},
            Scalar<DataVector>{});
  } else {
    expected_specific_internal_energy =
        TestHelpers::hydro::random_specific_internal_energy(generator,
                                                            used_for_size);
  }
  expected_temperature = equation_of_state.temperature_from_density_and_energy(
      expected_rest_mass_density, expected_specific_internal_energy,
      expected_electron_fraction);
  expected_pressure = equation_of_state.pressure_from_density_and_temperature(
      expected_rest_mass_density, expected_temperature,
      expected_electron_fraction);

  auto expected_magnetic_field = TestHelpers::hydro::random_magnetic_field(
      generator, expected_pressure, spatial_metric);
  if constexpr (!UseMagneticField) {
    get<0>(expected_magnetic_field) = 0.;
    get<1>(expected_magnetic_field) = 0.;
    get<2>(expected_magnetic_field) = 0.;
  }

  const auto expected_divergence_cleaning_field =
      TestHelpers::hydro::random_divergence_cleaning_field(generator,
                                                           used_for_size);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  const size_t number_of_points = used_for_size.size();
  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_ye(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_ye),
      make_not_null(&tilde_tau), make_not_null(&tilde_s),
      make_not_null(&tilde_b), make_not_null(&tilde_phi),
      expected_rest_mass_density, expected_electron_fraction,
      expected_specific_internal_energy, expected_pressure,
      expected_spatial_velocity, expected_lorentz_factor,
      expected_magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      expected_divergence_cleaning_field);

  const double cutoff_d_for_inversion = 0.0;
  const double density_when_skipping_inversion = 0.0;
  const double kastaun_max_lorentz = std::numeric_limits<double>::max();
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      primitive_from_conservative_options(cutoff_d_for_inversion,
                                          density_when_skipping_inversion,
                                          kastaun_max_lorentz);
  Scalar<DataVector> rest_mass_density(number_of_points);
  Scalar<DataVector> electron_fraction(number_of_points);
  Scalar<DataVector> specific_internal_energy(number_of_points);
  tnsr::I<DataVector, 3> spatial_velocity(number_of_points);
  tnsr::I<DataVector, 3> magnetic_field(number_of_points);
  Scalar<DataVector> divergence_cleaning_field(number_of_points);
  Scalar<DataVector> lorentz_factor(number_of_points);
  // need to zero-initialize pressure because the recovery schemes assume it is
  // not nan
  Scalar<DataVector> pressure(number_of_points, 0.0);
  Scalar<DataVector> temperature(number_of_points);
  grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>::apply(
          make_not_null(&rest_mass_density), make_not_null(&electron_fraction),
          make_not_null(&specific_internal_energy),
          make_not_null(&spatial_velocity), make_not_null(&magnetic_field),
          make_not_null(&divergence_cleaning_field),
          make_not_null(&lorentz_factor), make_not_null(&pressure),
          make_not_null(&temperature), tilde_d, tilde_ye, tilde_tau, tilde_s,
          tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
          sqrt_det_spatial_metric, equation_of_state,
          primitive_from_conservative_options);
  INFO("Checking random-value primitive recovery.");
  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e8);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_rest_mass_density, rest_mass_density,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_electron_fraction, electron_fraction,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_specific_internal_energy,
                               specific_internal_energy, larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_lorentz_factor, lorentz_factor,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_pressure, pressure, larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_temperature, temperature,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_spatial_velocity, spatial_velocity,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_magnetic_field, magnetic_field,
                               larger_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_divergence_cleaning_field,
                               divergence_cleaning_field, larger_approx);
}

template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool UseMagneticField = true>
void test_primitive_from_conservative_known(const DataVector& used_for_size) {
  const auto expected_rest_mass_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 2.0);
  const auto expected_electron_fraction =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.1);
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
  const auto expected_temperature =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto expected_pressure =
      make_with_value<Scalar<DataVector>>(used_for_size, 2.0);
  auto expected_magnetic_field =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  get<0>(expected_magnetic_field) = 36.0 / 13.0;
  get<1>(expected_magnetic_field) = 9.0 / 26.0;
  get<2>(expected_magnetic_field) = 3.0 / 13.0;

  if constexpr (!UseMagneticField) {
    INFO("Not using Magnetic Field");
    get<0>(expected_magnetic_field) = 0.;
    get<1>(expected_magnetic_field) = 0.;
    get<2>(expected_magnetic_field) = 0.;
  }
  const auto expected_divergence_cleaning_field =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.5);

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  const size_t number_of_points = used_for_size.size();
  Scalar<DataVector> tilde_d(number_of_points);
  Scalar<DataVector> tilde_ye(number_of_points);
  Scalar<DataVector> tilde_tau(number_of_points);
  tnsr::i<DataVector, 3> tilde_s(number_of_points);
  tnsr::I<DataVector, 3> tilde_b(number_of_points);
  Scalar<DataVector> tilde_phi(number_of_points);

  const double cutoff_d_for_inversion = 0.0;
  const double density_when_skipping_inversion = 0.0;
  const double kastaun_max_lorentz = 1.0e4;
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      primitive_from_conservative_options(cutoff_d_for_inversion,
                                          density_when_skipping_inversion,
                                          kastaun_max_lorentz);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_ye),
      make_not_null(&tilde_tau), make_not_null(&tilde_s),
      make_not_null(&tilde_b), make_not_null(&tilde_phi),
      expected_rest_mass_density, expected_electron_fraction,
      expected_specific_internal_energy, expected_pressure,
      expected_spatial_velocity, expected_lorentz_factor,
      expected_magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      expected_divergence_cleaning_field);

  Scalar<DataVector> rest_mass_density(number_of_points);
  Scalar<DataVector> electron_fraction(number_of_points);
  Scalar<DataVector> specific_internal_energy(number_of_points);
  Scalar<DataVector> temperature(number_of_points);
  tnsr::I<DataVector, 3> spatial_velocity(number_of_points);
  tnsr::I<DataVector, 3> magnetic_field(number_of_points);
  Scalar<DataVector> divergence_cleaning_field(number_of_points);
  Scalar<DataVector> lorentz_factor(number_of_points);
  // need to zero-initialize pressure because the recovery schemes assume it is
  // not nan
  Scalar<DataVector> pressure(number_of_points, 0.0);
  EquationsOfState::Equilibrium3D ideal_fluid{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}};
  grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfPrimitiveRecoverySchemes>::apply(
          make_not_null(&rest_mass_density), make_not_null(&electron_fraction),
          make_not_null(&specific_internal_energy),
          make_not_null(&spatial_velocity), make_not_null(&magnetic_field),
          make_not_null(&divergence_cleaning_field),
          make_not_null(&lorentz_factor), make_not_null(&pressure),
          make_not_null(&temperature), tilde_d, tilde_ye, tilde_tau, tilde_s,
          tilde_b, tilde_phi, spatial_metric, inv_spatial_metric,
          sqrt_det_spatial_metric, ideal_fluid,
          primitive_from_conservative_options);
  INFO("test primitive from conservative with known values");
  CHECK_ITERABLE_APPROX(expected_rest_mass_density, rest_mass_density);
  CHECK_ITERABLE_APPROX(expected_electron_fraction, electron_fraction);
  CHECK_ITERABLE_APPROX(expected_specific_internal_energy,
                        specific_internal_energy);
  CHECK_ITERABLE_APPROX(expected_lorentz_factor, lorentz_factor);
  CHECK_ITERABLE_APPROX(expected_temperature, temperature);
  CHECK_ITERABLE_APPROX(expected_pressure, pressure);
  CHECK_ITERABLE_APPROX(expected_spatial_velocity, spatial_velocity);
  CHECK_ITERABLE_APPROX(expected_magnetic_field, magnetic_field);
  CHECK_ITERABLE_APPROX(expected_divergence_cleaning_field,
                        divergence_cleaning_field);

  if constexpr (not UseMagneticField) {
    // Test KastaunHydro for FPE safety
    tilde_tau = make_with_value<Scalar<DataVector>>(used_for_size, -10.);

    grmhd::ValenciaDivClean::PrimitiveFromConservative<
        OrderedListOfPrimitiveRecoverySchemes,
        false>::apply(make_not_null(&rest_mass_density),
                      make_not_null(&electron_fraction),
                      make_not_null(&specific_internal_energy),
                      make_not_null(&spatial_velocity),
                      make_not_null(&magnetic_field),
                      make_not_null(&divergence_cleaning_field),
                      make_not_null(&lorentz_factor), make_not_null(&pressure),
                      make_not_null(&temperature), tilde_d, tilde_ye, tilde_tau,
                      tilde_s, tilde_b, tilde_phi, spatial_metric,
                      inv_spatial_metric, sqrt_det_spatial_metric, ideal_fluid,
                      primitive_from_conservative_options);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.PrimitiveFromConservative",
                  "[Unit][GrMhd]") {
  MAKE_GENERATOR(generator);

  EquationsOfState::Barotropic3D wrapped_3d_polytrope(
      EquationsOfState::PolytropicFluid<true>(100.0, 2.0));
  EquationsOfState::Equilibrium3D wrapped_3d_polytrope_hot(
      EquationsOfState::HybridEos<EquationsOfState::PolytropicFluid<true>>(
          EquationsOfState::PolytropicFluid<true>(100.0, 2.0), 5.0 / 3.0));
  EquationsOfState::Equilibrium3D wrapped_ideal_fluid{
      EquationsOfState::IdealFluid<true>(4.0 / 3.0)};
  const DataVector dv(5);
  test_primitive_from_conservative_known<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>(dv);
  test_primitive_from_conservative_known<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>(dv);
  test_primitive_from_conservative_known<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>,
      false>(dv);
  test_primitive_from_conservative_known<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
      1>(&generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(
      &generator, wrapped_ideal_fluid, dv);
  test_primitive_from_conservative_random<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>(
      &generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>(
      &generator, wrapped_ideal_fluid, dv);
  test_primitive_from_conservative_random<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>(
      &generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>(
      &generator, wrapped_ideal_fluid, dv);

  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>,
      false>(&generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>,
      false>(&generator, wrapped_ideal_fluid, dv);
  INFO("3D EoS Kastaun");
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
      true>(&generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
      false>(&generator, wrapped_3d_polytrope, dv);
  INFO("3D EoS Kastaun Hydro");
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>,
      false>(&generator, wrapped_3d_polytrope, dv);
  INFO("3D EoS Newman-Hamlin");
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
      true>(&generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
      false>(&generator, wrapped_3d_polytrope, dv);
  INFO("3D EoS Palenzuela");
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
      true>(&generator, wrapped_3d_polytrope, dv);
  test_primitive_from_conservative_random<
      tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
      false>(&generator, wrapped_3d_polytrope, dv);

  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, -.1), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, 1.0), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, -.1), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAlHydro>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, 1.0), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, -.1), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, 1.0), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, -.1), &generator);
  test_potentially_eos_dependent_primitive_corrections<tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>>(
      wrapped_3d_polytrope_hot, make_with_value<Scalar<DataVector>>(dv, 1e-4),
      make_with_value<Scalar<DataVector>>(dv, 1e-1),
      make_with_value<Scalar<DataVector>>(dv, 1.0), &generator);
}
