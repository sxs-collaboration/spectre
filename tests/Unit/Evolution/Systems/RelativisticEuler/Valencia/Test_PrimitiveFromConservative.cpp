// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

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
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace {

template <size_t Dim, size_t ThermodynamicDim>
void test_primitive_from_conservative(
    const gsl::not_null<std::mt19937*> generator,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const DataVector& used_for_size) {
  const auto expected_rest_mass_density =
      TestHelpers::hydro::random_density(generator, used_for_size);
  const auto expected_lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(generator, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size);
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

  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inv.first))};

  auto tilde_d = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto tilde_tau = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto tilde_s = make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.0);

  RelativisticEuler::Valencia::ConservativeFromPrimitive<Dim>::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), expected_rest_mass_density,
      expected_specific_internal_energy, expected_specific_enthalpy,
      expected_pressure, expected_spatial_velocity, expected_lorentz_factor,
      sqrt_det_spatial_metric, spatial_metric);

  auto rest_mass_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto specific_internal_energy =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto lorentz_factor = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto specific_enthalpy =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto pressure = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.0);

  RelativisticEuler::Valencia::PrimitiveFromConservative<Dim>::apply(
      make_not_null(&rest_mass_density),
      make_not_null(&specific_internal_energy), make_not_null(&lorentz_factor),
      make_not_null(&specific_enthalpy), make_not_null(&pressure),
      make_not_null(&spatial_velocity), tilde_d, tilde_tau, tilde_s,
      inv_spatial_metric, sqrt_det_spatial_metric, equation_of_state);

  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e7);
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
  const DataVector dv(5);
  test_primitive_from_conservative<1, 1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative<2, 1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative<3, 1>(&generator, polytropic_fluid, dv);
  test_primitive_from_conservative<1, 2>(&generator, ideal_fluid, dv);
  test_primitive_from_conservative<2, 2>(&generator, ideal_fluid, dv);
  test_primitive_from_conservative<3, 2>(&generator, ideal_fluid, dv);
}
