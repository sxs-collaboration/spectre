// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

// parts of PiecewisePolytropicFluid
// choose high or low constants/exponents based on transition
// equate transition density, hi/lo constant/exponent
// pressure_from_density_impl

namespace {

// Calculate high density polytropic constant, based on low density constant
// and exponents.  This is derived from enforcing pressure continuity at
// transition density.
double calc_Khi(const double transition_density, const double poly_constant_lo,
                const double poly_exponent_lo, const double poly_exponent_hi) {
  return poly_constant_lo *
         pow(transition_density, poly_exponent_lo - poly_exponent_hi);
}

// Check solutions for datavector inputs
template <bool IsRelativistic>
void check_exact() {
  const double transition_density = 10.0;
  const double polytropic_constant_lo = 0.5;
  const double polytropic_exponent_lo = 1.5;
  const double polytropic_exponent_hi = 2.0;

  const Scalar<double> rest_mass_density_high{1.1 * transition_density};
  const Scalar<double> rest_mass_density_transition{transition_density};
  const Scalar<double> rest_mass_density_low{0.9 * transition_density};

  const auto eos = EquationsOfState::PiecewisePolytropicFluid<IsRelativistic>{
      transition_density, polytropic_constant_lo, polytropic_exponent_lo,
      polytropic_exponent_hi};

  // Khi = Klo * pow (rho_transition,gamma_lo - gamma_hi)
  const double polytropic_constant_hi =
      polytropic_constant_lo *
      pow(transition_density, polytropic_exponent_lo - polytropic_exponent_hi);

  // tests of DataVector type functions
  {
    const Scalar<DataVector> rho{DataVector{0.5 * transition_density,
                                            transition_density,
                                            2.0 * transition_density}};
    // pressure
    const Scalar<DataVector> p = eos.pressure_from_density(rho);
    const Scalar<DataVector> p_expected{
        DataVector{polytropic_constant_lo *
                       pow(0.5 * transition_density, polytropic_exponent_lo),
                   polytropic_constant_lo *
                       pow(transition_density, polytropic_exponent_lo),
                   polytropic_constant_hi *
                       pow(2.0 * transition_density, polytropic_exponent_hi)}};
    CHECK(p == p_expected);

    // eint
    const Scalar<DataVector> eint =
        eos.specific_internal_energy_from_density(rho);
    const double eint_hi_constant =
        (polytropic_exponent_hi - polytropic_exponent_lo) /
        ((polytropic_exponent_hi - 1.0) * (polytropic_exponent_lo - 1.0)) *
        polytropic_constant_lo *
        pow(transition_density, polytropic_exponent_lo - 1.0);
    const Scalar<DataVector> eint_expected{DataVector{
        polytropic_constant_lo / (polytropic_exponent_lo - 1.0) *
            pow(0.5 * transition_density, polytropic_exponent_lo - 1.0),
        polytropic_constant_lo / (polytropic_exponent_lo - 1.0) *
            pow(transition_density, polytropic_exponent_lo - 1.0),
        polytropic_constant_hi / (polytropic_exponent_hi - 1.0) *
                pow(2.0 * transition_density, polytropic_exponent_hi - 1.0) +
            eint_hi_constant}};
    CHECK(eint == eint_expected);

    // chi
    // Note the sound speeds at the transition density are different depending
    // on which pair of constant/exponent is picked.  Here the hi constant &
    // exponent must be chosen to match choose_polytropic_properties() in
    // PiecewisePolytropicFluid.cpp
    const Scalar<DataVector> chi = eos.chi_from_density(rho);
    const Scalar<DataVector> chi_expected{DataVector{
        polytropic_constant_lo * polytropic_exponent_lo *
            pow(0.5 * transition_density, polytropic_exponent_lo - 1.0),
        polytropic_constant_hi * polytropic_exponent_hi *
            pow(transition_density, polytropic_exponent_hi - 1.0),
        polytropic_constant_hi * polytropic_exponent_hi *
            pow(2.0 * transition_density, polytropic_exponent_hi - 1.0)}};
    CHECK(chi == chi_expected);

    // kappa
    const Scalar<DataVector> kappa_x_p_over_rho_sq_expected{
        DataVector{0.0, 0.0, 0.0}};
    const auto kappa_x_p_over_rho_sq =
        eos.kappa_times_p_over_rho_squared_from_density(rho);
    CHECK(kappa_x_p_over_rho_sq == kappa_x_p_over_rho_sq_expected);

    // rho from h
    if (IsRelativistic) {
      const Scalar<DataVector> spec_enthalpy{
          DataVector{1.0 + get_element(get(eint), 0) +
                         get_element(get(p), 0) / get_element(get(rho), 0),
                     1.0 + get_element(get(eint), 1) +
                         get_element(get(p), 1) / get_element(get(rho), 1),
                     1.0 + get_element(get(eint), 2) +
                         get_element(get(p), 2) / get_element(get(rho), 2)}};
      const auto rho_from_enthalpy =
          eos.rest_mass_density_from_enthalpy(spec_enthalpy);
      CHECK_ITERABLE_APPROX(rho, rho_from_enthalpy);
    } else {
      const Scalar<DataVector> spec_enthalpy{
          DataVector{get_element(get(eint), 0) +
                         get_element(get(p), 0) / get_element(get(rho), 0),
                     get_element(get(eint), 1) +
                         get_element(get(p), 1) / get_element(get(rho), 1),
                     get_element(get(eint), 2) +
                         get_element(get(p), 2) / get_element(get(rho), 2)}};
      const auto rho_from_enthalpy =
          eos.rest_mass_density_from_enthalpy(spec_enthalpy);
      CHECK_ITERABLE_APPROX(rho, rho_from_enthalpy);
    }
  }

  // tests of double type functions
  {
    // pressure
    const Scalar<double> rho{1.2 * transition_density};
    const auto p = eos.pressure_from_density(rho);
    const double p_expected =
        polytropic_constant_hi * pow(get(rho), polytropic_exponent_hi);
    CHECK(get(p) == p_expected);

    // eint
    const auto eint = eos.specific_internal_energy_from_density(rho);
    const double eint_hi_constant =
        (polytropic_exponent_hi - polytropic_exponent_lo) /
        ((polytropic_exponent_hi - 1.0) * (polytropic_exponent_lo - 1.0)) *
        polytropic_constant_lo *
        pow(transition_density, polytropic_exponent_lo - 1.0);
    const double eint_expected =
        polytropic_constant_hi / (polytropic_exponent_hi - 1.0) *
            pow(get(rho), polytropic_exponent_hi - 1.0) +
        eint_hi_constant;
    CHECK(get(eint) == eint_expected);

    // chi
    const auto chi = eos.chi_from_density(rho);
    const double chi_expected = polytropic_constant_hi *
                                polytropic_exponent_hi *
                                pow(get(rho), polytropic_exponent_hi - 1.0);
    CHECK(get(chi) == chi_expected);

    // kappa
    const auto kappa_x_p_over_rho_sq =
        eos.kappa_times_p_over_rho_squared_from_density(rho);
    const double kappa_x_p_over_rho_sq_expected = 0.0;
    CHECK(get(kappa_x_p_over_rho_sq) == kappa_x_p_over_rho_sq_expected);

    // rho from h
    double spec_enthalpy_constant = 0.0;
    if (IsRelativistic) {
      spec_enthalpy_constant = 1.0;
    }

    const Scalar<double> spec_enthalpy{spec_enthalpy_constant + get(eint) +
                                       get(p) / get(rho)};

    const auto rho_from_enthalpy =
        eos.rest_mass_density_from_enthalpy(spec_enthalpy);
    CHECK_ITERABLE_APPROX(get(rho), get(rho_from_enthalpy));
  }
}

template <bool IsRelativistic>
void check_edge_cases() {
  // Two edge cases checked are:
  // * The piecewise polytrope (with identical exponents) matches the single
  // polytrope case
  //
  // * The rest mass density at the transition density matches using both the
  //  high and low density constants/exponents

  const double transition_density = 10.0;
  const double polytropic_constant = 0.5;
  const double polytropic_exponent = 1.5;

  const Scalar<double> rest_mass_density_high{1.1 * transition_density};
  const Scalar<double> rest_mass_density_transition{transition_density};
  const Scalar<double> rest_mass_density_low{0.9 * transition_density};

  const auto eos = EquationsOfState::PiecewisePolytropicFluid<IsRelativistic>{
      transition_density, polytropic_constant, 0.9 * polytropic_exponent,
      1.1 * polytropic_exponent};

  // At the transition density, this EoS defaults to using the high density
  // constants and exponents to calculate remaining thermodynamic quantities.
  // Here check that eint matches at the transition density using low density
  // constants and exponents as well.
  const double transition_pressure =
      polytropic_constant * pow(transition_density, 0.9 * polytropic_exponent);

  const double transition_spec_eint_low_params =
      transition_pressure /
      ((0.9 * polytropic_exponent - 1.0) * transition_density);

  CHECK(get(eos.specific_internal_energy_from_density(
            rest_mass_density_transition)) ==
        approx(transition_spec_eint_low_params));

  // Compare single and piecewise polytrope with same parameters
  const auto eos_piecewise_polytrope =
      EquationsOfState::PiecewisePolytropicFluid<IsRelativistic>{
          transition_density, polytropic_constant, polytropic_exponent,
          polytropic_exponent};
  const auto eos_single_polytrope =
      EquationsOfState::PolytropicFluid<IsRelativistic>{polytropic_constant,
                                                        polytropic_exponent};
  // Compare single and identical piecewise polytrope
  CHECK(get(eos_piecewise_polytrope.specific_internal_energy_from_density(
            rest_mass_density_low)) ==
        get(eos_single_polytrope.specific_internal_energy_from_density(
            rest_mass_density_low)));
  CHECK(get(eos_piecewise_polytrope.specific_internal_energy_from_density(
            rest_mass_density_high)) ==
        get(eos_single_polytrope.specific_internal_energy_from_density(
            rest_mass_density_high)));
  CHECK(get(eos_piecewise_polytrope.chi_from_density(rest_mass_density_low)) ==
        get(eos_single_polytrope.chi_from_density(rest_mass_density_low)));
  CHECK(get(eos_piecewise_polytrope.chi_from_density(rest_mass_density_high)) ==
        get(eos_single_polytrope.chi_from_density(rest_mass_density_high)));
}

void check_dominant_energy_condition_at_bound() {
  MAKE_GENERATOR(generator);
  auto distribution = std::uniform_real_distribution<>{2.0, 3.0};  //[a,b)
  const double polytropic_exponent = distribution(generator);
  const double transition_density = 1.0;
  const double polytropic_constant = 0.5;
  const auto eos = EquationsOfState::PiecewisePolytropicFluid<true>{
      transition_density, 0.5 * polytropic_constant, 0.5 * polytropic_exponent,
      polytropic_exponent};
  const Scalar<double> rest_mass_density{eos.rest_mass_density_upper_bound()};
  const double specific_internal_energy =
      get(eos.specific_internal_energy_from_density(rest_mass_density));
  const double pressure =
      get(eos.pressure_from_density(Scalar<double>{rest_mass_density}));
  const double energy_density =
      get(rest_mass_density) * (1.0 + specific_internal_energy);
  CAPTURE(rest_mass_density);
  CAPTURE(specific_internal_energy);
  CHECK(approx(pressure) == energy_density);
}

template <bool IsRelativistic>
void check_bounds() {
  const double transition_density = 10.0;
  const double polytropic_constant = 0.5;
  const double polytropic_exponent = 1.5;
  const auto eos = EquationsOfState::PiecewisePolytropicFluid<IsRelativistic>{
      transition_density, 0.5 * polytropic_constant, 0.5 * polytropic_exponent,
      1.0 * polytropic_exponent};
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(0.0 == eos.specific_internal_energy_lower_bound(1.0));
  if constexpr (IsRelativistic) {
    CHECK(1.0 == eos.specific_enthalpy_lower_bound());
  } else {
    CHECK(0.0 == eos.specific_enthalpy_lower_bound());
  }
  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound(1.0));

  const auto eos_high_gamma =
      EquationsOfState::PiecewisePolytropicFluid<IsRelativistic>{
          transition_density, 0.5 * polytropic_constant,
          0.5 * polytropic_exponent, 2.0 * polytropic_exponent};
  double density_upper_limit = max_double;

  if constexpr (IsRelativistic) {
    const double polytropic_constant_hi =
        calc_Khi(transition_density, 0.5 * polytropic_constant,
                 0.5 * polytropic_exponent, 2.0 * polytropic_exponent);

    const double boundary_constant =
        (2.0 * polytropic_exponent - 0.5 * polytropic_exponent) * 0.5 *
        polytropic_constant /
        ((2.0 * polytropic_exponent - 1.0) *
         (0.5 * polytropic_exponent - 1.0)) *
        pow(transition_density, 0.5 * polytropic_exponent - 1.0);

    density_upper_limit =
        pow((2.0 * polytropic_exponent - 1.0) /
                (polytropic_constant_hi * (2.0 * polytropic_exponent - 2.0)) *
                (1.0 + boundary_constant),
            1.0 / (2.0 * polytropic_exponent - 1.0));
  }
  CHECK(density_upper_limit == eos_high_gamma.rest_mass_density_upper_bound());
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.EquationsOfState.PiecewisePolytropicFluid",
    "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
  register_derived_classes_with_charm<EoS::EquationOfState<false, 1>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  TestHelpers::EquationsOfState::test_get_clone(
      EoS::PiecewisePolytropicFluid<true>(10.0, 0.5, 1.5, 2.0));

  const auto eos = EoS::PiecewisePolytropicFluid<true>{10.0, 0.5, 1.5, 2.0};
  const auto other_eos =
      EoS::PiecewisePolytropicFluid<true>{10.0, 0.7, 1.5, 2.0};
  const auto other_type_eos = EoS::DarkEnergyFluid<true>{0.5};
  // Same parameters should match
  CHECK(eos == eos);
  // Different parameters should NOT match
  CHECK(eos != other_eos);
  // Different eos types should NOT match
  CHECK(eos != other_type_eos);
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);
  const double transition_density = 10.0;
  const double poly_exponent_lo = 1.5;
  const double poly_constant_lo = 10.0;
  const double poly_exponent_hi = 2.0;

  // Relativistic checks
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<true>{transition_density, poly_constant_lo,
                                          poly_exponent_lo, poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo, poly_exponent_hi);
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<true>{transition_density,
                                          1.1 * poly_constant_lo,
                                          poly_exponent_lo, poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, 1.1 * poly_constant_lo, poly_exponent_lo,
      poly_exponent_hi);

  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<true>{transition_density, poly_constant_lo,
                                          1.2 * poly_exponent_lo,
                                          poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, 1.2 * poly_exponent_lo,
      poly_exponent_hi);
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<true>{transition_density, poly_constant_lo,
                                          poly_exponent_lo,
                                          1.4 * poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo,
      1.4 * poly_exponent_hi);

  // Non relativistic checks
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<false>{transition_density, poly_constant_lo,
                                           poly_exponent_lo, poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo, poly_exponent_hi);
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<false>{transition_density,
                                           1.1 * poly_constant_lo,
                                           poly_exponent_lo, poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, 1.1 * poly_constant_lo, poly_exponent_lo,
      poly_exponent_hi);
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<false>{transition_density, poly_constant_lo,
                                           1.2 * poly_exponent_lo,
                                           poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, 1.2 * poly_exponent_lo,
      poly_exponent_hi);
  TestHelpers::EquationsOfState::check(
      EoS::PiecewisePolytropicFluid<false>{transition_density, poly_constant_lo,
                                           poly_exponent_lo,
                                           1.4 * poly_exponent_hi},
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo,
      1.4 * poly_exponent_hi);

  // Relativistic
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 1>>>(
          {"PiecewisePolytropicFluid:\n"
           "  PiecewisePolytropicTransitionDensity: " +
           std::to_string(transition_density) +
           "\n"
           "  PolytropicConstantLow: " +
           std::to_string(poly_constant_lo) +
           "\n"
           "  PolytropicExponentLow: " +
           std::to_string(poly_exponent_lo) +
           "\n"
           "  PolytropicExponentHigh: " +
           std::to_string(poly_exponent_hi) + "\n"}),
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo, poly_exponent_hi);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 1>>>(
          {"PiecewisePolytropicFluid:\n"
           "  PiecewisePolytropicTransitionDensity: " +
           std::to_string(transition_density) +
           "\n"
           "  PolytropicConstantLow: " +
           std::to_string(poly_constant_lo) +
           "\n"
           "  PolytropicExponentLow: " +
           std::to_string(poly_exponent_lo) +
           "\n"
           "  PolytropicExponentHigh: " +
           std::to_string(1.5 * poly_exponent_hi) + "\n"}),
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo,
      1.5 * poly_exponent_hi);

  // Nonrelativistic
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 1>>>(
          {"PiecewisePolytropicFluid:\n"
           "  PiecewisePolytropicTransitionDensity: " +
           std::to_string(transition_density) +
           "\n"
           "  PolytropicConstantLow: " +
           std::to_string(poly_constant_lo) +
           "\n"
           "  PolytropicExponentLow: " +
           std::to_string(poly_exponent_lo) +
           "\n"
           "  PolytropicExponentHigh: " +
           std::to_string(poly_exponent_hi) + "\n"}),
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo, poly_exponent_hi);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 1>>>(
          {"PiecewisePolytropicFluid:\n"
           "  PiecewisePolytropicTransitionDensity: " +
           std::to_string(transition_density) +
           "\n"
           "  PolytropicConstantLow: " +
           std::to_string(poly_constant_lo) +
           "\n"
           "  PolytropicExponentLow: " +
           std::to_string(poly_exponent_lo) +
           "\n"
           "  PolytropicExponentHigh: " +
           std::to_string(1.2 * poly_exponent_hi) + "\n"}),
      "PiecewisePolytropicFluid", "piecewisepolytropic", d_for_size,
      transition_density, poly_constant_lo, poly_exponent_lo,
      1.2 * poly_exponent_hi);

  check_bounds<true>();
  check_bounds<false>();
  check_dominant_energy_condition_at_bound();
  check_edge_cases<true>();
  check_edge_cases<false>();
  check_exact<true>();
  check_exact<false>();
}
