// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {

void check_exact() {
  // Test creation
  namespace EoS = EquationsOfState;
  const std::vector coefs = {3.0, 0.25, 0.375, 0.5};
  TestHelpers::test_creation<std::unique_ptr<EoS::EquationOfState<true, 1>>>(
      {"Spectral:\n"
       "  ReferenceDensity: 2.0\n"
       "  ReferencePressure: 4.0\n"
       "  Coefficients: [3.0,0.25,0.375,0.5]\n"
       "  UpperDensity: 15.0\n"});

  EquationsOfState::Spectral eos(2.0, 4.0, {3.0, 0.25, 0.375, 0.5},
                                 2.0 * exp(2.0));
  TestHelpers::EquationsOfState::test_get_clone(eos);

  EquationsOfState::Spectral other_eos(1.0, 4.0, {3.0, 0.25, 0.375, 0.5},
                                       2.0 * exp(2.0));
  const auto other_type_eos = EoS::PolytropicFluid<true>{100.0, 2.0};
  CHECK(eos == eos);
  CHECK(eos != other_eos);
  CHECK(eos != other_type_eos);
  // Test DataVector functions
  {
    const Scalar<DataVector> rho{DataVector{
        2.0 * exp(-1.0), 2.0, 2.0 * exp(1.0), 2.0 * exp(2.0), 2.0 * exp(3.0)}};
    const Scalar<DataVector> p = eos.pressure_from_density(rho);
    INFO(rho);
    INFO(p);
    const Scalar<DataVector> p_expected{
        DataVector{4.0 * exp(-3.0), 4.0, 4.0 * exp(3.375), 4.0 * exp(9.5),
                   4.0 * exp(18.5)}};
    CHECK(p == p_expected);
    const auto eps_c = eos.specific_internal_energy_from_density(rho);
    const Scalar<DataVector> eps_expected{
        DataVector{exp(-2.0), 1.0, 8.519830698147826, 528.9617452597413,
                   1347501.570212399}};
    CHECK_ITERABLE_APPROX(eps_c, eps_expected);
    const auto chi_c = eos.chi_from_density(rho);
    const Scalar<DataVector> gamma{DataVector{3.0, 3.0, 4.125, 9.0, 9.0}};
    const auto chi_expected = get(p_expected) * get(gamma) / get(rho);
    CHECK_ITERABLE_APPROX(chi_expected, get(chi_c));
    const Scalar<DataVector> p_c_kappa_c_over_rho_sq_expected{
        DataVector{0.0, 0.0, 0.0, 0.0, 0.0}};
    const auto p_c_kappa_c_over_rho_sq =
        eos.kappa_times_p_over_rho_squared_from_density(rho);
    CHECK_ITERABLE_APPROX(p_c_kappa_c_over_rho_sq_expected,
                          p_c_kappa_c_over_rho_sq);
    const auto rho_from_enthalpy = eos.rest_mass_density_from_enthalpy(
        hydro::relativistic_specific_enthalpy(rho, eps_c, p));
    CHECK_ITERABLE_APPROX(rho, rho_from_enthalpy);
  }
  // Test double functions
  {
    const Scalar<double> rho{2.0 * exp(1.0)};
    const auto p = eos.pressure_from_density(rho);
    const double p_expected = 4.0 * exp(3.375);
    CHECK(get(p) == p_expected);
    const auto eps = eos.specific_internal_energy_from_density(rho);
    const double eps_expected = 8.519830698147826;
    CHECK_ITERABLE_APPROX(get(eps), eps_expected);
    const auto chi = eos.chi_from_density(rho);
    const double chi_expected = p_expected * 4.125 / get(rho);
    CHECK_ITERABLE_APPROX(chi_expected, get(chi));
    const auto p_c_kappa_c_over_rho_sq =
        eos.kappa_times_p_over_rho_squared_from_density(rho);
    const double p_c_kappa_c_over_rho_sq_expected = 0.0;
    CHECK_ITERABLE_APPROX(p_c_kappa_c_over_rho_sq_expected,
                          get(p_c_kappa_c_over_rho_sq));
    const auto rho_from_enthalpy = eos.rest_mass_density_from_enthalpy(
        hydro::relativistic_specific_enthalpy(rho, eps, p));
    CHECK_ITERABLE_APPROX(get(rho), get(rho_from_enthalpy));
  }
  // Test bounds
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(0.0 == eos.specific_internal_energy_lower_bound(1.0));
  CHECK(1.0 == eos.specific_enthalpy_lower_bound());
  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound(1.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Spectral",
                  "[Unit][EquationsOfState]") {
  check_exact();
}
