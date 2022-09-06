// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"

namespace EquationsOfState {
namespace {

void check_exact() {
  // Test creation
  Parallel::register_derived_classes_with_charm<EquationOfState<true, 1>>();
  const auto eos_pointer = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<EquationOfState<true, 1>>>(
          {"Enthalpy:\n"
           "  ReferenceDensity: 2.0\n"
           "  MinimumDensity: 4.0\n"
           "  MaximumDensity: 100.0\n"
           "  TrigScaling: 1.5\n"
           "  PolynomialCoefficients: [1.0,0.2,0.0,0.0,0.0001]\n"
           "  SinCoefficients: [0.01,0.003,-0.0001,0.0001]\n"
           "  CosCoefficients: [0.01,0.003,0.0001,0.00001]\n"
           "  TransitionDeltaEpsilon: 0.0\n"
           "  Spectral:\n"
           "    ReferenceDensity: 1.054388552462907080\n"
           "    ReferencePressure: 0.02168181441607176\n"
           "    Coefficients: [1.4, 0, -0.022880893142188646, "
           "0.7099134558804311]\n"
           "    UpperDensity: 4.0\n"}));

  const Enthalpy<Spectral>& eos =
      dynamic_cast<const Enthalpy<Spectral>&>(*eos_pointer);
  TestHelpers::EquationsOfState::test_get_clone(eos);

  const auto other_eos = Enthalpy<Spectral>{
      1.0,
      2.0,
      1.5,
      1.0,
      std::vector<double>{1.0, 1.0},
      std::vector<double>{0.1},
      std::vector<double>{0.1},
      Spectral{.5, .3, std::vector<double>{1.4, 0.0, 0.0}, 1.0},
      0.0};
  const auto other_low_eos =
      Enthalpy<PolytropicFluid<true>>{1.0,
                                      2.0,
                                      1.5,
                                      1.0,
                                      std::vector<double>{1.0, 1.0},
                                      std::vector<double>{0.1},
                                      std::vector<double>{0.1},
                                      PolytropicFluid<true>{100.0, 2.0},
                                      0.0};
  const auto other_hi_eos = Enthalpy<Spectral>{
      1.0,
      2.0,
      1.5,
      1.0,
      std::vector<double>{1.0, .95},
      std::vector<double>{0.05},
      std::vector<double>{0.05},
      Spectral{.5, .3, std::vector<double>{1.4, 0.0, 0.0}, 1.0},
      0.0};
  const auto other_type_eos = PolytropicFluid<true>{100.0, 2.0};
  CHECK(eos == eos);
  CHECK(other_eos != other_low_eos);
  CHECK(other_eos != other_hi_eos);
  CHECK(eos != other_eos);
  CHECK(eos != other_type_eos);
  CHECK(other_low_eos != other_type_eos);
  // Test DataVector functions
  {
    const Scalar<DataVector> rho{DataVector{1.5 * exp(1.0), 1.5 * exp(2.0),
                                            1.5 * exp(3.0), 1.5 * exp(4.0)}};
    const Scalar<DataVector> p = eos.pressure_from_density(rho);
    CAPTURE(rho);
    CAPTURE(p);
    const Scalar<DataVector> p_expected{
        DataVector{0.25505331617202415, 1.53718607768077375,
                   5.29831173857097415, 17.13581872708278553}};
    CHECK_ITERABLE_APPROX(p, p_expected);
    const auto eps_c = eos.specific_internal_energy_from_density(rho);
    const Scalar<DataVector> eps_expected{
        DataVector{0.09425055282366418, 0.19998963900481931,
                   0.36012641619526214, 0.55066416955025821}};
    CHECK_ITERABLE_APPROX(eps_c, eps_expected);
    const auto chi_c = eos.chi_from_density(rho);
    const Scalar<DataVector> chi_expected{
        DataVector{0.18207356789438428, 0.1923165022214991, 0.19908149093638267,
                   0.25186554704031844}};
    CHECK_ITERABLE_APPROX(get(chi_expected), get(chi_c));
    const Scalar<DataVector> p_c_kappa_c_over_rho_sq_expected{
        DataVector{0.0, 0.0, 0.0, 0.0}};
    const auto p_c_kappa_c_over_rho_sq =
        eos.kappa_times_p_over_rho_squared_from_density(rho);
    CHECK_ITERABLE_APPROX(p_c_kappa_c_over_rho_sq_expected,
                          p_c_kappa_c_over_rho_sq);
    const auto rho_from_enthalpy = eos.rest_mass_density_from_enthalpy(
        hydro::relativistic_specific_enthalpy(rho, eps_c, p));
    CHECK_ITERABLE_APPROX(rho, rho_from_enthalpy);
  }
  // Test low density stitched EoS
  {
    const Scalar<DataVector> rho{DataVector{1.5 * exp(-1.0), 1.5}};
    const Scalar<DataVector> p = eos.pressure_from_density(rho);
    const Scalar<DataVector> p_expected{
        DataVector{0.00875810516598451, 0.03560143071808177}};
    CHECK_ITERABLE_APPROX(p, p_expected);
    const auto eps_c = eos.specific_internal_energy_from_density(rho);
    const Scalar<DataVector> eps_expected{
        DataVector{0.03967833020738167, 0.05919690661251007}};
    CHECK_ITERABLE_APPROX(eps_c, eps_expected);
    const auto chi_c = eos.chi_from_density(rho);
    const Scalar<DataVector> chi_expected{
        DataVector{0.02221986491613373, 0.03389855168318545}};
    CHECK_ITERABLE_APPROX(get(chi_expected), get(chi_c));
    const Scalar<DataVector> p_c_kappa_c_over_rho_sq_expected{
        DataVector{0.0, 0.0}};
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
    const Scalar<double> rho{1.5 * exp(1.0)};
    const auto p = eos.pressure_from_density(rho);
    const double p_expected = 0.25505331617202415;
    CHECK(get(p) == approx(p_expected));
    const auto eps = eos.specific_internal_energy_from_density(rho);
    const double eps_expected = 0.09425055282366418;
    CHECK(get(eps) == approx(eps_expected));
    const auto chi = eos.chi_from_density(rho);
    const double chi_expected = 0.18207356789438428;
    CHECK(get(chi) == approx(chi_expected));
    const auto p_c_kappa_c_over_rho_sq =
        eos.kappa_times_p_over_rho_squared_from_density(rho);
    const double p_c_kappa_c_over_rho_sq_expected = 0.0;
    CHECK(get(p_c_kappa_c_over_rho_sq) ==
          approx(p_c_kappa_c_over_rho_sq_expected));
    const auto rho_from_enthalpy = eos.rest_mass_density_from_enthalpy(
        hydro::relativistic_specific_enthalpy(rho, eps, p));
    CHECK(get(rho) == approx(get(rho_from_enthalpy)));
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

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Enthalpy",
                  "[Unit][EquationsOfState]") {
  check_exact();
}
}  // namespace EquationsOfState
