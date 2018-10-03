// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace {

constexpr double polytropic_constant = 8.0;
double expected_newtonian_radius() noexcept {
  /* Analytic solution in Newtonian limit for polytropic TOV integration has
  $R = \pi / \alpha$ ; $M = 4\pi^{2}\rho_{0c} / \alpha^{3}$
   Here $\rho_{0c}$ is the central mass density and $\alpha = sqrt(2\pi/K)$
   with $K$ the polytropic constant
  */
  return M_PI / sqrt(2.0 * M_PI / polytropic_constant);
}

double expected_newtonian_mass(const double central_mass_density) noexcept {
  return 4.0 * central_mass_density * square(M_PI) /
         (cube(sqrt(2.0 * M_PI / polytropic_constant)));
}

void test_tov(const std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>&
                  equation_of_state,
              const double central_mass_density, const size_t num_pts,
              const size_t current_iteration,
              const bool newtonian_limit) noexcept {
  Approx custom_approx = Approx::custom().epsilon(1.0e-08).scale(1.0);
  const double initial_log_enthalpy =
      std::log(get(equation_of_state->specific_enthalpy_from_density(
          Scalar<double>{central_mass_density})));
  const double surface_log_enthalpy = 0.0;
  const double step = (surface_log_enthalpy - initial_log_enthalpy) / num_pts;
  const double final_log_enthalpy =
      initial_log_enthalpy + (current_iteration + 1.0) * step;
  const gr::Solutions::TovSolution tov_out_full(
      equation_of_state, central_mass_density, surface_log_enthalpy);

  if (newtonian_limit) {
    const double final_radius{tov_out_full.outer_radius()};
    const double final_mass{tov_out_full.mass(final_radius)};
    CHECK(expected_newtonian_radius() == custom_approx(final_radius));
    CHECK(expected_newtonian_mass(central_mass_density) ==
          custom_approx(final_mass));
  } else {
    //  Values in relativistic limit obtained from SpEC for
    //  polytropic_constant = 8.0 and polytropic_exponent = 2.0
    constexpr double expected_relativistic_radius = 3.4685521362;
    constexpr double expected_relativistic_mass = 0.0531036941;
    const double final_radius{tov_out_full.outer_radius()};
    const double final_mass{tov_out_full.mass(final_radius)};
    CHECK(expected_relativistic_radius == custom_approx(final_radius));
    CHECK(expected_relativistic_mass == custom_approx(final_mass));
  }

  // Integrate only to some intermediate value, not the surface. Then compare to
  // interpolated values.
  const gr::Solutions::TovSolution tov_out_intermediate(
      equation_of_state, central_mass_density, final_log_enthalpy);
  const double intermediate_radius{tov_out_intermediate.outer_radius()};
  const double intermediate_mass{
      tov_out_intermediate.mass(intermediate_radius)};
  const double intermediate_log_enthalpy{
      tov_out_intermediate.log_specific_enthalpy(intermediate_radius)};
  const double intermediate_enthalpy{
      tov_out_intermediate.specific_enthalpy(intermediate_radius)};
  const double interpolated_mass{tov_out_full.mass(intermediate_radius)};
  const double interpolated_log_enthalpy{
      tov_out_full.log_specific_enthalpy(intermediate_radius)};
  const double interpolated_enthalpy{
      tov_out_full.specific_enthalpy(intermediate_radius)};
  CHECK(intermediate_mass == custom_approx(interpolated_mass));
  CHECK(intermediate_log_enthalpy == custom_approx(interpolated_log_enthalpy));
  CHECK(intermediate_enthalpy == custom_approx(interpolated_enthalpy));

  const Scalar<DataVector> intermediate_radius_dv{5_st, intermediate_radius};
  const Scalar<DataVector> intermediate_mass_dv =
      tov_out_intermediate.mass(intermediate_radius_dv);
  const Scalar<DataVector> intermediate_log_enthalpy_dv =
      tov_out_intermediate.log_specific_enthalpy(intermediate_radius_dv);
  const Scalar<DataVector> intermediate_enthalpy_dv =
      tov_out_intermediate.specific_enthalpy(intermediate_radius_dv);
  const Scalar<DataVector> interpolated_mass_dv =
      tov_out_full.mass(intermediate_radius_dv);
  const Scalar<DataVector> interpolated_log_enthalpy_dv =
      tov_out_full.log_specific_enthalpy(intermediate_radius_dv);
  const Scalar<DataVector> interpolated_enthalpy_dv =
      tov_out_full.specific_enthalpy(intermediate_radius_dv);
  CHECK_ITERABLE_CUSTOM_APPROX(intermediate_mass_dv, interpolated_mass_dv,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(intermediate_log_enthalpy_dv,
                               interpolated_log_enthalpy_dv, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(intermediate_enthalpy_dv,
                               interpolated_enthalpy_dv, custom_approx);

  const auto deserialized_tov_out_full =
      serialize_and_deserialize(tov_out_full);
  const auto deserialized_tov_out_intermediate =
      serialize_and_deserialize(tov_out_intermediate);
  const double intermediate_radius_ds{
      deserialized_tov_out_intermediate.outer_radius()};
  const double intermediate_mass_ds{
      deserialized_tov_out_intermediate.mass(intermediate_radius_ds)};
  const double intermediate_log_enthalpy_ds{
      deserialized_tov_out_intermediate.log_specific_enthalpy(
          intermediate_radius_ds)};
  const double intermediate_enthalpy_ds{
      deserialized_tov_out_intermediate.specific_enthalpy(
          intermediate_radius_ds)};
  const double interpolated_mass_ds{
      deserialized_tov_out_full.mass(intermediate_radius_ds)};
  const double interpolated_log_enthalpy_ds{
      deserialized_tov_out_full.log_specific_enthalpy(intermediate_radius_ds)};
  const double interpolated_enthalpy_ds{
      deserialized_tov_out_full.specific_enthalpy(intermediate_radius_ds)};
  CHECK(intermediate_mass_ds == custom_approx(interpolated_mass_ds));
  CHECK(intermediate_log_enthalpy_ds ==
        custom_approx(interpolated_log_enthalpy_ds));
  CHECK(intermediate_enthalpy_ds == custom_approx(interpolated_enthalpy_ds));
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.Tov",
                  "[Unit][PointwiseFunctions]") {
  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
      equation_of_state =
          std::make_unique<EquationsOfState::PolytropicFluid<true>>(
              polytropic_constant, 2.0);
  /* Each iteration of the loop is for a different value of the final
     log_enthalpy in the integration. This is done to test the interpolation:
     the integration is stopped at some final log_enthalpy between the center
     and surface, and values are compared to those obtained by interpolation of
     the full integration up to the surface.
  */
  const size_t num_pts = 25;
  for (size_t i = 0; i < num_pts; i++) {
    test_tov(equation_of_state, 1.0e-10, num_pts, i, true);
    test_tov(equation_of_state, 1.0e-03, num_pts, i, false);
  }
}

}  // namespace
