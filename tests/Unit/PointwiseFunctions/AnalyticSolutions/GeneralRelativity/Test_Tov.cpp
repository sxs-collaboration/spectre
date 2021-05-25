// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"

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

void test_tov(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const double central_mass_density, const size_t num_pts,
    const size_t current_iteration, const bool newtonian_limit) noexcept {
  Approx custom_approx = Approx::custom().epsilon(1.0e-08).scale(1.0);
  const double initial_log_enthalpy =
      std::log(get(equation_of_state.specific_enthalpy_from_density(
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
  const double interpolated_mass{tov_out_full.mass(intermediate_radius)};
  const double interpolated_log_enthalpy{
      tov_out_full.log_specific_enthalpy(intermediate_radius)};
  CHECK(intermediate_mass == custom_approx(interpolated_mass));
  CHECK(intermediate_log_enthalpy == custom_approx(interpolated_log_enthalpy));

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
  const double interpolated_mass_ds{
      deserialized_tov_out_full.mass(intermediate_radius_ds)};
  const double interpolated_log_enthalpy_ds{
      deserialized_tov_out_full.log_specific_enthalpy(intermediate_radius_ds)};
  CHECK(intermediate_mass_ds == custom_approx(interpolated_mass_ds));
  CHECK(intermediate_log_enthalpy_ds ==
        custom_approx(interpolated_log_enthalpy_ds));
}

void test_tov_dp_dr(
    const std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>&
        equation_of_state) {
  const gr::Solutions::TovSolution radial_tov_solution(*equation_of_state,
                                                       1.0e-3);
  const size_t num_radial_pts = 500;
  auto coords = make_with_value<tnsr::I<DataVector, 3>>(num_radial_pts, 0.0);
  const double dx = radial_tov_solution.outer_radius() * 1.1 / num_radial_pts;
  for (size_t i = 0; i < get<0>(coords).size(); ++i) {
    get<0>(coords)[i] = i * dx;
  }
  const auto vars =
      radial_tov_solution.radial_variables(*equation_of_state, coords);
  const auto& pressure = vars.pressure;
  const auto& density = vars.rest_mass_density;
  const auto& specific_internal_energy = vars.specific_internal_energy;
  const auto& dp_dr = vars.dr_pressure;

  const auto& radii = vars.radial_coordinate;
  auto custom_approx = Approx::custom().epsilon(1.e-6);
  for (size_t i = 2; i < num_radial_pts - 2; ++i) {
    CAPTURE(i);
    CAPTURE(radii[i]);
    CAPTURE(radial_tov_solution.outer_radius());
    CAPTURE(get(pressure)[i]);
    CAPTURE(get(density)[i]);
    if (radii[i + 2] < radial_tov_solution.outer_radius()) {
      CHECK((-get(pressure)[i + 2] / 12.0 + 2.0 / 3.0 * get(pressure)[i + 1] -
             2.0 / 3.0 * get(pressure)[i - 1] + get(pressure)[i - 2] / 12.0) /
                dx ==
            custom_approx(dp_dr[i]));
    } else if (radii[i] >= radial_tov_solution.outer_radius()) {
      CHECK(dp_dr[i] == 0.0);
    }
  }

  for (size_t i = 0; i < num_radial_pts; ++i) {
    const double total_energy_density =
        get(density)[i] * (1.0 + get(specific_internal_energy)[i]);
    const double p = get(pressure)[i];
    const double radius = radii[i];
    CAPTURE(total_energy_density);
    CAPTURE(p);
    CAPTURE(radius);
    if (radius < radial_tov_solution.outer_radius() and radius > 0.0) {
      const double m = radial_tov_solution.mass(radius);
      CHECK(approx(dp_dr[i]) == -total_energy_density * m / square(radius) *
                                    (1.0 + p / total_energy_density) *
                                    (1.0 + 4.0 * M_PI * p * cube(radius) / m) /
                                    (1.0 - 2.0 * m / radius));
    } else {
      CHECK(approx(dp_dr[i]) == 0.0);
    }
  }
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
    test_tov(*equation_of_state, 1.0e-10, num_pts, i, true);
    test_tov(*equation_of_state, 1.0e-03, num_pts, i, false);
  }

  test_tov_dp_dr(equation_of_state);
}

}  // namespace
