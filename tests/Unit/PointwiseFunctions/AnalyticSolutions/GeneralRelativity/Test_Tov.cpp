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
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace {

constexpr double polytropic_constant = 8.0;
double expected_newtonian_radius() {
  /* Analytic solution in Newtonian limit for polytropic TOV integration has
  $R = \pi / \alpha$ ; $M = 4\pi^{2}\rho_{0c} / \alpha^{3}$
   Here $\rho_{0c}$ is the central mass density and $\alpha = sqrt(2\pi/K)$
   with $K$ the polytropic constant
  */
  return M_PI / sqrt(2.0 * M_PI / polytropic_constant);
}

double expected_newtonian_mass(const double central_mass_density) {
  return 4.0 * central_mass_density * square(M_PI) /
         (cube(sqrt(2.0 * M_PI / polytropic_constant)));
}

void test_tov(
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const double central_mass_density, const size_t num_pts,
    const size_t current_iteration, const bool newtonian_limit) {
  Approx custom_approx = Approx::custom().epsilon(1.0e-08).scale(1.0);
  const double initial_log_enthalpy =
      std::log(get(equation_of_state.specific_enthalpy_from_density(
          Scalar<double>{central_mass_density})));
  const double surface_log_enthalpy = 0.0;
  const double step = (surface_log_enthalpy - initial_log_enthalpy) / num_pts;
  const double final_log_enthalpy =
      initial_log_enthalpy + (current_iteration + 1.0) * step;
  const gr::Solutions::TovSolution tov_out_full(
      equation_of_state, central_mass_density,
      gr::Solutions::TovCoordinates::Schwarzschild, surface_log_enthalpy);

  if (newtonian_limit) {
    const double final_radius{tov_out_full.outer_radius()};
    const double final_mass{tov_out_full.total_mass()};
    CHECK(final_mass ==
          approx(tov_out_full.mass_over_radius(final_radius) * final_radius));
    CHECK(expected_newtonian_radius() == custom_approx(final_radius));
    CHECK(expected_newtonian_mass(central_mass_density) ==
          custom_approx(final_mass));
  } else {
    //  Values in relativistic limit obtained from SpEC for
    //  polytropic_constant = 8.0 and polytropic_exponent = 2.0
    constexpr double expected_relativistic_radius = 3.4685521362;
    constexpr double expected_relativistic_mass = 0.0531036941;
    const double final_radius{tov_out_full.outer_radius()};
    const double final_mass{tov_out_full.total_mass()};
    CHECK(final_mass ==
          approx(tov_out_full.mass_over_radius(final_radius) * final_radius));
    CHECK(expected_relativistic_radius == custom_approx(final_radius));
    CHECK(expected_relativistic_mass == custom_approx(final_mass));
  }
  CHECK(tov_out_full.injection_energy() ==
        approx(sqrt(1. - 2. * tov_out_full.total_mass() /
                             tov_out_full.outer_radius())));

  // Integrate only to some intermediate value, not the surface. Then compare to
  // interpolated values.
  const gr::Solutions::TovSolution tov_out_intermediate(
      equation_of_state, central_mass_density,
      gr::Solutions::TovCoordinates::Schwarzschild, final_log_enthalpy);
  const double intermediate_radius{tov_out_intermediate.outer_radius()};
  const double intermediate_mass{tov_out_intermediate.total_mass()};
  CHECK(intermediate_mass ==
        approx(tov_out_intermediate.mass_over_radius(intermediate_radius) *
               intermediate_radius));
  const double intermediate_log_enthalpy{
      tov_out_intermediate.log_specific_enthalpy(intermediate_radius)};
  const double interpolated_mass{
      tov_out_full.mass_over_radius(intermediate_radius) * intermediate_radius};
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
      deserialized_tov_out_intermediate.total_mass()};
  const double intermediate_log_enthalpy_ds{
      deserialized_tov_out_intermediate.log_specific_enthalpy(
          intermediate_radius_ds)};
  const double interpolated_mass_ds{
      deserialized_tov_out_full.mass_over_radius(intermediate_radius_ds) *
      intermediate_radius_ds};
  const double interpolated_log_enthalpy_ds{
      deserialized_tov_out_full.log_specific_enthalpy(intermediate_radius_ds)};
  CHECK(intermediate_mass_ds == custom_approx(interpolated_mass_ds));
  CHECK(intermediate_log_enthalpy_ds ==
        custom_approx(interpolated_log_enthalpy_ds));
}

void test_tov_dp_dr(const EquationsOfState::EquationOfState<true, 1>& eos) {
  const double central_rest_mass_density = 1.0e-3;
  const gr::Solutions::TovSolution radial_tov_solution{
      eos, central_rest_mass_density};

  // Evaluate the radial solution at equally spaced points
  const size_t num_radial_pts = 500;
  const double dx = radial_tov_solution.outer_radius() / num_radial_pts;
  DataVector radii{num_radial_pts};
  Scalar<DataVector> density{num_radial_pts};
  Scalar<DataVector> pressure{num_radial_pts};
  DataVector dp_dr{num_radial_pts};
  for (size_t i = 0; i < num_radial_pts; ++i) {
    const double radius = i * dx;
    radii[i] = radius;
    const double mass_over_radius =
        radial_tov_solution.mass_over_radius(radius);
    const double specific_enthalpy =
        exp(radial_tov_solution.log_specific_enthalpy(radius));
    const Scalar<double> density_i =
        eos.rest_mass_density_from_enthalpy(Scalar<double>(specific_enthalpy));
    get(density)[i] = get(density_i);
    get(pressure)[i] = get(eos.pressure_from_density(density_i));
    // This is the TOV equation in its standard form
    if (radius > 0.) {
      const double specific_internal_energy =
          get(eos.specific_internal_energy_from_density(density_i));
      dp_dr[i] =
          -(get(density)[i] * (1. + specific_internal_energy) +
            get(pressure)[i]) *
          (mass_over_radius / radius + 4. * M_PI * radius * get(pressure)[i]) /
          (1. - 2 * mass_over_radius);
    } else {
      dp_dr[i] = 0.;
      // Check the central density
      CHECK(get(density)[i] == approx(central_rest_mass_density));
    }
  }

  // Take a finite-difference numerical derivative of the pressure from the
  // radial solution, and compare to the TOV equation dp/dr
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
}

void test_baumgarte_shapiro() {
  // Reproduces Fig. 1.2 in BaumgarteShapiro, as suggested in footnote 25 on p.
  // 18, and as listed in Table 14.1
  const EquationsOfState::PolytropicFluid<true> eos{1., 2.};
  const double central_energy_density = 0.42;
  // For polytropic_exponent = 2
  const double central_rest_mass_density =
      0.5 * (sqrt(1. + 4. * central_energy_density) - 1.);
  const gr::Solutions::TovSolution tov{eos, central_rest_mass_density};
  // The values in BaumgarteShapiro are given to this precision
  Approx custom_approx = Approx::custom().epsilon(1.e-03).scale(1.0);
  CHECK(tov.total_mass() == custom_approx(0.164));
  CHECK(tov.outer_radius() == custom_approx(0.763));
}

void test_tov_isotropic(const EquationsOfState::EquationOfState<true, 1>& eos,
                        const double central_mass_density) {
  const gr::Solutions::TovSolution tov_areal{
      eos, central_mass_density, gr::Solutions::TovCoordinates::Schwarzschild};
  const gr::Solutions::TovSolution tov_isotropic{
      eos, central_mass_density, gr::Solutions::TovCoordinates::Isotropic};
  const double outer_isotropic_radius = tov_isotropic.outer_radius();
  CHECK(tov_areal.total_mass() == approx(tov_isotropic.total_mass()));
  const double outer_areal_radius =
      outer_isotropic_radius *
      square(tov_isotropic.conformal_factor(outer_isotropic_radius));
  CHECK(tov_areal.outer_radius() == approx(outer_areal_radius));
  CHECK(tov_isotropic.mass_over_radius(0.) == approx(0.));
  CHECK(tov_isotropic.mass_over_radius(outer_isotropic_radius) ==
        approx(tov_isotropic.total_mass() / outer_areal_radius));
  CHECK(tov_isotropic.log_specific_enthalpy(0.) ==
        approx(log(get(eos.specific_enthalpy_from_density(
            Scalar<double>{central_mass_density})))));
  CHECK(tov_isotropic.log_specific_enthalpy(outer_isotropic_radius) ==
        approx(0.));
  CHECK(tov_isotropic.conformal_factor(outer_isotropic_radius) ==
        approx(1 + 0.5 * tov_isotropic.total_mass() / outer_isotropic_radius));
}

void test_rueter() {
  // Reproduces the values in Sec. V.C and Fig. 8 in
  // https://arxiv.org/abs/1708.07358
  const EquationsOfState::PolytropicFluid<true> eos{123.6489, 2};
  gr::Solutions::TovSolution solution{
      eos,
      0.0008087415253997405,  // Central enthalpy h=1.2
      gr::Solutions::TovCoordinates::Isotropic};
  const double outer_radius = solution.outer_radius();
  // The values in the paper are given to this precision
  Approx custom_approx = Approx::custom().epsilon(1e-4).scale(1.);
  CHECK(outer_radius == custom_approx(9.7098));
  CHECK(solution.conformal_factor(0.) == custom_approx(1.16));
  CHECK(solution.conformal_factor(outer_radius) == custom_approx(1.0727));
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

  test_tov_dp_dr(*equation_of_state);
  test_baumgarte_shapiro();
  test_tov_isotropic(*equation_of_state, 1.e-3);
  test_rueter();
}

}  // namespace
