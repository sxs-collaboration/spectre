// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/TOV.hpp"
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/EquationsOfState/PolytropicFluid.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"

// Analytic solution in Newtonian limit for polytropic TOV integration has R =
// pi/alpha, M = 4*pi^2*rho0c/(alpha^3)
// Here rho0c is the central mass density and alpha = sqrt(2pi/K) where K is the
// polytropic constant

namespace {

double expected_radius() noexcept {
  const double polytropic_constant = 4.3419;
  const double alpha = std::sqrt(2 * M_PI / polytropic_constant);

  return M_PI / alpha;
}

double expected_mass() noexcept {
  const double polytropic_constant = 4.3419;
  const double alpha = std::sqrt(2 * M_PI / polytropic_constant);
  const double central_mass_density = 1.0e-10;

  return 4 * central_mass_density * pow<2>(M_PI) / (pow<3>(alpha));
}

template <bool IsRelativistic, size_t dim>
void test_newtonian_tov(
    std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, dim>>&
        polyM,
    double central_mass_density, double h_final) noexcept {
  Approx custom_approx = Approx::custom().epsilon(1.0e-08).scale(1.0);

  tov::TOV_Output tov_instance;

  tov::InterpolationOutput tov_out_full =
      tov_instance.tov_solver(polyM, central_mass_density);

  tov::InterpolationOutput tov_out_test =
      tov_instance.tov_solver_for_testing(polyM, central_mass_density, h_final);

  double final_radius{tov_out_full.final_radius()};
  double final_mass{tov_out_full.mass_from_radius(expected_radius())};

  double expectedradius = expected_radius();
  double expectedmass = expected_mass();

  CHECK_ITERABLE_CUSTOM_APPROX(expectedradius, final_radius, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expectedmass, final_mass, custom_approx);

  double intermediate_radius{tov_out_test.final_radius()};
  double intermediate_mass{tov_out_test.mass_from_radius(intermediate_radius)};
  double intermediate_log_enthalpy{
      tov_out_test.log_specific_enthalpy_from_radius(intermediate_radius)};

  double interpolated_mass{tov_out_full.mass_from_radius(intermediate_radius)};
  double interpolated_log_enthalpy{
      tov_out_full.log_specific_enthalpy_from_radius(intermediate_radius)};

  CHECK_ITERABLE_CUSTOM_APPROX(intermediate_mass, interpolated_mass,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(intermediate_log_enthalpy,
                               interpolated_log_enthalpy, custom_approx);
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.EinsteinSolutions.TOV",
    "[Unit][PointwiseFunctions]") {
  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>> polyM =
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(4.3419, 2.0);

  for (size_t i; i < 50; i++) {
    double initial_h = 5.0e-10;
    double final_h = 0.000;
    double step = (final_h - initial_h) / 50.0;

    double final_h_for_integration = initial_h + i * step;
    double central_density = 1.0e-10;

    test_newtonian_tov(polyM, central_density, final_h_for_integration);
  };
}

}  // end namespace
