// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests of power monitors.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {

void test_power_monitors_impl() {
  const size_t number_of_points_per_dimension = 4;
  const size_t number_of_points = pow<2>(number_of_points_per_dimension);

  // Test a constant function
  const DataVector test_data_vector = DataVector{number_of_points, 1.0};

  const Mesh<2_st> mesh{number_of_points_per_dimension,
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto test_power_monitors =
      PowerMonitors::power_monitors<2_st>(test_data_vector, mesh);

  // The only non-zero modal coefficient of a constant is the one corresponding
  // to the first Legendre polynomial
  DataVector check_data_vector =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector[0] = 1.0 / sqrt(number_of_points_per_dimension);

  const std::array<DataVector, 2> expected_power_monitors{check_data_vector,
                                                          check_data_vector};

  CHECK_ITERABLE_APPROX(test_power_monitors, expected_power_monitors);
}

void test_power_monitors_second_impl() {
  const size_t number_of_points_per_dimension = 4;

  const Mesh<2_st> mesh{number_of_points_per_dimension,
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto logical_coords = logical_coordinates(mesh);

  // Build a test function containing only one Legendre basis function
  // per dimension
  const size_t x_mode = 0;
  const size_t y_mode = 1;
  const std::array<size_t, 2> coeff = {x_mode, y_mode};

  DataVector u_nodal(mesh.number_of_grid_points(), 1.0);
  for (size_t dim = 0; dim < 2; ++dim) {
    u_nodal *=
        Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
            gsl::at(coeff, dim), logical_coords.get(dim));
  }

  const auto test_power_monitors =
      PowerMonitors::power_monitors<2_st>(u_nodal, mesh);

  // The only non-zero modal coefficient of a constant is the one corresponding
  // to the specified Legendre polynomial

  // In the x direction
  DataVector check_data_vector_x =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector_x[x_mode] = 1.0 / sqrt(number_of_points_per_dimension);

  // In the y direction
  DataVector check_data_vector_y =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector_y[y_mode] = 1.0 / sqrt(number_of_points_per_dimension);

  // We compare against the expected array
  const std::array<DataVector, 2> expected_power_monitors{check_data_vector_x,
                                                          check_data_vector_y};

  CHECK_ITERABLE_APPROX(test_power_monitors, expected_power_monitors);
}

void test_relative_truncation_error_impl() {
  // We recompute the truncation error for a function where we know the
  // power monitors analytically
  const size_t number_of_points_per_dimension = 8;
  const Mesh<1_st> mesh{number_of_points_per_dimension,
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  // Build a test function with no zero power monitors
  const std::vector<int> coeffs = {0, 1, 2, 3, 4, 5, 6, 7};
  DataVector u_nodal(mesh.number_of_grid_points(), 0.0);
  double ampl = 0.0;
  for (auto coeff : coeffs) {
    ampl = pow(10.0, -coeff);
    u_nodal +=
        ampl *
        Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
            static_cast<size_t>(coeff), logical_coords.get(0_st));
  }

  // Compute the relative truncation error
  const int last_coeff = 7;
  double weight = 0.0;
  double avg = 0.0;
  double weight_sum = 0.0;
  for (auto coeff : coeffs) {
    ampl = pow(10.0, -coeff);
    weight = exp(-square(coeff - last_coeff + 0.5));
    avg += log10(ampl) * weight;
    weight_sum += weight;
  }
  avg = avg / weight_sum;
  // By construction the maximum of the magnitude of the first two modes is
  // unity.
  // We test the order of magnitude of the relative error
  const double expected_relative_truncation_error = pow(10.0, avg);

  const auto power_monitors =
      PowerMonitors::power_monitors<1_st>(u_nodal, mesh);
  const DataVector& power_monitor_x = gsl::at(power_monitors, 0_st);
  // We use all of the modes as above
  const double test_relative_truncation_error =
      pow(10.0, -1.0 * PowerMonitors::relative_truncation_error(
                           power_monitor_x, power_monitor_x.size()));

  CHECK_ITERABLE_APPROX(expected_relative_truncation_error,
                        test_relative_truncation_error);

  // Test truncation error
  const double test_truncation_error =
      PowerMonitors::absolute_truncation_error(u_nodal, mesh)[0];

  // Compare with the result from the relative truncation error
  const double expected_truncation_error_x =
      max(abs(u_nodal)) *
      pow(10.0, -1.0 * PowerMonitors::relative_truncation_error(
                           power_monitor_x, power_monitor_x.size()));

  CHECK_ITERABLE_APPROX(test_truncation_error, expected_truncation_error_x);
}

void test_relative_truncation_error_with_symmetry() {
  // Try to resolve half a period of a sinusoid
  const size_t num_modes = 12;
  const Mesh<1> mesh{num_modes, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto xi = Spectral::collocation_points(mesh);
  const double wave_number = 0.5;
  const DataVector u_nodal = sin((xi + 1.) * M_PI * wave_number);
  CAPTURE(u_nodal);
  auto modes = PowerMonitors::power_monitors(u_nodal, mesh)[0];
  // Add some more noise to the modes
  modes += 10. * std::numeric_limits<double>::epsilon();
  CAPTURE(modes);
  const double relative_truncation_error =
      PowerMonitors::relative_truncation_error(modes, num_modes);
  // Last mode should be zero by symmetry
  REQUIRE(modes[num_modes - 1] == approx(0.));
  // Expect the relative truncation error to be the ratio of the first and last
  // nonzero modes
  const double expected_relative_truncation_error =
      log10(modes[0] / modes[num_modes - 2]);
  Approx custom_approx = Approx::custom().epsilon(1e-2).scale(1.);
  CHECK(relative_truncation_error ==
        custom_approx(expected_relative_truncation_error));
}

void test_relative_truncation_error_linear_function() {
  // Resolve a linear function with a few modes. We technically need only 2.
  const auto get_modes = [](const size_t num_modes) {
    const Mesh<1> mesh{num_modes, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto xi = Spectral::collocation_points(mesh);
    const DataVector u_nodal = (xi + 1.) * 0.5;
    auto modes = PowerMonitors::power_monitors(u_nodal, mesh)[0];
    // Add some noise to the modes
    modes += 10. * std::numeric_limits<double>::epsilon();
    const double relative_truncation_error =
        PowerMonitors::relative_truncation_error(modes, num_modes);
    return std::make_pair(modes, relative_truncation_error);
  };
  {
    INFO("2 modes");
    const auto [modes, rel_error_digits] = get_modes(2);
    CAPTURE(modes);
    CHECK_ITERABLE_APPROX(modes, (DataVector{0.5, 0.5}));
    // We don't know for sure that we have resolved the function exactly,
    // because we have two nonzero modes and nothing else.
    CHECK(rel_error_digits == approx(0.));
  }
  {
    INFO("3 modes");
    const auto [modes, rel_error_digits] = get_modes(3);
    CAPTURE(modes);
    CHECK_ITERABLE_APPROX(modes, (DataVector{0.5, 0.5, 0.}));
    // The last mode is zero, but we still don't know if we have resolved the
    // function because the last mode could be zero by symmetry.
    CHECK(rel_error_digits == approx(0.));
  }
  {
    INFO("4 modes");
    const auto [modes, rel_error_digits] = get_modes(4);
    CAPTURE(modes);
    CHECK_ITERABLE_APPROX(modes, (DataVector{0.5, 0.5, 0., 0.}));
    // We have two zero modes, so we know we have resolved the function exactly.
    CHECK(rel_error_digits > 14.);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PowerMonitors",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_power_monitors_impl();
  test_power_monitors_second_impl();
  test_relative_truncation_error_impl();
  test_relative_truncation_error_with_symmetry();
  test_relative_truncation_error_linear_function();
}
