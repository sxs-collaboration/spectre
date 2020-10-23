// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/LineSearch.hpp"

SPECTRE_TEST_CASE("Unit.ParallelNewtonRaphson.LineSearch",
                  "[Unit][ParallelAlgorithms]") {
  {
    // At the initial globalization step do a _quadratic_ polynomial
    // interpolation and select the minimum
    const double step_length = 0.5;
    const double residual = 1.;
    const double residual_slope = -2.;
    const double next_residual = 1.5;
    const double next_step_length =
        NonlinearSolver::newton_raphson::next_step_length(
            0, step_length, std::numeric_limits<double>::signaling_NaN(),
            residual, residual_slope, next_residual,
            std::numeric_limits<double>::signaling_NaN());
    // The minimum of a quadratic function f(x) that passes through f(0) = 1 and
    // f(0.5) = 1.5 with derivative f'(0) = -2 is at x = 1 / 6
    CHECK(next_step_length == approx(1. / 6.));
  }
  {
    // At subsequent initial globalization steps do a _cubic_ polynomial
    // interpolation and select the minimum
    const double step_length = 0.5;
    const double residual = 1.;
    const double residual_slope = -2.;
    const double next_residual = 1.5;
    const double prev_step_length = 1.;
    const double prev_residual = 2.;
    const double next_step_length =
        NonlinearSolver::newton_raphson::next_step_length(
            1, step_length, prev_step_length, residual, residual_slope,
            next_residual, prev_residual);
    // The minimum of a cubic function f(x) that passes through f(0) = 1,
    // f(0.5) = 1.5 and f(1) = 2 with f'(0) = -2 is at x = 0.12732200375003505
    CHECK(next_step_length == approx(0.12732200375003505));
  }
}
