// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Time/TimeSteppers/RungeKutta.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "Utilities/Math.hpp"
#include "Utilities/Numeric.hpp"

namespace TestHelpers::RungeKutta {
void check_tableau(const TimeSteppers::RungeKutta::ButcherTableau& tableau,
                   const size_t expected_order,
                   const size_t expected_error_order) {
  const auto& substep_times = tableau.substep_times;
  const auto& substep_coefficients = tableau.substep_coefficients;
  const auto& result_coefficients = tableau.result_coefficients;
  const auto& error_coefficients = tableau.error_coefficients;
  const auto& dense_coefficients = tableau.dense_coefficients;

  const size_t max_substeps =
      std::max(result_coefficients.size(), error_coefficients.size());

  // First substep is implicitly 0 with no coefficients.
  CHECK(substep_times.size() + 1 == max_substeps);
  CHECK(substep_coefficients.size() + 1 == max_substeps);
  // Optional FSAL term at the end
  CHECK((dense_coefficients.size() == result_coefficients.size() or
         dense_coefficients.size() == max_substeps + 1));

  // Results are order 1
  CHECK(alg::accumulate(result_coefficients, 0.0) == approx(1.0));
  CHECK(alg::accumulate(error_coefficients, 0.0) == approx(1.0));

  for (size_t substep = 1; substep < max_substeps; ++substep) {
    const auto& time = substep_times[substep - 1];
    CHECK(time >= 0.0);
    CHECK(time <= 1.0);
    const auto& coefficients = substep_coefficients[substep - 1];
    // Substep is explicit
    CHECK(coefficients.size() <= substep);
    // Substep is order 1
    CHECK(alg::accumulate(coefficients, 0.0) == approx(time));
  }

  for (size_t substep = 0; substep < result_coefficients.size(); ++substep) {
    CHECK(evaluate_polynomial(dense_coefficients[substep], 0.0) == 0.0);
    CHECK(evaluate_polynomial(dense_coefficients[substep], 1.0) ==
          approx(result_coefficients[substep]));
  }
  if (dense_coefficients.size() > result_coefficients.size()) {
    CHECK(not dense_coefficients.back().empty());
    CHECK(evaluate_polynomial(dense_coefficients.back(), 0.0) == 0.0);
    CHECK(evaluate_polynomial(dense_coefficients.back(), 1.0) == approx(0.0));
    for (size_t error_only_substep = result_coefficients.size();
         error_only_substep < dense_coefficients.size() - 1;
         ++error_only_substep) {
      CHECK(dense_coefficients[error_only_substep].empty());
    }
  }

  // Check order for quadrature (RHS depends only on time).
  const auto check_quadrature_order =
      [&substep_times](const std::vector<double>& coefficients,
                       const size_t expected) {
        CAPTURE(coefficients);
        CAPTURE(expected);
        // Don't require the next order to be inconsistent, as the
        // method may do better for quadrature than for an ODE.  Order
        // 0 (i.e., that the stepper is at least first order) was
        // checked above.
        for (size_t order = 1; order < expected; ++order) {
          CAPTURE(order);
          double integral = 0.0;
          for (size_t substep = 1; substep < coefficients.size(); ++substep) {
            integral += coefficients[substep] *
                        std::pow(substep_times[substep - 1], order);
          }
          CHECK(integral == approx(1.0 / (order + 1.0)));
        }
      };
  check_quadrature_order(result_coefficients, expected_order);
  check_quadrature_order(error_coefficients, expected_error_order);
}

void check_tableau(const TimeSteppers::RungeKutta& stepper) {
  check_tableau(stepper.butcher_tableau(), stepper.order(),
                stepper.error_estimate_order());
}
}  // namespace TestHelpers::RungeKutta
