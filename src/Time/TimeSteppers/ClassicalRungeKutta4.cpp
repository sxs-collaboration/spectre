// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"

#include <utility>

namespace TimeSteppers {

size_t ClassicalRungeKutta4::order() const { return 4; }

size_t ClassicalRungeKutta4::error_estimate_order() const { return 3; }

// The growth function for RK4 is (e.g. page 60 of
// http://www.staff.science.uu.nl/~frank011/Classes/numwisk/ch10.pdf
//
//   g = 1 + mu + mu^2 / 2 + mu^3 / 6 + mu^4 / 24,
//
// where mu = lambda * dt. The equation dy/dt = -lambda * y evolves
// stably if |g| < 1. For lambda=-2, chosen so the stable_step() for
// RK1 (i.e. forward Euler) would be 1, RK4 has a stable step
// determined by inserting mu->-2 dt into the above equation. Finding the
// solutions with a numerical root find yields a stable step of about 1.39265.
double ClassicalRungeKutta4::stable_step() const { return 1.3926467817026411; }

const RungeKutta::ButcherTableau& ClassicalRungeKutta4::butcher_tableau()
    const {
  // See (17.1.3) of Numerical Recipes 3rd Edition
  static const ButcherTableau tableau{
      // Substep times
      {{1, 2}, {1, 2}, {1}},
      // Substep coefficients
      {{1.0 / 2.0},
       {0.0, 1.0 / 2.0},
       {0.0, 0.0, 1.0}},
      // Result coefficients
      {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},
      // An extra step is needed for error estimation.
      {},
      // Dense output coefficient polynomials.  Numerical Recipes
      // Eq. (17.2.15). This implements cubic interpolation throughout
      // the step.
      {{0.0, 1.0, -3.0 / 2.0, 2.0 / 3.0},
       {0.0, 0.0, 1.0, -2.0 / 3.0},
       {0.0, 0.0, 1.0, -2.0 / 3.0},
       {0.0, 0.0, 1.0 / 2.0, -1.0 / 3.0},
       {0.0, 0.0, -1.0, 1.0}}};
  return tableau;
}

const RungeKutta::ButcherTableau& ClassicalRungeKutta4::error_tableau() const {
  // The embedded Zonneveld 4(3) scheme adds an extra substep at 3/4
  // that is used only by the third-order error estimation scheme
  static const ButcherTableau tableau = [this]() {
    auto error_tableau = butcher_tableau();
    error_tableau.substep_times.emplace_back(3, 4);
    error_tableau.substep_coefficients.push_back(
        {5.0 / 32.0, 7.0 / 32.0, 13.0 / 32.0, -1.0 / 32.0});
    error_tableau.result_coefficients.push_back(0.0);
    error_tableau.error_coefficients =
        {-1.0 / 2.0, 7.0 / 3.0, 7.0 / 3.0, 13.0 / 6.0, -16.0 / 3.0};
    // The extra substep is not used, but the final value is
    // renumbered, so we have to insert an empty coefficient second
    // from the end.
    auto last_coefficient = std::move(error_tableau.dense_coefficients.back());
    error_tableau.dense_coefficients.back().clear();
    error_tableau.dense_coefficients.push_back(std::move(last_coefficient));
    return error_tableau;
  }();
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::ClassicalRungeKutta4::my_PUP_ID = 0;  // NOLINT
