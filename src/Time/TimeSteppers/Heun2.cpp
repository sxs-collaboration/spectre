// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Heun2.hpp"

namespace TimeSteppers {

size_t Heun2::order() const { return 2; }

size_t Heun2::error_estimate_order() const { return 1; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1). It is the same as for forward Euler.
double Heun2::stable_step() const { return 1.0; }

const RungeKutta::ButcherTableau& Heun2::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {1.0},
      // Substep coefficients
      {{1.0}},
      // Result coefficients
      {0.5, 0.5},
      // Coefficients for the embedded method for generating an error measure.
      {1.0, 0.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -0.5},
       {0.0, 0.0, 0.5}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Heun2::my_PUP_ID = 0;  // NOLINT
