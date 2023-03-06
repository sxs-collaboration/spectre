// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk3Owren.hpp"

namespace TimeSteppers {

size_t Rk3Owren::order() const { return 3; }

size_t Rk3Owren::error_estimate_order() const { return 2; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1).
double Rk3Owren::stable_step() const { return 1.2563726633091645; }

const RungeKutta::ButcherTableau& Rk3Owren::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {12.0 / 23.0, 4.0 / 5.0},
      // Substep coefficients
      {{12.0 / 23.0},
       {-68.0 / 375.0, 368.0 / 375.0}},
      // Result coefficients
      {31.0 / 144.0, 529.0 / 1152.0, 125.0 / 384.0},
      // Coefficients for the embedded method for generating an error measure.
      {1.0 / 24.0, 23.0 / 24.0, 0.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -65.0 / 48.0, 41.0 / 72.0},
       {0.0, 0.0, 529.0 / 384.0, -529.0 / 576.0},
       {0.0, 0.0, 125.0 / 128.0, -125.0 / 192.0},
       {0.0, 0.0, -1.0, 1.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk3Owren::my_PUP_ID = 0;  // NOLINT
