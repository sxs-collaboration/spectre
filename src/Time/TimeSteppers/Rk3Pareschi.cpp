// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk3Pareschi.hpp"

namespace TimeSteppers {

size_t Rk3Pareschi::order() const { return 3; }

double Rk3Pareschi::stable_step() const { return 1.25637; }

size_t Rk3Pareschi::imex_order() const { return 3; }

size_t Rk3Pareschi::implicit_stage_order() const { return 0; }

namespace {
const double alpha = 0.24169426078821;
const double beta = 0.06042356519705;
const double eta = 0.12915286960590;
}  // namespace

const RungeKutta::ButcherTableau& Rk3Pareschi::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {0.0, 0.0, 1.0, 0.5},
      // Substep coefficients
      {{0.0},
       {0.0},
       {0.0, 0.0, 1.0},
       {0.0, 0.0, 1.0 / 4.0, 1.0 / 4.0}},
      // Result coefficients
      {0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0},
      // Coefficients for the embedded method for generating an error measure.
      //
      // Not given in the reference.  Any set of coefficients of the form
      // (0, 0, x, x, 1 - 2 x) works.
      {0.0, 0.0, 0.25, 0.25, 0.5},
      // Dense output coefficient polynomials
      {{},
       {0.0, 1.0, -1.0},
       {0.0, 0.0, 1.0 / 6.0},
       {0.0, 0.0, 1.0 / 6.0},
       {0.0, 0.0, 2.0 / 3.0}}};
  return tableau;
}

const ImexRungeKutta::ImplicitButcherTableau&
Rk3Pareschi::implicit_butcher_tableau() const {
  static const ImplicitButcherTableau tableau{
      {{0.0, alpha},  // This stage is zeroth-order
       {0.0, -alpha, alpha},
       {0.0, 0.0, 1 - alpha, alpha},
       {0.0, beta, eta, 0.5 - beta - eta - alpha, alpha}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk3Pareschi::my_PUP_ID = 0;  // NOLINT
