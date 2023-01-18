// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk5Tsitouras.hpp"

namespace TimeSteppers {
Rk5Tsitouras::Rk5Tsitouras(CkMigrateMessage* /*msg*/) {}

size_t Rk5Tsitouras::order() const { return 5; }

size_t Rk5Tsitouras::error_estimate_order() const { return 4; }

double Rk5Tsitouras::stable_step() const { return 1.7534234969024887; }

const RungeKutta::ButcherTableau& Rk5Tsitouras::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0},
      // Substep coefficients
      {{0.161},
       {-0.008480655492356992, 0.3354806554923570},
       {2.897153057105495, -6.359448489975075, 4.362295432869581},
       {5.32586482843926, -11.74888356406283, 7.495539342889836,
        -0.09249506636175525},
       {5.86145544294642, -12.92096931784711, 8.159367898576159,
        -0.07158497328140100, -0.02826905039406838},
       {0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
        -3.290069515436081, 2.324710524099774}},
      // Result coefficients
      {0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
       -3.290069515436081, 2.324710524099774},
      // Coefficients for the embedded method for generating an error
      // measure.  The values given in the reference are actually the
      // differences between these and the result coefficients, and
      // the last value has a sign error.
      {0.09824077787029123, 0.010816434459657, 0.4720087724042376,
       1.5237195812770048, -3.872426680888636, 2.7827926300289607, -1.0 / 66.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -2.763706197274826, 2.9132554618219126, -1.0530884977290216},
       {0.0, 0.0, 0.1317, -0.2234, 0.1017},
       {0.0, 0.0, 3.930296236894751, -5.941033872131505, 2.490627285651252793},
       {0.0, 0.0, -12.411077166933676, 30.33818863028232,
        -16.54810288924490272},
       {0.0, 0.0, 37.50931341651104, -88.1789048947664, 47.37952196281928122},
       {0.0, 0.0, -27.89652628919729, 65.09189467479368, -34.87065786149660974},
       {},
       {0.0, 0.0, 1.5, -4.0, 2.5}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk5Tsitouras::my_PUP_ID = 0;  // NOLINT
